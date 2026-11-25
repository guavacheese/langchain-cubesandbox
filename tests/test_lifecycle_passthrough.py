"""P0 lifecycle 透传单测：pause-on-idle 策略必须从调用方一路传到创建端点。

背景（2026-11-25 资源分档方案 P0）：
  CubeAPI（SandboxLifecycleConfig）与 e2b SDK ≥2.50 都支持创建时传
  lifecycle={"on_timeout": "pause", "auto_resume": bool}——TTL 到期 pause
  而非 kill，配额随之可被调度器回收。本仓是透传层，要钉死四件事：

1. __init__：lifecycle 有值 → create_kwargs 带 lifecycle；None → **键都不出现**
   （出现 lifecycle=None 是行为变更信号，必须红）；
2. get_or_create 三条分支：running 复用 / paused-resume / 新建，全部透传不吞；
3. _rebuild：回收重建必须沿用 self._lifecycle——漏传 = pause 策略静默降级回
   服务端默认 kill，且线上无从察觉（这类"重建丢配置"在 _template 上踩过一次，
   见 sandbox.py connect() 的 2026-08-19 注释）；
4. 实例属性 _lifecycle 在 __init__/connect 两条构造路径都存在（预初始化纪律，
   防 __del__ 路径踩空）。

不连 MicroVM：Sandbox.create/connect/list 全部桩掉，断言 kwargs 形态。
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

pytest.importorskip("e2b_code_interpreter", reason="e2b SDK not available")

from langchain_cubesandbox import sandbox as sbmod  # noqa: E402
from langchain_cubesandbox.sandbox import CubeSandbox  # noqa: E402

LC = {"on_timeout": "pause", "auto_resume": True}


class _FakeSbx:
    """最小沙箱替身：只暴露被保存/被读的属性。"""

    def __init__(self, sandbox_id: str = "sbx-fake"):
        self.sandbox_id = sandbox_id


class _Recorder:
    """记录 create/connect 的调用 kwargs。"""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        # connect 是位置参（Sandbox.connect(sandbox_id, timeout=...)），
        # create 全 kwargs —— 两种形态都记录下来
        self.calls.append({"_args": args, **kwargs})
        return _FakeSbx()


@pytest.fixture()
def stub(monkeypatch):
    rec_create = _Recorder()
    rec_connect = _Recorder()
    monkeypatch.setattr(sbmod.Sandbox, "create", staticmethod(rec_create))
    monkeypatch.setattr(sbmod.Sandbox, "connect", staticmethod(rec_connect))
    return rec_create, rec_connect


# ── 1. __init__ ────────────────────────────────────────────────────────────

def test_init_passes_lifecycle(stub):
    rec_create, _ = stub
    CubeSandbox(template="tpl", timeout=600, lifecycle=dict(LC))
    assert len(rec_create.calls) == 1
    assert rec_create.calls[0]["lifecycle"] == LC


def test_init_without_lifecycle_omits_key(stub):
    """回归护栏：默认路径必须与改造前**逐字节一致**——create_kwargs 里
    连 lifecycle 键都不允许出现（传 lifecycle=None 给服务端是行为变更）。"""
    rec_create, _ = stub
    CubeSandbox(template="tpl", timeout=600)
    assert "lifecycle" not in rec_create.calls[0]


def test_init_stores_lifecycle_attr(stub):
    _, _ = stub
    sbx = CubeSandbox(template="tpl", lifecycle=dict(LC))
    assert sbx._lifecycle == LC
    sbx2 = CubeSandbox(template="tpl")
    assert sbx2._lifecycle is None


# ── 2. get_or_create 三分支 ───────────────────────────────────────────────

@pytest.fixture()
def no_existing(monkeypatch):
    """让 get_or_create 的 running/paused 查找都落空 → 走新建分支。"""
    monkeypatch.setattr(CubeSandbox, "list", staticmethod(lambda **kw: []))


def test_get_or_create_create_branch(stub, no_existing):
    rec_create, _ = stub
    CubeSandbox.get_or_create(
        template="tpl", thread_id="GY001:s1", timeout=600, lifecycle=dict(LC)
    )
    assert rec_create.calls[0]["lifecycle"] == LC


def test_get_or_create_running_branch_keeps_lifecycle(stub, monkeypatch):
    """running 复用走 connect：lifecycle 不进 create 请求（沙箱策略已随
    建箱落在服务端），但必须存进实例——该实例日后 _rebuild 靠它重建。"""
    _, rec_connect = stub
    monkeypatch.setattr(
        CubeSandbox, "list",
        staticmethod(lambda **kw: [{"id": "sbx-run"}] if kw.get("state") == "running" else []),
    )
    sbx = CubeSandbox.get_or_create(
        template="tpl", thread_id="GY001:s1", lifecycle=dict(LC)
    )
    assert len(rec_connect.calls) == 1
    # 下发给 SDK 的只有 id/timeout（策略已随建箱落在服务端）；
    # 透传保证 = 实例属性带住了它，供 _rebuild 沿用
    assert sbx._lifecycle == LC


def test_get_or_create_paused_branch_keeps_lifecycle(stub, monkeypatch):
    """paused 命中 → connect + resume；lifecycle 同样必须保存（P0 的主场景
    正是"pause 后被下一轮 get_or_create 从这里捞回来"）。

    2026-11-25 起 resume() 有真实 wire 语义（e2b 2.50 用 connect 端点唤醒，
    见 test_resume_reconnects_and_swaps_handle），所以本分支会看到**两次**
    connect：第一次构造句柄、第二次唤醒。断言钉住这个形态——退化成一次
    意味着唤醒丢了（paused 沙箱捞回来还是死的，P0 主场景直接破产）。"""
    _, rec_connect = stub

    def _list(**kw):
        if kw.get("state") == "paused":
            return [{"id": "sbx-paused"}]
        return []

    monkeypatch.setattr(CubeSandbox, "list", staticmethod(_list))
    sbx = CubeSandbox.get_or_create(
        template="tpl", thread_id="GY001:s1", lifecycle=dict(LC)
    )
    assert len(rec_connect.calls) == 2, (
        "paused 分支必须是 connect(构造)+connect(唤醒) 两次，got %d"
        % len(rec_connect.calls))
    # 第一次按 paused 列表命中的 id 构造句柄；第二次 resume 按**当前句柄**
    # 的 id 唤醒（桩句柄的 id 是替身默认值，真实实现传 self._sandbox.sandbox_id）
    assert rec_connect.calls[0]["_args"] == ("sbx-paused",)
    assert len(rec_connect.calls[1]["_args"]) == 1
    assert sbx._lifecycle == LC


# ── 3. _rebuild 沿用 ──────────────────────────────────────────────────────

def test_rebuild_preserves_lifecycle(stub, no_existing):
    """钉死"重建不丢 pause 策略"。这是 P0 最容易静默回归的一条腿：
    线上只有沙箱被回收过一次才走到这里，漏传的表现是"pause 用着用着
    变回 kill"，日志上完全不可见。"""
    rec_create, _ = stub
    sbx = CubeSandbox(
        template="tpl", timeout=600, lifecycle=dict(LC),
        metadata={"thread_id": "GY001:s1"},
    )
    rec_create.calls.clear()
    assert sbx._rebuild() is True
    assert len(rec_create.calls) == 1
    assert rec_create.calls[0]["lifecycle"] == LC
    assert rec_create.calls[0]["template"] == "tpl"


def test_rebuild_without_lifecycle_omits_key(stub, no_existing):
    """旧行为兼容：没配 pause 的实例重建后也**不得**多出 lifecycle 键。"""
    rec_create, _ = stub
    sbx = CubeSandbox(template="tpl", timeout=600, metadata={"thread_id": "GY001:s1"})
    rec_create.calls.clear()
    # __init__ 分支不自动带 thread_id metadata，_rebuild 依赖 metadata 里的
    # thread_id —— 上面构造已显式给出
    assert sbx._rebuild() is True
    assert "lifecycle" not in rec_create.calls[0]


# ── 4. connect 构造路径属性完整 ───────────────────────────────────────────

def test_connect_stores_lifecycle_attr(stub):
    _, _ = stub
    sbx = CubeSandbox.connect(sandbox_id="sbx-x", template="tpl", lifecycle=dict(LC))
    assert sbx._lifecycle == LC
    sbx2 = CubeSandbox.connect(sandbox_id="sbx-x", template="tpl")
    assert sbx2._lifecycle is None


# ── 5. resume() 真实语义（2026-11-25 真平台 e2e 抓出的第二个 bug）──────────

def test_resume_reconnects_and_swaps_handle(stub):
    """e2b SDK 2.50 没有 Sandbox.resume()（生产容器 hasattr 实测 false）。
    旧实现 self._sandbox.resume() 在真唤醒 paused 沙箱时 AttributeError——
    get_or_create 里被 except pass 吞掉才没炸穿。正确姿势=类方法
    Sandbox.connect(id)（connect 端点即 resume），且句柄必须换新对象。"""
    _, rec_connect = stub
    sbx = CubeSandbox(template="tpl", lifecycle=dict(LC))
    old_handle = sbx._sandbox
    rec_connect.calls.clear()
    sbx.resume()
    assert len(rec_connect.calls) == 1, "resume 必须且只发一次 connect"
    call = rec_connect.calls[0]
    assert call["_args"] == (old_handle.sandbox_id,), (
        "connect 必须以现有 sandbox_id 为位置参（SDK connect 签名）")
    assert sbx._sandbox is not old_handle, "唤醒后句柄必须替换为新连接对象"


# ── 6. 接线契约（AST）：三条分支的调用点都必须真的写了 lifecycle= ────────
# 行为测试可能被分支组合漏检（如某分支 fake 数据没触发），接线断言直接数
# 调用点——这是本仓判据纪律：新增透传必须逐分支点名。

def _kwarg_names(call: ast.Call) -> set[str]:
    return {kw.arg for kw in call.keywords if kw.arg}


def test_get_or_create_all_branches_wire_lifecycle():
    # getsource 保留类体缩进，必须 dedent 后再 parse（strip 只动首行，
    # 第二行 `    def ...` 依旧 IndentationError——实测踩过的坑）
    src = textwrap.dedent(inspect.getsource(CubeSandbox.get_or_create))
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and getattr(n.func.value, "id", "") == "CubeSandbox"
        and n.func.attr in ("connect",)
    ]
    creates = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "CubeSandbox"
    ]
    # 两条 connect 分支（running / paused）+ 一条新建分支，一个都不能漏
    assert len(calls) == 2, f"expected 2 CubeSandbox.connect call sites, got {len(calls)}"
    assert len(creates) == 1, f"expected 1 CubeSandbox(...) call site, got {len(creates)}"
    for c in calls + creates:
        assert "lifecycle" in _kwarg_names(c), ast.dump(c)[:120]
