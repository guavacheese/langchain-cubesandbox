"""lifecycle wire 兼容层单测（2026-11-25 真平台 e2e 实锤的静默失效修复）。

背景：e2b SDK（2.50 线）把 lifecycle 序列化成**平铺** {"autoPause": true,
"autoResume": {"enabled": bool}}；CubeAPI 的 NewSandbox 只认**嵌套**
lifecycle:{onTimeout,autoResume}，平铺字段被 serde 静默忽略 → pause 全程
无效、TTL 到期照旧 kill（agent 容器内抓真实请求体实锤）。修复是把
NewSandbox.to_dict 出口做平铺→嵌套翻译（_install_lifecycle_wire_compat）。

判据四件套：
1. 真实 e2b NewSandbox 过 to_dict：autoPause=true → 嵌套 lifecycle 出现、
   平铺三键清除；autoResume.enabled 如实映射（True/False 两支都要钉）；
2. 未开 pause（autoPause false/缺省）→ **不得**多出 lifecycle 键（旧流量
   wire 语义零变化，缺省=kill）；
3. 幂等：install 两次不叠加异常行为，标志位/函数属性完整；
4. 接线（AST）：__init__ 里 install 必须出现在 Sandbox.create 之前——
   晚注入 = 首个沙箱漏翻译，正是这类"顺序型接线"最容易静默回归。
外加：注入失败路径必须 error 级日志透出（不许 warning 淹没 P0 失效）。
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap

import pytest

pytest.importorskip("e2b_code_interpreter", reason="e2b SDK not available")

from langchain_cubesandbox import sandbox as sbmod  # noqa: E402
from langchain_cubesandbox.sandbox import CubeSandbox  # noqa: E402

try:
    from e2b.api.client.models.new_sandbox import NewSandbox
    from e2b.api.client.models.sandbox_auto_resume_config import (
        SandboxAutoResumeConfig,
    )
except ImportError:  # pragma: no cover - e2b 内部结构变了，兼容层本身会 error
    NewSandbox = None
    SandboxAutoResumeConfig = None

needs_models = pytest.mark.skipif(NewSandbox is None, reason="e2b NewSandbox model moved")


@pytest.fixture()
def wire_patched():
    """安装兼容层并还原现场（防污染同会话其它测试）。"""
    to_dict_before = NewSandbox.to_dict
    flag_before = sbmod._e2b_wire_patched
    sbmod._install_lifecycle_wire_compat()
    yield
    NewSandbox.to_dict = to_dict_before
    sbmod._e2b_wire_patched = flag_before


def _new_sandbox(**kw):
    # 生成式模型的必填参数是 templateID（python 名 template_id）
    return NewSandbox(template_id="tpl-x", **kw)


# ── 1. 翻译正确性 ──────────────────────────────────────────────────────────

@needs_models
def test_auto_pause_true_becomes_nested_lifecycle(wire_patched):
    d = _new_sandbox(
        auto_pause=True,
        auto_resume=SandboxAutoResumeConfig(enabled=False),
    ).to_dict()
    assert d.get("lifecycle") == {"onTimeout": "pause", "autoResume": False}
    for gone in ("autoPause", "autoPauseMemory", "autoResume"):
        assert gone not in d, f"平铺键 {gone} 必须被清除（CubeAPI 会无视它）"


@needs_models
def test_auto_resume_true_maps_through(wire_patched):
    d = _new_sandbox(
        auto_pause=True,
        auto_resume=SandboxAutoResumeConfig(enabled=True),
    ).to_dict()
    assert d["lifecycle"] == {"onTimeout": "pause", "autoResume": True}


@needs_models
def test_auto_pause_true_without_resume_defaults_false(wire_patched):
    """autoPause=true 但没带 autoResume 结构 → autoResume 必须落 False，
    不得因缺省而抛或漏键。"""
    d = _new_sandbox(auto_pause=True).to_dict()
    assert d["lifecycle"] == {"onTimeout": "pause", "autoResume": False}


# ── 2. 未开 pause 的流量零变化 ────────────────────────────────────────────

@needs_models
def test_auto_pause_false_adds_no_lifecycle(wire_patched):
    """autoPause=false（SDK 生成式模型的常值默认）→ 不加 lifecycle 键。
    缺省语义=kill，与改造前 wire 等价；多出 {"onTimeout":"kill"} 虽然
    服务端同义，但违反"未开启分档/pause 的流量形态最小变化"承诺。"""
    d = _new_sandbox(auto_pause=False).to_dict()
    assert "lifecycle" not in d


# ── 3. 幂等 ────────────────────────────────────────────────────────────────

@needs_models
def test_install_idempotent(wire_patched):
    first = NewSandbox.to_dict
    sbmod._install_lifecycle_wire_compat()  # 第二次必须原地返回
    assert NewSandbox.to_dict is first
    assert getattr(NewSandbox.to_dict, "_geesun_wire_compat", False) is True
    # 行为不受重复 install 影响
    d = _new_sandbox(auto_pause=True).to_dict()
    assert d["lifecycle"] == {"onTimeout": "pause", "autoResume": False}


# ── 4. 接线契约（AST）：install 在 Sandbox.create 之前 ────────────────────

def test_wire_compat_installed_before_first_create():
    """钉死顺序：__init__ 里 _install_lifecycle_wire_compat() 的行号必须
    小于 Sandbox.create( 的行号。晚注入=进程内第一个沙箱漏翻译——
    功能测试很难复现（第二次调用才正常），只有接线断言抓得住。"""
    src = textwrap.dedent(inspect.getsource(CubeSandbox.__init__))
    tree = ast.parse(src)
    install_lines = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_install_lifecycle_wire_compat"
    ]
    create_lines = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "Sandbox"
        and n.func.attr == "create"
    ]
    assert install_lines, "CubeSandbox.__init__ 必须调用 _install_lifecycle_wire_compat()"
    assert create_lines, "Sandbox.create 调用点丢了？（透传测试的既有前提）"
    assert min(install_lines) < min(create_lines), (
        f"install 行 {min(install_lines)} 必须早于 create 行 {min(create_lines)}"
    )


# ── 5. 注入失败路径：error 级日志、不抛 ──────────────────────────────────

def test_install_failure_logs_error_and_does_not_raise(monkeypatch, caplog):
    """e2b 升级把 NewSandbox 挪走 → import 失败：必须 error 级（P0 静默
    回退 kill 必须可见），且不得抛异常杀建箱主链路。"""
    import builtins

    real_import = builtins.__import__

    def _fail_new_sandbox(name, *a, **kw):
        if name.endswith("new_sandbox"):
            raise ImportError("simulated e2b upgrade moved NewSandbox")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fail_new_sandbox)
    monkeypatch.setattr(sbmod, "_e2b_wire_patched", False)
    # 清掉可能的模块缓存路径依赖不是必须的：失败点就在 import 行
    with caplog.at_level(logging.ERROR, logger="langchain_cubesandbox.sandbox"):
        sbmod._install_lifecycle_wire_compat()  # 不得抛
    assert any(
        r.levelno == logging.ERROR and "wire" in r.getMessage()
        for r in caplog.records
    ), "注入失败必须 error 级透出（warning 会淹没 P0 失效）"
    assert sbmod._e2b_wire_patched is False, "失败不得置成功标志位"
