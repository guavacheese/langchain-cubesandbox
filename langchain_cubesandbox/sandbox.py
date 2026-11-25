from __future__ import annotations
import asyncio
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ─── e2b CA 局部注入（2026-09-09 根治全局 SSL_CERT_FILE 污染）────────────────
# 旧实现把 ssl_cert 写进 os.environ["SSL_CERT_FILE"]（进程级全局变量），污染同进程
# 其他 httpx 客户端：后端 ChatOpenAI/openai 访问公网 api.deepseek.com 时被内网 mkcert
# CA 校验 → CERTIFICATE_VERIFY_FAILED（Linux 上 SSL_CERT_FILE 一设即只信该文件，
# 系统正规 CA 全被忽略）。
# 为什么不能用"C 方案（建连后恢复 env）"：e2b transport 是惰性创建（首次真实请求
# 才 new）+ 线程级缓存（e2b/api/client_sync/__init__.py get_transport →
# TransportWithLogger._thread_local），首个请求发生在恢复之后 → e2b 自己连内网 Cube
# 会退回系统 CA 而失败。所以 env 必须在进程生命周期内持续有效 = 必然全局污染。
# 根治：给 e2b 的 transport 类打子类补丁，构造时强制 verify=「仅含内网 CA」的
# SSLContext——只影响 e2b 自身连接（控制面 + envd 数据面），不再碰全局 env。
# patch 类而非函数的原因：e2b 各子模块（commands/filesystem/paginator/template/volume）
# 以 `from e2b.api.client_sync import get_envd_transport` 形式固化绑定函数，
# patch 模块函数对它们无效；但它们最终 new 的都是模块全局 transport 类 → 类级 patch
# 全局生效。
_e2b_ca_patched = False
_e2b_ca_lock = threading.Lock()


def _install_e2b_ca(ssl_cert: str | None) -> None:
    """让 e2b 的 HTTP transport 只信任 ssl_cert 指向的内网 CA（局部，非全局 env）。

    幂等：进程内只 patch 一次（加锁 + 标志位）。ssl_cert 为空/文件不存在 → 跳过
    （e2b 保持 verify=True 默认，与旧实现 `if ssl_cert:` 不触发时行为一致）。
    e2b 版本升级导致类名不匹配 → warning 提示，不阻断（此时 e2b 回落默认校验，
    不写全局 env，不产生污染）。
    """
    global _e2b_ca_patched
    if _e2b_ca_patched:
        return
    if not ssl_cert or not os.path.isfile(ssl_cert):
        return
    try:
        import ssl

        import e2b.api.client_async as _e2b_async
        import e2b.api.client_sync as _e2b_sync

        ctx = ssl.create_default_context(cafile=ssl_cert)

        def _verify_subclass(base, ca_ctx: ssl.SSLContext):
            class _VerifyTransport(base):
                def __init__(self, *args, **kwargs):
                    super().__init__(verify=ca_ctx, *args, **kwargs)

            _VerifyTransport.__name__ = f"{base.__name__}WithLocalCA"
            return _VerifyTransport

        with _e2b_ca_lock:
            if _e2b_ca_patched:
                return
            patched: list[str] = []
            for _mod, _names in (
                (_e2b_sync, ("TransportWithLogger", "EnvdTransportWithLogger")),
                (
                    _e2b_async,
                    ("AsyncTransportWithLogger", "AsyncEnvdTransportWithLogger"),
                ),
            ):
                for _cls_name in _names:
                    _orig = getattr(_mod, _cls_name, None)
                    if _orig is not None:
                        setattr(_mod, _cls_name, _verify_subclass(_orig, ctx))
                        patched.append(f"{_mod.__name__}.{_cls_name}")
            _e2b_ca_patched = True
            if patched:
                logger.info("e2b transport CA 已局部注入 (%s): %s", ", ".join(patched), ssl_cert)
    except Exception as e:  # noqa: BLE001
        # 不抛：patch 失败仅意味着 e2b 回落默认证书校验（不写全局 env，无污染）
        logger.warning("e2b CA 局部注入失败（回落默认校验，未写全局 env）: %s", e)

# ─── lifecycle wire 兼容层（2026-11-25 真平台 e2e 实锤的静默失效）───────────
# 现象：Sandbox.create(lifecycle={"on_timeout":"pause"}) 建出的沙箱 TTL 到期
# 仍被 kill，pause 配置全程无声无效。
# 根因（两侧源码+抓包铁证）：e2b SDK（2.50 线）把 lifecycle 序列化成**平铺**
# 字段 {"autoPause": true, "autoResume": {"enabled": bool}}；而 CubeAPI 的
# NewSandbox（CubeAPI/src/models/mod.rs:172-184）声明的是**嵌套**对象
# lifecycle:{onTimeout,autoResume}，serde 对未声明的平铺字段静默忽略 →
# 策略根本没进 CubeMaster。SDK 透传单测（test_lifecycle_passthrough）桩掉了
# create，断言的是 kwargs 形态，看不到 wire 层——正是这类"透传≠落地"盲区。
# 修复落点：CubeSandbox 自有层（用户明令不 fork CubeSandbox/e2b 源码，防
# 升级不兼容）。patch 点选 NewSandbox.to_dict——e2b 生成式 API client 的
# body 序列化唯一出口，spike（2026-11-25，agent 容器内 E2B_API_URL 指向
# 桩服务器抓真实请求体）实证补丁后 wire 携带嵌套 lifecycle 且平铺字段清除。
# 语义：autoPause=true → lifecycle:{onTimeout:"pause", autoResume:enabled}；
# autoPause=false/缺省 → **不加** lifecycle 键（缺省即 kill，与改造前 wire
# 语义等价，未开 pause 的流量形态最小变化）。
_e2b_wire_patched = False
_e2b_wire_lock = threading.Lock()


def _install_lifecycle_wire_compat() -> None:
    """把 lifecycle 的 SDK 平铺 wire 形态翻译成 CubeAPI 认的嵌套形态。

    幂等（锁+标志位+函数属性三重）。e2b 升级导致类/方法名变化 → error 日志
    （不抛）：此时行为回落到"pause 静默失效"＝本 patch 之前的已知现状，
    不产生新故障面，但 P0 依赖方必须能从 error 日志立刻察觉。
    """
    global _e2b_wire_patched
    if _e2b_wire_patched:
        return
    try:
        from e2b.api.client.models.new_sandbox import NewSandbox

        with _e2b_wire_lock:
            if _e2b_wire_patched:
                return
            if getattr(NewSandbox.to_dict, "_geesun_wire_compat", False):
                # 极端情形：模块重载但模块级标志位复位——函数属性仍在即视为已打
                _e2b_wire_patched = True
                return
            orig = NewSandbox.to_dict

            def patched(self):  # noqa: ANN001
                d = orig(self)
                ap = d.pop("autoPause", None)
                d.pop("autoPauseMemory", None)
                ar = d.pop("autoResume", None)
                if ap is True:
                    enabled = bool(ar.get("enabled")) if isinstance(ar, dict) else False
                    d["lifecycle"] = {"onTimeout": "pause", "autoResume": enabled}
                return d

            patched._geesun_wire_compat = True  # type: ignore[attr-defined]
            NewSandbox.to_dict = patched
            _e2b_wire_patched = True
            logger.info("e2b lifecycle wire 兼容层已注入（autoPause 平铺 → lifecycle 嵌套）")
    except Exception as e:  # noqa: BLE001
        # error 级：pause 是 P0 能力，静默回退成 kill 必须可见，不允许 warning 淹没
        logger.error("e2b lifecycle wire 兼容层注入失败（pause 将静默失效，回退 kill）: %s", e)

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)

from deepagents.backends.sandbox import BaseSandbox

from e2b_code_interpreter import (
    Sandbox,
    SandboxQuery,
    SandboxState,
)

try:
    # e2b 对「命令非零退出」抛这个异常，而不是返回带 exit_code 的结果对象。
    # 它**同时携带** stdout / stderr / exit_code 三个字段。
    from e2b.sandbox.commands.command_handle import CommandExitException
except ImportError:  # pragma: no cover - 老版本 e2b 无此异常
    CommandExitException = ()  # type: ignore[assignment]


def _metadata_may_mismatch(value: str) -> bool:
    """CubeAPI filter_by_metadata 不做 URL-decode。
    当 value 含 URL 特殊字符（:, %, &, = 等）时，服务端过滤可能漏掉。"""
    return any(c in value for c in (":", "%", "&", "="))


# CubeSandbox 原生兼容 E2B SDK;
# LangChain 的 BaseSandbox 已经通过 python3 内置实现了文件系统工具（ls, read, write, edit, glob, grep）。
# 需要实现 execute()、upload_files()、download_files()、id 及对应的 async 方法。
class CubeSandbox(BaseSandbox):
    """
    CubeSandbox backend for LangChain DeepAgents.

    利用 CubeSandbox 的 E2B 兼容层，将硬件级隔离的 MicroVM
    接入 LangChain 生态。冷启动 <60ms，单实例内存开销 <5MB。

    支持：
    - metadata（存 thread_id 等上下文）
    - timeout / 续期
    - pause / resume
    - get-or-create 模式复用沙箱
    """

    def __init__(
        self,
        template: str,
        api_url: str | None = None,
        api_key: str | None = None,
        ssl_cert: str | None = None,
        metadata: dict[str, str] | None = None,
        timeout: int | None = None,
        auto_pause: bool = True,
        timeout_on_refresh: int | None = None,
        lifecycle: dict[str, Any] | None = None,
    ) -> None:
        # timeout_on_refresh 默认跟随 timeout（否则 refresh_timeout 会用默认 300
        # 把创建时的 timeout 覆盖掉——2026-08-13 实测：创建 timeout=3600，
        # 第一次 execute 前 refresh 就 set_timeout(300)，TTL 被打回 5 分钟）
        if timeout_on_refresh is None:
            timeout_on_refresh = timeout if timeout is not None else 300
        # 1. 预初始化所有属性，防止 __del__ 崩溃
        self._sandbox = None
        self._template = template
        self._timeout = timeout
        self._timeout_on_refresh = timeout_on_refresh
        self._auto_pause = auto_pause
        self._closed = False
        # 保存 metadata（含 thread_id），供沙箱被回收后自动重建（130404）使用
        self._metadata = dict(metadata or {})
        # 保存 lifecycle（2026-11-25 资源分档 P0）：{"on_timeout": "pause",
        # "auto_resume": bool}——TTL 到期 pause 而非 kill。_rebuild 重建时沿用，
        # 否则回收重建一次就把 pause 策略静默降级回 kill。
        self._lifecycle = dict(lifecycle) if lifecycle else None

        # 2. CubeSandbox 通过环境变量拦截 E2B SDK 请求（E2B_API_URL/E2B_API_KEY
        #    为 e2b 专属变量，其他库不读，无污染；仅保留这两个）
        if api_url:
            os.environ["E2B_API_URL"] = api_url
        if api_key:
            os.environ["E2B_API_KEY"] = api_key
        # 3. 内网 CA 只注入 e2b transport（局部 SSLContext）——不再写全局
        #    SSL_CERT_FILE（2026-09-09 根治：全局 env 会污染同进程 openai/httpx
        #    访问公网 API，见 _install_e2b_ca 注释）
        _install_e2b_ca(ssl_cert)
        # 4. lifecycle wire 兼容层（2026-11-25）：e2b SDK 把 lifecycle 平铺成
        #    autoPause 发出，CubeAPI 只认嵌套 lifecycle 对象——不翻译则 pause
        #    静默失效（真平台抓包实锤）。必须在任何 Sandbox.create 之前注入。
        _install_lifecycle_wire_compat()
        # 注意：不要设 E2B_DEBUG=true！会让 SDK 进入 debug 模式返回假 sandbox_id。
        # 也不要 monkeypatch ssl._create_default_https_context——它只对 http.client/urllib 生效，
        # e2b 用的 httpx/httpcore 直接调用 ssl.create_default_context()，替换无效。
        # 证书验证：e2b transport 由 _install_e2b_ca 局部注入内部 CA（或公网合法证书）。

        # 传给SDK的kwargs
        create_kwargs: dict[str, Any] = {"template": template}
        if metadata:
            create_kwargs["metadata"] = metadata
        # v0.6.0 起 CubeMaster/CubeAPI 支持创建沙箱时设置空闲回收 TTL（timeout）；
        # SDK 默认 Config.timeout=300，此处显式透传调用方指定的值（-1 = never timeout）。
        if timeout is not None:
            create_kwargs["timeout"] = timeout
        # lifecycle：SDK 侧收 {"on_timeout": "pause", "auto_resume": bool}，
        # 但它在 wire 层发的是平铺 autoPause，CubeAPI 只认嵌套 lifecycle——
        # 由 _install_lifecycle_wire_compat 在 to_dict 出口翻译（见其注释）。
        # None 时不传，保持服务端默认 kill 行为不变。
        if lifecycle:
            create_kwargs["lifecycle"] = lifecycle

        self._sandbox = Sandbox.create(**create_kwargs)

    @classmethod
    def connect(
        cls,
        sandbox_id: str,
        api_url: str | None = None,
        api_key: str | None = None,
        ssl_cert: str | None = None,
        timeout: int | None = None,
        timeout_on_refresh: int | None = None,
        metadata: dict[str, str] | None = None,
        template: str | None = None,
        lifecycle: dict[str, Any] | None = None,
    ) -> "CubeSandbox":
        """连接到一个已有的沙箱（用于 resume 或跨进程复用）。"""
        # timeout_on_refresh 默认跟随 timeout（理由同 __init__）
        if timeout_on_refresh is None:
            timeout_on_refresh = timeout if timeout is not None else 300
        # 先用一个临时沙箱获取连接
        instance = cls.__new__(cls)

        # 预初始化所有属性，保持一致性
        instance._sandbox = None
        # 注意：template 必须保存——沙箱被 TTL 回收后 _rebuild 依赖它重建
        # （2026-08-19 实测：get_or_create 复用 running 旧沙箱走 connect 分支，
        #  _template 空 → 130404 后 rebuild skipped '缺 template' → execute 5 连败）
        instance._template = template or ""
        instance._timeout = timeout
        instance._timeout_on_refresh = timeout_on_refresh
        instance._auto_pause = True
        instance._closed = False
        # 保存 metadata（含 thread_id），供沙箱被回收后自动重建（130404）使用
        instance._metadata = dict(metadata or {})
        # 同 __init__：pause 策略随实例保存，_rebuild 重建时不静默丢（P0）
        instance._lifecycle = dict(lifecycle) if lifecycle else None

        if api_url:
            os.environ["E2B_API_URL"] = api_url
        if api_key:
            os.environ["E2B_API_KEY"] = api_key
        _install_e2b_ca(ssl_cert)
        # 与 __init__ 一致：不设 E2B_DEBUG、不 monkeypatch ssl（对 httpx/httpcore 无效）。
        # 证书验证走 _install_e2b_ca 的局部 transport 注入。

        try:
            instance._sandbox = Sandbox.connect(sandbox_id, timeout=timeout)
            return instance
        except Exception:
            # 如果失败，清理半成品对象
            instance._closed = True
            raise

    @classmethod
    def list(
        cls,
        metadata: dict[str, str] | None = None,
        state: str | list[str] | None = None,
    ) -> list[dict]:

        query_kwargs = {}
        if metadata:
            query_kwargs["metadata"] = metadata
        if state:
            if isinstance(state, str):
                query_kwargs["state"] = [SandboxState(state)]
            else:
                query_kwargs["state"] = [SandboxState(s) for s in state]

        query = SandboxQuery(**query_kwargs) if query_kwargs else None
        paginator = Sandbox.list(query=query)

        items = []
        while paginator.has_next:
            items.extend(paginator.next_items())

        return [
            {
                "id": s.sandbox_id,
                "metadata": s.metadata,
                "state": s.state.value,
                "started_at": str(s.started_at),
                "end_at": str(s.end_at),
            }
            for s in items
        ]

    @staticmethod
    def get_or_create(
        template: str,
        thread_id: str,
        api_url: str | None = None,
        api_key: str | None = None,
        ssl_cert: str | None = None,
        timeout: int = 300,
        timeout_on_refresh: int | None = None,
        lifecycle: dict[str, Any] | None = None,
    ) -> "CubeSandbox":
        """
        每次调用都创建新沙箱（带 thread_id 作为 metadata）。
        如果需要复用已有沙箱，外部自行管理 sandbox_id → thread_id 的映射。
        timeout_on_refresh 默认跟随 timeout（理由同 __init__）。
        lifecycle（P0，2026-11-25）：{"on_timeout": "pause", "auto_resume": bool}
        透传给创建端点；复用/paused-resume 分支仅保存供 _rebuild 沿用。
        """
        if timeout_on_refresh is None:
            timeout_on_refresh = timeout
        if api_url:
            os.environ["E2B_API_URL"] = api_url
        if api_key:
            os.environ["E2B_API_KEY"] = api_key
        _install_e2b_ca(ssl_cert)
        # 与 __init__/connect 一致：不设 E2B_DEBUG、不 monkeypatch ssl（对 httpx/httpcore 无效）。

        # 先查找已有的沙箱 —— 通过 metadata filter
        #   CubeAPI 的 filter_by_metadata 不会 URL-decode metadata value，
        #   当 thread_id 含 `:` 等特殊字符时可能查不到。加了 client-side fallback。
        def _find_by_metadata(state_filter: str | list[str] = "running") -> list[dict]:
            try:
                return CubeSandbox.list(
                    metadata={"thread_id": thread_id},
                    state=state_filter,
                )
            except Exception:
                return []

        def _find_client_side(state_filter: str | list[str] = "running") -> list[dict]:
            """回退：列出所有沙箱，在客户端按 thread_id 过滤。"""
            try:
                all_sbs = CubeSandbox.list(state=state_filter)
            except Exception:
                return []
            return [
                sb
                for sb in all_sbs
                if (sb.get("metadata") or {}).get("thread_id") == thread_id
            ]

        # Step 1: 先找 running 的
        existing = _find_by_metadata("running")

        # 服务端 metadata 过滤有空窗期 → client-side 兜底
        if not existing and _metadata_may_mismatch(thread_id):
            existing = _find_client_side("running")

        if existing:
            # 复用已有沙箱
            sandbox_id = existing[0]["id"]
            return CubeSandbox.connect(
                sandbox_id=sandbox_id,
                api_url=api_url,
                api_key=api_key,
                ssl_cert=ssl_cert,
                timeout=timeout,
                timeout_on_refresh=timeout_on_refresh,
                metadata={"thread_id": thread_id},
                template=template,
                lifecycle=lifecycle,
            )

        # Step 2: running 没找到，查 paused 沙箱
        paused = _find_by_metadata("paused")
        if not paused and _metadata_may_mismatch(thread_id):
            paused = _find_client_side("paused")

        if paused:
            sandbox_id = paused[0]["id"]
            try:
                sb = CubeSandbox.connect(
                    sandbox_id=sandbox_id,
                    api_url=api_url,
                    api_key=api_key,
                    ssl_cert=ssl_cert,
                    timeout=timeout,
                    timeout_on_refresh=timeout_on_refresh,
                    metadata={"thread_id": thread_id},
                    template=template,
                    lifecycle=lifecycle,
                )
                # 尝试恢复暂停的沙箱（如果 connect 成功但沙箱仍处于 paused 状态）
                try:
                    sb.resume()
                except Exception:
                    pass
                return sb
            except Exception:
                # connect/resume 失败，fall through 创建新的
                pass

        # Step 3: 都没有，创建新的
        return CubeSandbox(
            template=template,
            api_url=api_url,
            api_key=api_key,
            ssl_cert=ssl_cert,
            metadata={"thread_id": thread_id},
            timeout=timeout,
            timeout_on_refresh=timeout_on_refresh,
            lifecycle=lifecycle,
        )

    @property
    def id(self) -> str:
        return self._sandbox.sandbox_id

    @property
    def sandbox_id(self) -> str:
        """别名，方便调试。"""
        return self._sandbox.sandbox_id

    # ─── TTL 管理 ───

    def set_timeout(self, seconds: int) -> None:
        """设置超时时间（秒），超时后沙箱自动销毁。

        v0.6.0 起 CubeMaster 实现了 POST /sandboxes/:id/timeout 接口，
        续期调用真正下发到后端（重置空闲时钟 + 重设 TTL）。
        """
        self._timeout = seconds

        try:
            self._sandbox.set_timeout(seconds)
        except Exception as e:
            logger.warning("set_timeout failed: seconds=%s err=%s", seconds, e)

    def _is_sandbox_missing(self, exc: Exception) -> bool:
        """判断异常是否为"沙箱不存在"（CubeMaster 130404 / SDK not found）。"""
        s = str(exc)
        return "130404" in s or "sandbox id not found" in s.lower()

    def _rebuild(self) -> bool:
        """沙箱被回收/误删（130404）时自动重建，替换底层连接。返回是否成功。

        沙箱是无状态计算容器（状态在宿主 /uploads、/reports、DB），
        丢失后重建成本秒级——上层任务不应因沙箱回收而中断。
        """
        tid = (self._metadata or {}).get("thread_id")
        if not tid or not self._template:
            logger.warning("sandbox rebuild skipped: 缺 template/thread_id")
            return False
        try:
            new = CubeSandbox.get_or_create(
                template=self._template,
                thread_id=tid,
                timeout=self._timeout,
                timeout_on_refresh=self._timeout_on_refresh,
                # pause 策略必须随重建沿用（P0）：漏传 = 回收重建一次就把
                # on_timeout=pause 静默降级回服务端默认 kill，且无从察觉。
                lifecycle=self._lifecycle,
            )
            self._sandbox = new._sandbox
            self._closed = False
            logger.warning("sandbox(thread=%s) 已回收，已自动重建", tid)
            return True
        except Exception as e:
            logger.warning("sandbox rebuild failed: %s", e)
            return False

    def refresh_timeout(self) -> None:
        """续期 TTL（默认 300 秒 = 5 分钟）；沙箱不存在时自动重建后续期。"""
        try:
            self._sandbox.set_timeout(self._timeout_on_refresh)
        except Exception as e:
            if self._is_sandbox_missing(e) and self._rebuild():
                # 重建后重试一次续期
                self._sandbox.set_timeout(self._timeout_on_refresh)
            else:
                raise

    # ─── Pause / Resume ───

    def pause(self) -> None:
        """暂停沙箱（释放 CPU/内存资源，保留磁盘状态）。"""
        self._sandbox.pause()

    def resume(self) -> None:
        """恢复已暂停的沙箱。

        ⚠ e2b SDK 2.50（本仓 pin 的版本）**没有 Sandbox.resume()**——生产
        容器 hasattr 实测 + 2026-11-25 真平台 e2e 唤醒 paused 沙箱时当场
        AttributeError（get_or_create 里被 except pass 吞掉才一直没暴露）。
        唤醒语义由类方法 connect 承担：POST /sandboxes/:id/connect 把
        paused→running 并重设 TTL（真机 raw HTTP 实证 connect 后 state
        =running 且 commands.run 正常）。
        句柄必须换成新连接对象：pause 之前的旧句柄不能继续用（恢复后
        envd 会话状态已换）。
        """
        self._sandbox = Sandbox.connect(
            self._sandbox.sandbox_id,
            timeout=self._timeout_on_refresh,
        )

    def is_alive(self) -> bool:
        """探活：沙箱当前是否 running（阻塞 ~1 次数据面 HTTP）。

        背景（2026-11-25 geesun_test e2e 实测）：pause 模式下平台把 TTL 到期的
        箱转 paused，但上层进程级缓存（geesun_agent `_sandbox_cache`）仍持有旧
        句柄——不探活直接复用会让 execute 报 `service unavailable`（箱明明可
        唤醒，agent 却抱着死句柄）。调用方对 False 应逐出缓存改走 get_or_create
        （其 paused 分支会 connect 唤醒，running 分支原样复用，不存在误重建）。

        ⚠ paused 时 SDK `is_running()` 的行为不是返回 False 而是**抛
        SandboxException 503**（SDK 源码只对 502 返回 False；CubeAPI 数据面
        代理回的是 503——agent 容器内 spike 实证）。故异常一律判 False。
        网络抖动误判 False 无副作用：get_or_create 查到 running 仍复用原箱。
        """
        try:
            return bool(self._sandbox.is_running())
        except Exception as e:  # noqa: BLE001 —— paused/已销毁/网络故障都按非活处理
            logger.debug("is_alive 探活未通过（按非活处理）: %s", e)
            return False

    # ─── 代码执行 ───

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """执行 shell 命令，映射到 CubeSandbox MicroVM 内。"""
        try:
            # 尝试续期 TTL，失败也不影响代码执行
            try:
                self.refresh_timeout()
            except Exception as e:
                logger.warning("refresh_timeout failed: %s", e)

            # 标准 E2B SDK 的 shell 命令执行入口；沙箱不存在时自动重建并重试一次
            try:
                result = self._sandbox.commands.run(
                    command,
                    timeout=timeout or self._timeout_on_refresh,
                )
            except CommandExitException as e:
                # 命令「非零退出」是**正常结果**，不是沙箱故障。
                # e2b 的 CommandExitException.__str__ 只输出 stderr，
                # 而很多失败原因写在 stdout（如 deepagents write_file preflight 的
                # "Error: File already exists: '<path>'"）—— 原实现落到最外层
                # except 后只剩 "Execution error:Command exited with code 1 and error:"，
                # stdout 全丢，调用方（乃至 LLM）拿不到任何可诊断信息。
                # 这里还原 stdout/stderr/exit_code 三件套。
                return self._response_from_exit(e.stdout, e.stderr, e.exit_code)
            except Exception as e:
                if self._is_sandbox_missing(e) and self._rebuild():
                    logger.warning("execute: 沙箱已回收，已自动重建，重试命令")
                    result = self._sandbox.commands.run(
                        command,
                        timeout=timeout or self._timeout_on_refresh,
                    )
                else:
                    raise

            return self._response_from_exit(
                result.stdout,
                result.stderr,
                result.exit_code or 0,
                truncated=getattr(result, "truncated", False),
            )
        except Exception as e:
            # 只吞掉LLM 可重试/修复的路径类错误，其余抛出
            return ExecuteResponse(
                output=f"Execution error:{str(e)}",
                exit_code=1,
                truncated=False,
            )

    @staticmethod
    def _response_from_exit(
        stdout: str | None,
        stderr: str | None,
        exit_code: int | None,
        *,
        truncated: bool = False,
    ) -> ExecuteResponse:
        """把 (stdout, stderr, exit_code) 组装成 ExecuteResponse，并保证 output 非空。

        对齐 deer-flow `community/e2b_sandbox/e2b_sandbox.py:155-168`
        （boxlite / opensandbox / tenki 四个后端同一写法）：
          1. stdout 与 stderr 都有 → 换行拼接；只有一个 → 取它；
          2. 非零退出且**确实没有输出** → 补 "Command exited with code N"；
          3. 仍为空 → 补 "(no output)"。
        第 2/3 条保证调用方永远拿不到空字符串——空输出会让上层误判为「什么都没发生」。
        """
        output = stdout or ""
        if stderr:
            output += f"\n<stderr>{stderr}</stderr>"
        output = output.strip()
        if exit_code not in (0, None) and not output:
            output = f"Command exited with code {exit_code}"
        return ExecuteResponse(
            output=output or "(no output)",
            exit_code=exit_code or 0,
            truncated=truncated,
        )

    def run_code(self, code: str, *, timeout: int | None = None):
        """运行 Python等 代码（直接转 E2B SDK 的 run_code）。"""
        try:
            self.refresh_timeout()
        except Exception:
            pass
        return self._sandbox.run_code(code, timeout=timeout)

    # ─── 文件操作 ───

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:

        responses = []

        for path, content in files:
            # 新增：拒绝相对路径
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                self._sandbox.files.write(
                    path,
                    content,
                )

                responses.append(
                    FileUploadResponse(
                        path=path,
                        error=None,
                    )
                )
            except Exception as e:
                err = str(e)
                if "not_found" in err.lower() or "invalid_path" in err.lower():
                    responses.append(FileUploadResponse(path=path, error=err))
                else:
                    raise
        return responses

    def download_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:

        responses = []
        for path in paths:
            # 新增：拒绝相对路径
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
                continue
            try:
                # 优先用 bytes 格式读取（二进制安全，也能读文本）
                content = self._sandbox.files.read(path, format="bytes")
                # E2B SDK 返回 str，需编码为 bytes
                if isinstance(content, str):
                    content = content.encode("utf-8")
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
            except Exception as e:
                err = str(e)
                # 按异常类型映射到标准错误码
                err_lower = err.lower()
                if (
                    "does not exist" in err_lower
                    or "not found" in err_lower
                    or "no such file" in err_lower
                ):
                    responses.append(
                        FileDownloadResponse(
                            path=path, content=None, error="file_not_found"
                        )
                    )
                elif "is a directory" in err_lower:
                    responses.append(
                        FileDownloadResponse(
                            path=path, content=None, error="is_directory"
                        )
                    )
                elif "permission denied" in err_lower:
                    responses.append(
                        FileDownloadResponse(
                            path=path, content=None, error="permission_denied"
                        )
                    )
                elif "invalid" in err_lower or "relative" in err_lower:
                    responses.append(
                        FileDownloadResponse(
                            path=path, content=None, error="invalid_path"
                        )
                    )
                else:
                    raise
        return responses

    # ─── Async ───

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """异步版 execute：**必须走线程池**，不得同步直调。

        2026-10-10 修复（生产 P0）：本方法原实现是
        ``return self.execute(command, timeout=timeout)`` —— 在 ``async def`` 里
        同步调用阻塞的 e2b SDK，**把整个事件循环占死**。调用链是
        deepagents 的 execute 工具 → ``CompositeBackend.aexecute``
        （deepagents/backends/composite.py:588）→ 本方法。agent 服务是单个
        uvicorn worker（单事件循环），于是一条长命令（execute 的 timeout 上限
        600s）会冻住**全部会话，连 /healthz 都不响应** —— swarm healthcheck
        （30s×5）随后判 unhealthy 并发 SIGKILL（`task: non-zero exit (137):
        dockerexec: unhealthy container`），重启后又冻，形成重启循环。
        现场实证：容器内裸 TCP 连 127.0.0.1:8009 仅 3ms 成功，而同进程
        ``GET /healthz`` 连续 3 次 8.008s 超时（端口在听/accept 正常，
        但请求进不了事件循环）；时间线与「模型决定调 execute 跑 150 页 OCR」
        严格吻合。

        注意这不是"可选的优化"：deepagents 上游 ``BackendProtocol.aexecute``
        的默认实现（deepagents/backends/protocol.py:860）本来就是
        ``await asyncio.to_thread(self.execute, ...)``；本类覆写默认实现时丢掉
        了 ``to_thread`` —— 即与上游语义的偏差，而非取舍。

        线程安全依据（2026-10-10 实测，10 PASS / 0 FAIL）：同一 CubeSandbox
        对象 8 路线程并发 ``execute``（各带唯一标记）全部 exit_code=0、无异常、
        **无串台**，总耗时 3.26s vs 串行估算 25.64s；并发 ``execute`` +
        ``upload_files`` 混合亦全绿，风暴后沙箱仍可正常 execute。故**不需要**
        per-sandbox 锁（加锁会白白牺牲同一沙箱内的并发度）。
        """
        return await asyncio.to_thread(self.execute, command, timeout=timeout)

    async def aupload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """异步版 upload_files：同上，必须走线程池（大文件上传同样是长阻塞）。"""
        return await asyncio.to_thread(self.upload_files, files)

    async def adownload_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        """异步版 download_files：同上，必须走线程池。"""
        return await asyncio.to_thread(self.download_files, paths)

    # ─── 资源释放 ───

    def close(self) -> None:
        """关闭连接（不销毁沙箱）：仅标记已关闭，沙箱生命周期由平台空闲 TTL 回收。

        2026-08-14 修复：原实现在对象被 GC 时经 __del__ → close() → e2b close()/kill()
        直接 DELETE 沙箱，导致沙箱在活跃使用中被误杀（execute 全 504）。现在 close 只
        断开本实例引用，沙箱本体保留供 get_or_create 复用或由平台 TTL 自然回收。
        """
        self._closed = True

    def destroy(self) -> None:
        """显式销毁沙箱（调用方明确要求时才使用）：kill 掉 MicroVM。"""
        if self._closed:
            return
        sandbox = getattr(self, "_sandbox", None)
        if sandbox is not None:
            try:
                if hasattr(sandbox, "kill"):
                    sandbox.kill()
                elif hasattr(sandbox, "close"):
                    sandbox.close()
            except Exception as e:
                logger.warning("destroy failed: %s", e)
        self._closed = True

    def __del__(self) -> None:
        # 仅标记关闭，不销毁沙箱——对象被 GC 不应导致沙箱被删
        # （2026-08-14 实测：__del__ 调 close → e2b kill → 活跃沙箱被 DELETE，后续 execute 全 504）
        try:
            self._closed = True
        except Exception:
            pass  # 析构时禁止任何异常逃逸
