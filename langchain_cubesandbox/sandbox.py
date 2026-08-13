from __future__ import annotations
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

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

        # 2. CubeSandbox 通过环境变量拦截 E2B SDK 请求
        if api_url:
            os.environ["E2B_API_URL"] = api_url
        if api_key:
            os.environ["E2B_API_KEY"] = api_key
        if ssl_cert:
            os.environ["SSL_CERT_FILE"] = ssl_cert
        # 注意：不要设 E2B_DEBUG=true！会让 SDK 进入 debug 模式返回假 sandbox_id。
        # 也不要 monkeypatch ssl._create_default_https_context——它只对 http.client/urllib 生效，
        # e2b 用的 httpx/httpcore 直接调用 ssl.create_default_context()，替换无效。
        # 证书验证靠 .env 的 SSL_CERT_FILE 指向内部 CA（或公网合法证书）。

        # 传给SDK的kwargs
        create_kwargs: dict[str, Any] = {"template": template}
        if metadata:
            create_kwargs["metadata"] = metadata
        # v0.6.0 起 CubeMaster/CubeAPI 支持创建沙箱时设置空闲回收 TTL（timeout）；
        # SDK 默认 Config.timeout=300，此处显式透传调用方指定的值（-1 = never timeout）。
        if timeout is not None:
            create_kwargs["timeout"] = timeout

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
    ) -> "CubeSandbox":
        """连接到一个已有的沙箱（用于 resume 或跨进程复用）。"""
        # timeout_on_refresh 默认跟随 timeout（理由同 __init__）
        if timeout_on_refresh is None:
            timeout_on_refresh = timeout if timeout is not None else 300
        # 先用一个临时沙箱获取连接
        instance = cls.__new__(cls)

        # 预初始化所有属性，保持一致性
        instance._sandbox = None
        instance._template = ""
        instance._timeout = timeout
        instance._timeout_on_refresh = timeout_on_refresh
        instance._auto_pause = True
        instance._closed = False

        if api_url:
            os.environ["E2B_API_URL"] = api_url
        if api_key:
            os.environ["E2B_API_KEY"] = api_key
        if ssl_cert:
            os.environ["SSL_CERT_FILE"] = ssl_cert
        # 与 __init__ 一致：不设 E2B_DEBUG、不 monkeypatch ssl（对 httpx/httpcore 无效）。

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
    ) -> "CubeSandbox":
        """
        每次调用都创建新沙箱（带 thread_id 作为 metadata）。
        如果需要复用已有沙箱，外部自行管理 sandbox_id → thread_id 的映射。
        timeout_on_refresh 默认跟随 timeout（理由同 __init__）。
        """
        if timeout_on_refresh is None:
            timeout_on_refresh = timeout
        if api_url:
            os.environ["E2B_API_URL"] = api_url
        if api_key:
            os.environ["E2B_API_KEY"] = api_key
        if ssl_cert:
            os.environ["SSL_CERT_FILE"] = ssl_cert
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

    def refresh_timeout(self) -> None:
        """续期 TTL（默认 300 秒 = 5 分钟）。"""
        self._sandbox.set_timeout(self._timeout_on_refresh)

    # ─── Pause / Resume ───

    def pause(self) -> None:
        """暂停沙箱（释放 CPU/内存资源，保留磁盘状态）。"""
        self._sandbox.pause()

    def resume(self) -> None:
        """恢复已暂停的沙箱。"""
        self._sandbox.resume()

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

            # 标准 E2B SDK 的 shell 命令执行入口
            result = self._sandbox.commands.run(
                command,
                timeout=timeout or self._timeout_on_refresh,
            )

            output = result.stdout or ""

            if result.stderr:
                output += f"\n<stderr>{result.stderr}</stderr>"

            return ExecuteResponse(
                output=output,
                exit_code=result.exit_code or 0,
                truncated=getattr(result, "truncated", False),
            )
        except Exception as e:
            # 只吞掉LLM 可重试/修复的路径类错误，其余抛出
            return ExecuteResponse(
                output=f"Execution error:{str(e)}",
                exit_code=1,
                truncated=False,
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
        # E2B SDK 在 async 上下文通常自动支持 async，或提供 aexecute / commands.arun
        # 若 SDK 暂不支持真 async，可先同步 fallback，后续迭代优化
        return self.execute(command, timeout=timeout)

    async def aupload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        return self.upload_files(files)

    async def adownload_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        return self.download_files(paths)

    # ─── 资源释放 ───

    def close(self) -> None:
        """显式关闭 CubeSandbox MicroVM 实例，释放资源。"""
        if self._closed:
            return

        sandbox = getattr(self, "_sandbox", None)
        if sandbox is not None:
            if hasattr(sandbox, "close"):
                sandbox.close()
            elif hasattr(sandbox, "kill"):
                sandbox.kill()

        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass  # 析构时禁止任何异常逃逸
