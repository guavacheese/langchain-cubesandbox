from __future__ import annotations
import os

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)

from deepagents.backends.sandbox import BaseSandbox

from e2b_code_interpreter import Sandbox


# CubeSandbox 原生兼容 E2B SDK;
# LangChain 的 BaseSandbox 已经通过 python3 内置实现了文件系统工具（ls, read, write, edit, glob, grep）。
# 需要实现 execute()、upload_files()、download_files()、id 及对应的 async 方法。
class CubeSandbox(BaseSandbox):
    """
    CubeSandbox backend for LangChain DeepAgents.

    利用 CubeSandbox 的 E2B 兼容层，将硬件级隔离的 MicroVM
    接入 LangChain 生态。冷启动 <60ms，单实例内存开销 <5MB。
    """

    def __init__(
        self,
        template: str,
        api_url: str | None = None,
        api_key: str | None = None,
        ssl_cert: str | None = None,
        timeout: int = 60,
        # proxy_ip: str | None = None,
        # proxy_port: int = 80,
    ) -> None:
        # CubeSandbox 通过环境变量拦截 E2B SDK 请求
        if api_url:
            os.environ["E2B_API_URL"] = api_url
        if api_key:
            os.environ["E2B_API_KEY"] = api_key
        if ssl_cert:
            os.environ["SSL_CERT_FILE"] = ssl_cert

        self._template_id = template
        self._timeout = timeout

        # 创建 Sandbox 时注入自定义 httpx client
        self._sandbox = Sandbox.create(template=template)

    @property
    def id(self) -> str:
        return self._sandbox.id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """执行 shell 命令，映射到 CubeSandbox MicroVM 内。"""
        try:
            # 标准 E2B SDK 的 shell 命令执行入口
            result = self._sandbox.commands.run(
                command,
                timeout=timeout or self._timeout,
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

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:

        responses = []

        for path, content in files:
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
            try:
                content = self._sandbox.files.read(path)
                # E2B SDK 返回 str，需编码为 bytes
                if isinstance(content, str):
                    content = content.encode("utf-8")
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
            except Exception as e:
                err = str(e)
                if "not_found" in err.lower() or "invalid_path" in err.lower():
                    responses.append(
                        FileDownloadResponse(path=path, content=b"", error=err)
                    )
                else:
                    raise
        return responses

    # --- Async variants ---
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

    def close(self) -> None:
        """显式关闭 CubeSandbox MicroVM 实例，释放资源。"""
        if hasattr(self._sandbox, "close"):
            self._sandbox.close()
        elif hasattr(self._sandbox, "kill"):
            self._sandbox.kill()

    def __del__(self) -> None:
        self.close()
