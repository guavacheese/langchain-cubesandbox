"""防线 4 单测：execute() 必须还原非零退出的 stdout。

背景（生产事故 GY24428:5f65f8df）：
  `write_file` 二次写同一路径 → deepagents preflight `sys.exit(1)` →
  e2b 抛 CommandExitException（携带 stdout/stderr/exit_code）→
  旧实现落到最外层 except，只取 str(e)，而 __str__ 仅输出 stderr
  ⇒ 模型收到零信息 "Execution error:Command exited with code 1 and error:"，
  误判为「脚本太长」，绕道 heredoc 内联 execute。

本测试**不连 MicroVM**：用真实的 e2b CommandExitException 类 + 假 sandbox 触发，
断言 execute() 的分支行为。断言源是真实 SDK 类型，不是复刻实现。
"""

from __future__ import annotations

import pytest

e2b_exc = pytest.importorskip(
    "e2b.sandbox.commands.command_handle",
    reason="e2b SDK not available",
)
CommandExitException = e2b_exc.CommandExitException

from langchain_cubesandbox.sandbox import CubeSandbox  # noqa: E402


class _FakeCommands:
    """模拟 e2b commands.run：按脚本抛异常或返回结果对象。"""

    def __init__(self, mode: str, *, stdout: str = "", stderr: str = "", exit_code: int = 0):
        self.mode = mode
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code

    def run(self, command, timeout=None):  # noqa: ARG002
        if self.mode == "raise_exit":
            raise CommandExitException(
                stdout=self.stdout,
                stderr=self.stderr,
                exit_code=self.exit_code,
                error=None,
            )
        if self.mode == "raise_other":
            raise RuntimeError("connection reset by peer")

        class _R:
            pass

        r = _R()
        r.stdout = self.stdout
        r.stderr = self.stderr
        r.exit_code = self.exit_code
        r.truncated = False
        return r


class _FakeInner:
    def __init__(self, commands):
        self.commands = commands


def _sandbox(commands: _FakeCommands) -> CubeSandbox:
    """构造一个只替换了 _sandbox.commands 的 CubeSandbox（不触网、不建 MicroVM）。"""
    sb = CubeSandbox.__new__(CubeSandbox)
    sb._sandbox = _FakeInner(commands)
    sb._closed = False
    sb._metadata = {}
    sb._template = None
    sb._timeout = 300
    sb._timeout_on_refresh = 300
    sb.refresh_timeout = lambda: None  # type: ignore[method-assign]
    sb._is_sandbox_missing = lambda exc: False  # type: ignore[method-assign]
    sb._rebuild = lambda: False  # type: ignore[method-assign]
    return sb


# deepagents _WRITE_CHECK_TEMPLATE 在"文件已存在"时的真实输出
PREFLIGHT_STDOUT = "Error: File already exists: '/home/user/extract_light_params.py'\n"


class TestExecuteRestoresStdoutOnNonZeroExit:
    """核心回归：非零退出必须还原 stdout（防线 4 的本体）。"""

    def test_preflight_stdout_recovered(self):
        """write_file preflight 的失败原因必须回到调用方。"""
        sb = _sandbox(_FakeCommands("raise_exit", stdout=PREFLIGHT_STDOUT, stderr="", exit_code=1))
        res = sb.execute("preflight")
        assert "already exists" in res.output, res.output
        assert "/home/user/extract_light_params.py" in res.output, res.output
        assert res.exit_code == 1

    def test_not_labelled_as_execution_error(self):
        """非零退出是命令的正常结果，不得再谎报为沙箱异常。"""
        sb = _sandbox(_FakeCommands("raise_exit", stdout=PREFLIGHT_STDOUT, exit_code=1))
        res = sb.execute("preflight")
        assert "Execution error" not in res.output, res.output

    def test_stdout_and_stderr_both_kept(self):
        """两边都有内容时都要保留（stderr 用 <stderr> 包裹）。"""
        sb = _sandbox(
            _FakeCommands("raise_exit", stdout="out-line\n", stderr="err-line\n", exit_code=2)
        )
        res = sb.execute("x")
        assert "out-line" in res.output
        assert "err-line" in res.output
        assert res.exit_code == 2


class TestExecuteNeverReturnsEmptyOutput:
    """兜底：任何分支都不得返回空 output（对齐 deer-flow）。"""

    def test_nonzero_exit_without_output_gets_message(self):
        sb = _sandbox(_FakeCommands("raise_exit", stdout="", stderr="", exit_code=3))
        res = sb.execute("x")
        assert res.output == "Command exited with code 3"

    def test_success_without_output_gets_placeholder(self):
        sb = _sandbox(_FakeCommands("ok", stdout="", stderr="", exit_code=0))
        res = sb.execute("x")
        assert res.output == "(no output)"


class TestRealSandboxFailureStillReported:
    """反向闸门：新分支不得把真实故障误吞成"正常非零退出"。"""

    def test_other_exception_still_execution_error(self):
        sb = _sandbox(_FakeCommands("raise_other"))
        res = sb.execute("x")
        assert "Execution error" in res.output
        assert "connection reset by peer" in res.output
        assert res.exit_code == 1

    def test_success_path_unchanged(self):
        """成功路径行为不变（回归保护）。"""
        sb = _sandbox(_FakeCommands("ok", stdout="hello\n", stderr="", exit_code=0))
        res = sb.execute("echo hello")
        assert "hello" in res.output
        assert "Execution error" not in res.output
        assert res.exit_code == 0


class TestPreventsRegression:
    """反证：若把 CommandExitException 分支删掉，判据必须失败。"""

    def test_str_alone_loses_stdout(self):
        """证明根因是 __str__ 只输出 stderr——异常对象本身带着 stdout。"""
        exc = CommandExitException(stdout=PREFLIGHT_STDOUT, stderr="", exit_code=1, error=None)
        assert exc.stdout != "", "异常对象应携带 stdout"
        assert "already exists" not in str(exc), "str() 不含 stdout —— 这正是根因"
