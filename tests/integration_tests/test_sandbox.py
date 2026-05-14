from __future__ import annotations
from collections.abc import Iterator
from pathlib import Path
import os
import pytest
from dotenv import load_dotenv
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_cubesandbox import CubeSandbox

# 只加载项目目录下的 .env.test
env_path = Path(__file__).resolve().parent.parent.parent / ".env.test"
load_dotenv(env_path)


class TestCubeSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        backend = CubeSandbox(
            template=os.environ["CUBE_TEMPLATE_ID"],
            api_url=os.environ.get("CUBE_API_URL", "http://192.168.10.136:13000"),
            api_key=os.environ.get("CUBE_API_KEY", "dummy"),
            ssl_cert=os.environ.get("CUBE_SSL_CERT"),
        )
        try:
            yield backend
        finally:
            backend.close()
