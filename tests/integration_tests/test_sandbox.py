from __future__ import annotations
from collections.abc import Iterator
import os
import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_cubesandbox import CubeSandbox


class TestCubeSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        backend = CubeSandbox(
            template_id=os.environ["CUBE_TEMPLATE_ID"],
            api_url=os.environ.get("CUBE_API_URL", "http://192.168.10.136:13000"),
            api_key=os.environ.get("CUBE_API_KEY", "dummy"),
            ssl_cert=os.environ.get("CUBE_SSL_CERT"),
        )
        try:
            yield backend
        finally:
            backend.close()
