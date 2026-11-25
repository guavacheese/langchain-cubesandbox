# -*- coding: utf-8 -*-
"""CubeSandbox.is_alive() 单测（2026-11-25 geesun_test e2e 暴露的缓存死句柄修复配套）。

判据来源是真机 spike：paused 时 SDK is_running() 抛 SandboxException 503，
而不是返回 False——is_alive 必须把异常路径也归一为 False。
"""
from unittest.mock import MagicMock

from langchain_cubesandbox.sandbox import CubeSandbox


def _make(inner):
    sb = CubeSandbox.__new__(CubeSandbox)
    sb._sandbox = inner
    return sb


def test_is_alive_true_when_running():
    inner = MagicMock()
    inner.is_running.return_value = True
    assert _make(inner).is_alive() is True


def test_is_alive_false_when_sdk_returns_false():
    inner = MagicMock()
    inner.is_running.return_value = False
    assert _make(inner).is_alive() is False


def test_is_alive_false_when_paused_raises_503():
    """真机形态：CubeAPI 数据面对 paused 箱回 503，SDK 直接抛异常。"""
    inner = MagicMock()
    inner.is_running.side_effect = RuntimeError('503: {"error":"service unavailable"}')
    assert _make(inner).is_alive() is False


def test_is_alive_swallows_all_exceptions():
    inner = MagicMock()
    inner.is_running.side_effect = Exception("network flap")
    assert _make(inner).is_alive() is False


def test_is_alive_does_not_mutate_handle():
    """探活绝不能顺手换句柄（唤醒是 resume/connect 的职责，不是探活的）。"""
    inner = MagicMock()
    inner.is_running.return_value = True
    sb = _make(inner)
    sb.is_alive()
    assert sb._sandbox is inner
