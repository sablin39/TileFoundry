"""``T.cuda`` — CUDA instruction declarations."""

from __future__ import annotations

from tilefoundry.ir.tir.cuda.nn.sm80_mma import Mma
from tilefoundry.ir.tir.cuda.nn.sm89_mma import Mma as Sm89Mma
from tilefoundry.ir.tir.cuda.nn.wgmma import Form, Major, Wgmma


class _Sm80Namespace:
    Mma = Mma


class _Sm89Namespace:
    Mma = Sm89Mma


class _Sm90Namespace:
    Wgmma = Wgmma
    Form = Form
    Major = Major


class _CudaNamespace:
    """``T.cuda`` — compile-time CUDA instruction declarations."""

    sm80 = _Sm80Namespace()
    sm89 = _Sm89Namespace()
    sm90 = _Sm90Namespace()


cuda = _CudaNamespace()

PLATFORM_NAMESPACES = {"cuda": cuda}

__all__ = ["PLATFORM_NAMESPACES", "cuda"]
