"""SwiGLU backward shared by both expert-backward schedules."""

import tilelang
import tilelang.language as T


@tilelang.jit
def swiglu_backward(rows, width):
    @T.prim_func
    def kernel(
        gate_up: T.Tensor((rows, 2 * width), "bfloat16"),
        hidden_gradient: T.Tensor((rows, width), "bfloat16"),
        gate_up_gradient: T.Tensor((rows, 2 * width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * width, 256), threads=128) as block:
            for i in T.Parallel(256):
                index = block * 256 + i
                if index < rows * width:
                    row, col = index // width, index % width
                    gate = T.cast(gate_up[row, col], "float32")
                    up = T.cast(gate_up[row, col + width], "float32")
                    dh = T.cast(hidden_gradient[row, col], "float32")
                    sigmoid = 1.0 / (1.0 + T.exp(-gate))
                    gate_up_gradient[row, col] = dh * up * sigmoid * (1 + gate * (1 - sigmoid))
                    gate_up_gradient[row, col + width] = dh * gate * sigmoid

    return kernel
