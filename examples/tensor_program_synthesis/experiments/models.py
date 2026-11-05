"""Small authored HIR candidates for tensor lifetime and residual experiments."""

from tilefoundry.dsl import Mesh, Tensor, Topology, func, module, tf
from tilefoundry_nvidia import CudaTarget


@module(
    entry="forward_save_h",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class ResidualMLP:
    """Supplied forward/backward formulas with three explicit residual contracts."""

    @func
    def forward_save_zh(
        x: Tensor[(8, 16), "f32"],
        w: Tensor[(16, 32), "f32"],
        v: Tensor[(32, 8), "f32"],
    ):
        z = tf.matmul(x, w)
        h = tf.relu(z)
        y = tf.matmul(h, v)
        return y, z, h

    @func
    def forward_save_h(
        x: Tensor[(8, 16), "f32"],
        w: Tensor[(16, 32), "f32"],
        v: Tensor[(32, 8), "f32"],
    ):
        z = tf.matmul(x, w)
        h = tf.relu(z)
        y = tf.matmul(h, v)
        return y, h

    @func
    def forward_recompute(
        x: Tensor[(8, 16), "f32"],
        w: Tensor[(16, 32), "f32"],
        v: Tensor[(32, 8), "f32"],
    ):
        z = tf.matmul(x, w)
        h = tf.relu(z)
        return tf.matmul(h, v)

    @func
    def backward_save_zh(
        x: Tensor[(8, 16), "f32"],
        w: Tensor[(16, 32), "f32"],
        v: Tensor[(32, 8), "f32"],
        dy: Tensor[(8, 8), "f32"],
        z: Tensor[(8, 32), "f32"],
        h: Tensor[(8, 32), "f32"],
    ):
        dv = tf.matmul(tf.transpose(h, perm=(1, 0)), dy)
        dh = tf.matmul(dy, tf.transpose(v, perm=(1, 0)))
        dz = dh * tf.cast(z > 0.0, dtype="f32")
        dw = tf.matmul(tf.transpose(x, perm=(1, 0)), dz)
        dx = tf.matmul(dz, tf.transpose(w, perm=(1, 0)))
        return dx, dw, dv

    @func
    def backward_save_h(
        x: Tensor[(8, 16), "f32"],
        w: Tensor[(16, 32), "f32"],
        v: Tensor[(32, 8), "f32"],
        dy: Tensor[(8, 8), "f32"],
        h: Tensor[(8, 32), "f32"],
    ):
        dv = tf.matmul(tf.transpose(h, perm=(1, 0)), dy)
        dh = tf.matmul(dy, tf.transpose(v, perm=(1, 0)))
        dz = dh * tf.cast(h > 0.0, dtype="f32")
        dw = tf.matmul(tf.transpose(x, perm=(1, 0)), dz)
        dx = tf.matmul(dz, tf.transpose(w, perm=(1, 0)))
        return dx, dw, dv

    @func
    def backward_recompute(
        x: Tensor[(8, 16), "f32"],
        w: Tensor[(16, 32), "f32"],
        v: Tensor[(32, 8), "f32"],
        dy: Tensor[(8, 8), "f32"],
    ):
        z = tf.matmul(x, w)
        h = tf.relu(z)
        dv = tf.matmul(tf.transpose(h, perm=(1, 0)), dy)
        dh = tf.matmul(dy, tf.transpose(v, perm=(1, 0)))
        dz = dh * tf.cast(h > 0.0, dtype="f32")
        dw = tf.matmul(tf.transpose(x, perm=(1, 0)), dz)
        dx = tf.matmul(dz, tf.transpose(w, perm=(1, 0)))
        return dx, dw, dv


@module(
    entry="shared",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 1)),
)
class Lifetimes:
    """A multiply-used value and a sequentially carried accumulator."""

    @func
    def shared(x: Tensor[(8, 8), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(1,), names=("lane",)) as _thread:
                local = tf.reshard(x, (8, 8), "rmem")
                h = tf.neg(tf.neg(local))
                first = tf.add(h, local)
                return tf.mul(first, h)

    @func
    def carried(x: Tensor[(8, 8), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(1,), names=("lane",)) as _thread:
                local = tf.reshard(x, (8, 8), "rmem")
                state = tf.zeros(Tensor[(8, 8), "f32", "rmem"])
                for step in range(4):
                    state = tf.add(state, local)
                return state


@module(
    entry="reuse",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 1)),
)
class BufferReuse:
    """Call-result reuse when its actual argument is dead or remains live."""

    @func
    def doubled(x: Tensor[(2, 4), "f32"]):
        return tf.add(x, x)

    @func
    def reuse():
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            a = tf.zeros(Tensor[(2, 4), "f32"])
            return doubled(a)

    @func
    def keep_live():
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            a = tf.zeros(Tensor[(2, 4), "f32"])
            return doubled(a), a
