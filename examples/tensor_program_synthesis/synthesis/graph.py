"""A small functional tensor graph with an explicit forward/backward boundary."""

from dataclasses import dataclass, field
from collections import Counter
from itertools import zip_longest
from math import prod

import torch
import torch.nn.functional as F

DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "bool": torch.bool}


def broadcast(a, b):
    result = []
    for x, y in zip_longest(reversed(a), reversed(b), fillvalue=1):
        if x != y and x != 1 and y != 1:
            raise ValueError(f"Cannot broadcast {a} with {b}")
        result.append(max(x, y))
    return tuple(reversed(result))


@dataclass(frozen=True)
class Recipe:
    op: str
    args: tuple[int, ...]
    attrs: tuple[tuple[str, object], ...] = ()
    rule: str = "definition"


@dataclass
class Node:
    id: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    phase: str
    recipes: list[Recipe] = field(default_factory=list)

    @property
    def recipe(self):
        return self.recipes[0]

    @property
    def nbytes(self):
        return prod(self.shape) * {"f32": 4, "bf16": 2, "bool": 1}[self.dtype]


@dataclass(frozen=True)
class Value:
    graph: "Graph"
    id: int

    @property
    def node(self):
        return self.graph.nodes[self.id]

    def __add__(self, other):
        return self.graph.op("add", self, other)

    def __sub__(self, other):
        return self.graph.op("sub", self, other)

    def __mul__(self, other):
        return self.graph.op("mul", self, other)

    def __matmul__(self, other):
        return self.graph.op("matmul", self, other)

    def __neg__(self):
        return self.graph.op("neg", self)

    @property
    def T(self):
        return self.graph.op("transpose", self)

    def exp(self):
        return self.graph.op("exp", self)

    def relu(self):
        return self.graph.op("relu", self)

    def positive(self):
        return self.graph.op("gt", self, 0.0)

    def cast(self, dtype="f32"):
        return self.graph.op("cast", self, dtype=dtype)

    def sum(self, axis):
        return self.graph.op("sum", self, axis=axis)

    def rows(self, start, end):
        return self.graph.op("slice", self, start=start, end=end)

    def named(self, name):
        self.node.name = name
        return self


class Graph:
    """Supplied formulas and recipes; inputs are immutable across phases."""

    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.phase = "forward"
        self.forward_inputs = []
        self.backward_inputs = []
        self.forward_outputs = []
        self.backward_outputs = []
        self._cse = {}
        self.metadata = {}

    def _node(self, name, shape, dtype, recipe):
        node = Node(len(self.nodes), name, tuple(shape), dtype, self.phase, [recipe])
        self.nodes.append(node)
        return Value(self, node.id)

    def input(self, name, shape, dtype="f32"):
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype {dtype}")
        result = self._node(name, shape, dtype, Recipe("input", ()))
        (
            self.forward_inputs if self.phase == "forward" else self.backward_inputs
        ).append(result.id)
        return result

    def scalar(self, value):
        key = ("scalar", float(value))
        if key not in self._cse:
            self._cse[key] = self._node(
                str(value), (), "f32", Recipe("scalar", (), (("value", float(value)),))
            ).id
        return Value(self, self._cse[key])

    def constant(self, kind, size):
        key = ("constant", kind, size)
        if key not in self._cse:
            shape = (size, 1) if kind == "last" else (size, size)
            self._cse[key] = self._node(
                f"{kind}_{size}",
                shape,
                "f32",
                Recipe("constant", (), (("kind", kind), ("size", size))),
            ).id
        return Value(self, self._cse[key])

    def op(self, op, *values, **attrs):
        args = tuple(v if isinstance(v, Value) else self.scalar(v) for v in values)
        if any(v.graph is not self for v in args):
            raise ValueError("Cannot combine values from different graphs")
        shapes = [v.node.shape for v in args]
        dtype = args[0].node.dtype if args else attrs.get("dtype", "f32")
        if op == "zeros":
            shape = tuple(attrs["shape"])
        elif op in {"add", "sub", "mul", "gt"}:
            shape = broadcast(*shapes)
            if op == "gt":
                dtype = "bool"
        elif op == "matmul":
            a, b = shapes
            if len(a) < 2 or len(b) < 2 or a[-1] != b[-2]:
                raise ValueError(f"Invalid matmul: {a}, {b}")
            shape = (*broadcast(a[:-2], b[:-2]), a[-2], b[-1])
            if args[0].node.dtype != args[1].node.dtype:
                raise ValueError("Matmul operands must have the same dtype")
            dtype = attrs.get("out_dtype", dtype)
        elif op == "transpose":
            rank = len(shapes[0])
            if rank < 2:
                raise ValueError("Matrix transpose requires rank >= 2")
            perm = tuple(range(rank - 2)) + (rank - 1, rank - 2)
            if rank > 2:
                attrs["perm"] = perm
            shape = tuple(shapes[0][j] for j in perm)
        elif op == "sum":
            if not -len(shapes[0]) <= attrs["axis"] < len(shapes[0]):
                raise ValueError("Reduction axis out of range")
            attrs["axis"] %= len(shapes[0])
            shape = tuple(
                1 if i == attrs["axis"] else n for i, n in enumerate(shapes[0])
            )
        elif op == "slice":
            axis = attrs.get("axis", 0)
            attrs["rank"] = len(shapes[0])
            if not 0 <= attrs["start"] < attrs["end"] <= shapes[0][axis]:
                raise ValueError("Invalid row slice")
            shape = tuple(
                attrs["end"] - attrs["start"] if i == axis else s
                for i, s in enumerate(shapes[0])
            )
        elif op == "concat":
            axis = attrs.get("axis", 0)
            if any(
                len(s) != len(shapes[0])
                or any(
                    x != y for i, (x, y) in enumerate(zip(s, shapes[0])) if i != axis
                )
                for s in shapes
            ):
                raise ValueError("Concat needs matching non-concatenated dimensions")
            shape = tuple(
                sum(s[i] for s in shapes) if i == axis else n
                for i, n in enumerate(shapes[0])
            )
        elif op == "reshape":
            shape = tuple(attrs["shape"])
            if prod(shape) != prod(shapes[0]) or any(x <= 0 for x in shape):
                raise ValueError("Reshape must preserve the number of elements")
        elif op == "permute":
            perm = attrs["perm"]
            if sorted(perm) != list(range(len(shapes[0]))):
                raise ValueError("Invalid permutation")
            shape = tuple(shapes[0][i] for i in perm)
        elif op == "where":
            shape = broadcast(broadcast(shapes[0], shapes[1]), shapes[2])
            dtype = args[1].node.dtype
        elif op in {
            "neg",
            "relu",
            "exp",
            "cast",
            "rsqrt",
            "sigmoid",
            "silu",
            "softplus",
        }:
            shape = shapes[0]
            if op == "cast":
                dtype = attrs["dtype"]
        else:
            raise ValueError(f"Unsupported op {op}")
        recipe = Recipe(op, tuple(v.id for v in args), tuple(sorted(attrs.items())))
        key = (recipe, shape, dtype)
        if key not in self._cse:
            self._cse[key] = self._node(
                f"{op}_{len(self.nodes)}", shape, dtype, recipe
            ).id
        return Value(self, self._cse[key])

    def begin_backward(self, outputs):
        self.forward_outputs = [v.id for v in outputs]
        self.phase = "backward"

    def finish(self, outputs):
        self.backward_outputs = [v.id for v in outputs]
        self.validate()
        return self

    def ancestors(self, roots, *, alternatives=False):
        found = set()
        pending = list(roots)
        while pending:
            i = pending.pop()
            if i in found:
                continue
            found.add(i)
            for recipe in (
                self.nodes[i].recipes if alternatives else self.nodes[i].recipes[:1]
            ):
                pending.extend(recipe.args)
        return found

    def schedule(self, roots, *, available=(), recipes=None):
        """Iterative demanded topological order, stopping at phase interfaces."""
        done, visiting, order = set(available), set(), []
        for root in roots:
            pending = [(root, False)]
            while pending:
                i, finish = pending.pop()
                if i in done:
                    continue
                if finish:
                    visiting.remove(i)
                    done.add(i)
                    order.append(i)
                    continue
                if i in visiting:
                    raise ValueError("Recipe alternatives must form an acyclic graph")
                visiting.add(i)
                r = self.nodes[i].recipes[0 if recipes is None else recipes.get(i, 0)]
                pending.append((i, True))
                pending.extend((j, False) for j in reversed(r.args))
        return order

    @property
    def forward_available(self):
        available = set(self.forward_inputs)
        for n in self.nodes:
            if n.recipe.op in {"scalar", "constant"}:
                available.add(n.id)
            elif n.recipe.op != "input" and all(i in available for i in n.recipe.args):
                available.add(n.id)
        return available

    @property
    def external(self):
        return set(
            self.forward_inputs + self.backward_inputs + self.forward_outputs
        ) | {n.id for n in self.nodes if n.recipe.op in {"scalar", "constant"}}

    @property
    def candidates(self):
        wanted = self.ancestors(self.backward_outputs, alternatives=True)
        return sorted(
            i
            for i in self.forward_available & wanted - self.external
            if self.nodes[i].recipe.op not in {"transpose", "slice"}
        )

    def add_rules(self):
        """Register one bounded algebraic rule: positive(ReLU(z)) = positive(z)."""
        relus = {}
        for n in self.nodes:
            if n.recipe.op == "relu":
                relus.setdefault(n.recipe.args[0], []).append(n.id)
        count = 0
        for n in self.nodes:
            if n.recipe.op != "gt":
                continue
            z, zero = n.recipe.args
            scalar = self.nodes[zero].recipe
            if scalar.op != "scalar" or dict(scalar.attrs)["value"] != 0:
                continue
            for h in relus.get(z, []):
                alternative = Recipe("gt", (h, zero), rule="relu_positive_mask")
                if alternative not in n.recipes:
                    n.recipes.append(alternative)
                    count += 1
        self.validate()
        return count

    def validate(self):
        visiting, done = set(), set()

        def visit(i):
            if i in visiting:
                raise ValueError("Recipe alternatives must form an acyclic graph")
            if i in done:
                return
            visiting.add(i)
            for recipe in self.nodes[i].recipes:
                for j in recipe.args:
                    visit(j)
            visiting.remove(i)
            done.add(i)

        for n in self.nodes:
            visit(n.id)

    def work(self, i, recipe=None):
        """Elementary-operation proxy; exp/compare/cast each count as one unit."""
        n = self.nodes[i]
        recipe = recipe or n.recipe
        op = recipe.op
        if op in {
            "input",
            "constant",
            "scalar",
            "transpose",
            "slice",
            "concat",
            "reshape",
            "permute",
        }:
            return 0
        if op == "matmul":
            return 2 * prod(n.shape) * self.nodes[recipe.args[0]].shape[-1]
        if op == "sum":
            return prod(self.nodes[recipe.args[0]].shape) - prod(n.shape)
        return prod(n.shape)

    def dump(self):
        return {
            "name": self.name,
            "inputs": self.forward_inputs,
            "cotangents": self.backward_inputs,
            "forward_outputs": self.forward_outputs,
            "backward_outputs": self.backward_outputs,
            "residual_candidates": self.candidates,
            "metadata": self.metadata,
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "shape": n.shape,
                    "dtype": n.dtype,
                    "phase": n.phase,
                    "bytes": n.nbytes,
                    "recipes": [
                        {
                            "op": r.op,
                            "args": r.args,
                            "attrs": dict(r.attrs),
                            "rule": r.rule,
                            "work": self.work(n.id, r),
                        }
                        for r in n.recipes
                    ],
                }
                for n in self.nodes
            ],
        }


def constant_tensor(node, dtype=torch.float32):
    attrs = dict(node.recipe.attrs)
    size, kind = attrs["size"], attrs["kind"]
    if kind == "eye":
        return torch.eye(size, dtype=dtype)
    if kind == "last":
        result = torch.zeros(size, 1, dtype=dtype)
        result[-1] = 1
        return result
    if kind in {"lower", "strict"}:
        return torch.ones(size, size, dtype=dtype).tril(-1 if kind == "strict" else 0)
    raise ValueError(kind)


def apply_recipe(recipe, args, *, strict_dtypes=False):
    attrs = dict(recipe.attrs)
    op = recipe.op
    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "matmul":
        if "out_dtype" in attrs:
            return (args[0].float() @ args[1].float()).to(DTYPES[attrs["out_dtype"]])
        return args[0] @ args[1]
    if op == "transpose":
        return args[0].transpose(-1, -2)
    if op == "permute":
        return args[0].permute(attrs["perm"])
    if op == "reshape":
        return args[0].reshape(attrs["shape"])
    if op == "where":
        return torch.where(*args)
    if op == "rsqrt":
        return args[0].rsqrt()
    if op == "sigmoid":
        return args[0].sigmoid()
    if op == "silu":
        return F.silu(args[0])
    if op == "softplus":
        return F.softplus(args[0])
    if op == "neg":
        return -args[0]
    if op == "relu":
        return args[0].relu()
    if op == "exp":
        return args[0].exp()
    if op == "gt":
        return args[0] > args[1]
    if op == "cast":
        return args[0].to(
            torch.float64
            if not strict_dtypes
            and attrs["dtype"] == "f32"
            and any(
                isinstance(a, torch.Tensor) and a.dtype == torch.float64 for a in args
            )
            else DTYPES[attrs["dtype"]]
        )
    if op == "sum":
        return args[0].sum(dim=attrs["axis"], keepdim=True)
    if op == "slice":
        slices = [slice(None)] * args[0].ndim
        slices[attrs.get("axis", 0)] = slice(attrs["start"], attrs["end"])
        return args[0][tuple(slices)]
    if op == "concat":
        return torch.cat(args, dim=attrs.get("axis", 0))
    raise ValueError(op)


def evaluate_graph(
    graph, roots, inputs, *, recipes=None, saved=None, retain_intermediates=True
):
    """Evaluate only demanded nodes, optionally using the solver's decisions."""
    values = dict(inputs)
    values.update(saved or {})
    strict_dtypes = graph.metadata.get("strict_dtypes", False) or any(
        n.dtype == "bf16" for n in graph.nodes
    )
    dtype = next(
        (
            v.dtype
            for v in inputs.values()
            if isinstance(v, torch.Tensor) and v.is_floating_point()
        ),
        torch.float32,
    )

    order = graph.schedule(roots, available=values, recipes=recipes)
    uses = Counter()
    if not retain_intermediates:
        for i in order:
            uses.update(
                graph.nodes[i].recipes[0 if recipes is None else recipes.get(i, 0)].args
            )
    root_set = set(roots)
    for i in order:
        n = graph.nodes[i]
        r = (
            n.recipe
            if recipes is None or n.recipe.op in {"constant", "scalar", "input"}
            else n.recipes[recipes[i]]
        )
        if r.op == "zeros":
            value = torch.zeros(
                n.shape, dtype=DTYPES[n.dtype] if strict_dtypes else dtype
            )
        elif r.op == "constant":
            value = constant_tensor(n, DTYPES[n.dtype] if strict_dtypes else dtype)
        elif r.op == "scalar":
            value = dict(r.attrs)["value"]
        elif r.op == "input":
            raise ValueError(f"Missing external input {n.name}")
        else:
            value = apply_recipe(
                r, [values[j] for j in r.args], strict_dtypes=strict_dtypes
            )
            if r.op == "cast" and not strict_dtypes:
                value = value.to(dtype)
        values[i] = value
        if not retain_intermediates:
            for dependency in r.args:
                uses[dependency] -= 1
                if uses[dependency] == 0 and dependency not in root_set:
                    values.pop(dependency, None)
    return tuple(values[i] for i in roots), values
