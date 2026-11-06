"""Measure one B300 expert or one verified two-rank fabric hop."""

import argparse
import json
import os
import re
import socket
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from workload import EXPERT, HIDDEN


def capture(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()


def samples(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1000)
    return times


def expert_inputs(rows, device):
    def rand(*shape):
        return torch.randn(shape, device=device, dtype=torch.bfloat16)

    x, w13, w2, dy = (
        rand(rows, HIDDEN),
        rand(HIDDEN, 2 * EXPERT),
        rand(EXPERT, HIDDEN),
        rand(rows, HIDDEN),
    )
    gate, up = (x @ w13).chunk(2, dim=-1)
    h = F.silu(gate) * up
    dh = dy @ w2.T

    def gate_grad():
        s = gate.sigmoid()
        return torch.cat((dh * up * s * (1 + gate * (1 - s)), dh * F.silu(gate)), dim=-1)

    dg = gate_grad()
    return {
        "gate_up": lambda: x @ w13,
        "down": lambda: (F.silu(gate) * up) @ w2,
        "down_input": lambda: dy @ w2.T,
        "down_weight": lambda: h.T @ dy,
        "up_input": lambda: gate_grad() @ w13.T,
        "up_weight": lambda: x.T @ dg,
    }


def identity(device):
    props = torch.cuda.get_device_properties(device)
    cc = [props.major, props.minor]
    if cc != [10, 3]:
        raise ValueError(f"expected SM103, CUDA reports {cc} for {props.name}")
    return {
        "name": props.name,
        "cc": cc,
        "uuid": str(props.uuid),
        "host": socket.gethostname(),
        "driver": capture(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ).splitlines()[0],
    }


def nvlink_path(devices):
    by_uuid = {}
    for line in capture(
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"
    ).splitlines():
        index, uuid = line.split(",")
        by_uuid[uuid.strip().lower().removeprefix("gpu-")] = "GPU" + index.strip()
    pair = [by_uuid[d["uuid"].lower().removeprefix("gpu-")] for d in devices]
    if pair[0] == pair[1]:
        raise ValueError("NVLink requires two distinct GPUs")
    topology = [
        line.split() for line in capture("nvidia-smi", "topo", "-m").splitlines() if line.strip()
    ]
    header = next(row for row in topology if row[0] == "GPU0" and row[1].startswith("GPU"))
    source = next(row for row in topology if row[0] == pair[0] and row != header)
    link = source[1 + header.index(pair[1])]
    if not re.fullmatch(r"NV\d+", link):
        raise ValueError(f"{pair}: topology reports {link}, expected an NVLink path")
    return f"{pair[0]} to {pair[1]}: {link}"


def ib_port(hca):
    if not hca or not re.fullmatch(r"[\w.-]+:[1-9]\d*", hca):
        raise ValueError("pass --hca DEVICE:PORT for the active InfiniBand port on this host")
    name, port = hca.split(":")
    path = Path("/sys/class/infiniband") / name / "ports" / port
    if (path / "link_layer").read_text().strip() != "InfiniBand" or "ACTIVE" not in (
        path / "state"
    ).read_text():
        raise ValueError(f"{hca} must be an active InfiniBand port")
    os.environ.update(NCCL_NET="IB", NCCL_IB_HCA="=" + hca, NCCL_IB_DISABLE="0")
    return hca


def peer(args, device, info, log):
    if int(os.getenv("WORLD_SIZE", "1")) != 2:
        raise ValueError("peer probes require exactly two torchrun ranks")
    if args.probe == "ib":
        info["hca"] = ib_port(args.hca)
    os.environ.update(
        NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET,P2P,GRAPH", NCCL_DEBUG_FILE=str(log)
    )
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=90))
    try:
        rank = dist.get_rank()
        info["boot"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        devices = [None, None]
        dist.all_gather_object(devices, info)
        same_node = devices[0].pop("boot") == devices[1].pop("boot")
        if same_node != (args.probe == "nvlink"):
            raise ValueError("NVLink needs two GPUs on one node; IB needs a rank on each node")
        evidence = nvlink_path(devices) if same_node else [d["hca"] for d in devices]
        buf = torch.empty((args.rows, HIDDEN), dtype=torch.bfloat16, device=device)
        result = {}
        for source, direction in enumerate(("out", "back")):
            buf.fill_(source + 1 if rank == source else 0)

            def transfer():
                if rank == source:
                    dist.send(buf, dst=1 - source)
                else:
                    dist.recv(buf, src=source)

            # Align both ranks before the per-trial window. Record receive-side completion.
            for _ in range(args.warmup):
                transfer()
            values = []
            for _ in range(args.repeats):
                dist.barrier(device_ids=[device.index])
                values.extend(samples(transfer, 0, 1))
            if not bool((buf == source + 1).all()):
                raise RuntimeError("peer payload check failed")
            peers = [None, None]
            dist.all_gather_object(peers, values)
            result[f"{args.probe}_{direction}"] = peers[1 - source]
        pattern = r"via P2P/" if same_node else r"via NET/IB"
        matches = [
            line.strip() for line in log.read_text().splitlines() if re.search(pattern, line)
        ]
        records = [None, None]
        dist.all_gather_object(records, matches[:2])
        if not any(records):
            raise ValueError(
                f"NCCL did not confirm {pattern} on this pair; no measurement published"
            )
        return devices, result, {"topology": evidence, "nccl_channels": records}
    finally:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("probe", choices=("compute", "nvlink", "ib"))
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--hca", help="local HCA:port for an inter-node IB probe")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if min(a.rows, a.repeats) < 1 or a.warmup < 0:
        p.error("rows and repeats must be positive; warmup must be nonnegative")
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    info = identity(device)
    with tempfile.TemporaryDirectory() as tmp:
        if a.probe == "compute":
            if int(os.getenv("WORLD_SIZE", "1")) != 1:
                p.error("run the compute probe as one process")
            with torch.no_grad():
                ops = expert_inputs(a.rows, device)
                values = {key: samples(op, a.warmup, a.repeats) for key, op in ops.items()}
                if not all(bool(torch.isfinite(op()).all()) for op in ops.values()):
                    raise RuntimeError("non-finite expert output")
            devices, evidence = [info], {"implementation": "PyTorch BF16 eager matrix products"}
        else:
            devices, values, evidence = peer(a, device, info, Path(tmp) / "nccl.log")
    if int(os.getenv("RANK", "0")) == 0:
        record = {
            "schema": 1,
            "probe": a.probe,
            "scope": "isolated",
            "dtype": "bf16",
            "rows": a.rows,
            "hidden": HIDDEN,
            "expert": EXPERT,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "devices": devices,
            "evidence": evidence,
            "warmup": a.warmup,
            "samples_us": values,
        }
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
