"""Bounded TileFoundry legality/compute checks and local-stage memory reports."""

import argparse
import json
from pathlib import Path

from tilefoundry.analysis import ComputeCostMetadata, analyze, check_program
from tilefoundry.ir.core import get_metadata

from .lower import function, lower, static_reports
from .parallel import scan_gated_delta_rule
from .planner import plan_from_saved
from .problems import chunk_gated_delta_rule

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        default="64,512",
        help="Monolithic compute analysis becomes expensive at larger lengths",
    )
    args = parser.parse_args()
    directory = ROOT / "results" / "training_scale" / "static"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for length in map(int, args.lengths.split(",")):
        for variant, factory in (
            ("sequential", chunk_gated_delta_rule),
            ("affine_scan", scan_gated_delta_rule),
        ):
            graph = factory(length, 64, 64, 64)
            plan = plan_from_saved(graph, [])
            destination = directory / f"{variant}_{length}"
            program = lower(graph, plan, destination / "program.py")
            row = {
                "length": length,
                "variant": variant,
                "residual_bytes": 0,
                "phases": {},
            }
            for name in ("forward", "backward"):
                try:
                    checked = check_program(
                        program.module, function(program.module, name)
                    )
                    result = analyze(*checked, analysis=("compute-cost",))
                    metadata = get_metadata(result.function, ComputeCostMetadata)
                    row["phases"][name] = {
                        "legal": True,
                        "flops": sum(v.total for _, v in metadata.flops.kinds),
                    }
                except Exception as error:
                    row["phases"][name] = {"error": f"{type(error).__name__}: {error}"}
            if length == 64:
                try:
                    row["local_stage_memory"] = static_reports(
                        program, destination, memory_timeout=15
                    )
                except Exception as error:
                    row["local_stage_memory_error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            (directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
