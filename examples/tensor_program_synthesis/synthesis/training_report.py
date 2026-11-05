"""Build a compact index of completed training-scale evidence."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "results" / "training_scale"
    rows = [
        json.loads(p.read_text()) for p in sorted(directory.glob("T*/summary.json"))
    ]
    audit = json.loads((directory / "precision_audit.json").read_text())
    policies = json.loads((directory / "policy_audit.json").read_text())
    checks = [
        c
        for r in rows
        for v in r["variants"].values()
        for p in v["policies"].values()
        for c in p.get("checks", [])
    ]
    overview = {
        "execution": "CPU HIR evaluator and tensor formulas; no native GPU benchmark",
        "tilefoundry_commit": (ROOT / "TILEFOUNDRY_COMMIT").read_text().strip(),
        "fp32_hir_case_count": len(checks),
        "fp32_hir_original_tolerance_passes": sum(c["passed"] for c in checks),
        "fp32_hir_original_tolerance_failures": sum(not c["passed"] for c in checks),
        "fp32_hir_tolerance": {"rtol": 2e-5, "atol": 2e-5},
        "fp64_formula_case_count": len(audit),
        "fp64_formula_passes": sum(
            x["formula"]["float64"]["passed_original_tolerance_against_fp64_reference"]
            for x in audit
        ),
        "fp64_formula_tolerance": {"rtol": 2e-9, "atol": 2e-9},
        "fp32_max_absolute_error_against_fp64": max(
            m["max_abs"]
            for x in audit
            for m in x["formula"]["float32"]["metrics"].values()
        ),
        "fp32_max_relative_l2_excluding_near_zero_strong_decay": max(
            m["relative_l2"]
            for x in audit
            if x["case"] != "strong_decay"
            for m in x["formula"]["float32"]["metrics"].values()
        ),
        "residual_policy_invariance_case_count": len(policies),
        "residual_policy_bitwise_equal_passes": sum(
            x["bitwise_equal_to_no_residual_formula"] for x in policies
        ),
        "shapes": [],
        "static_analysis": json.loads(
            (directory / "static" / "summary.json").read_text()
        ),
        "infrastructure_notes": (
            json.loads((directory / "infrastructure_notes.json").read_text())
            if (directory / "infrastructure_notes.json").exists()
            else {}
        ),
    }
    for r in rows:
        row = {
            k: r[k]
            for k in (
                "length",
                "chunk_size",
                "key_dim",
                "value_dim",
                "chunks",
                "replicas_executed",
            )
        }
        row["variants"] = {}
        for variant, values in r["variants"].items():
            assert all(
                p.get("checks") or "validation_same_as" in p
                for p in values["policies"].values()
            ), (r["length"], variant)
            row["variants"][variant] = {
                label: {
                    k: p[k]
                    for k in (
                        "residual_bytes",
                        "objective_work_units",
                        "total_training_work_units",
                        "solver_status",
                        "lower_bound_work_units",
                        "work_bound_gap_fraction",
                        "dependencies",
                    )
                }
                for label, p in values["policies"].items()
            }
        assert set(row["variants"]) == {"sequential", "affine_scan"}
        overview["shapes"].append(row)
    (directory / "overview.json").write_text(json.dumps(overview, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in overview.items()
                if k not in {"shapes", "static_analysis", "infrastructure_notes"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
