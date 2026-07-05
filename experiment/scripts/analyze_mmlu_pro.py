#!/usr/bin/env python3
"""Analyze MMLU-Pro Echo sweeps by category and routing quality."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cost_units import cost_units, escalated


CATEGORY_SPECIALIST_HINTS: dict[str, dict[str, str]] = {
    "math": {
        "specialist_type": "math-reasoning specialist",
        "why": "symbolic reasoning, multi-step derivations, and calculation-heavy questions",
    },
    "physics": {
        "specialist_type": "physics/STEM reasoning specialist",
        "why": "formula selection, physical intuition, and quantitative reasoning",
    },
    "chemistry": {
        "specialist_type": "chemistry/STEM reasoning specialist",
        "why": "domain vocabulary plus reaction/structure reasoning",
    },
    "engineering": {
        "specialist_type": "engineering/STEM reasoning specialist",
        "why": "technical constraints and applied quantitative reasoning",
    },
    "computer science": {
        "specialist_type": "code/computer-science specialist",
        "why": "algorithms, systems concepts, and programming-language knowledge",
    },
    "health": {
        "specialist_type": "biomedical/clinical QA specialist",
        "why": "medical terminology and safety-sensitive domain knowledge",
    },
    "biology": {
        "specialist_type": "biomedical/life-sciences specialist",
        "why": "biology terminology, mechanisms, and experimental reasoning",
    },
    "law": {
        "specialist_type": "legal reasoning specialist",
        "why": "statutory interpretation, precedent-like reasoning, and legal terminology",
    },
    "business": {
        "specialist_type": "business/finance specialist",
        "why": "accounting, strategy, and market-domain concepts",
    },
    "economics": {
        "specialist_type": "economics/finance specialist",
        "why": "economic theory, graphs, and quantitative tradeoffs",
    },
    "philosophy": {
        "specialist_type": "logic/philosophy reasoning specialist",
        "why": "argument analysis, definitions, and subtle conceptual distinctions",
    },
    "history": {
        "specialist_type": "history/humanities specialist",
        "why": "factual recall plus chronology and causal interpretation",
    },
    "psychology": {
        "specialist_type": "psychology/social-science specialist",
        "why": "experimental concepts, clinical vocabulary, and theory distinctions",
    },
    "other": {
        "specialist_type": "general reasoning specialist",
        "why": "mixed-domain questions that do not map cleanly to one expert model",
    },
}


def load_records(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    meta = None
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("_meta"):
            meta = obj
            continue
        rows.append(obj)
    return meta, rows


def category_for(row: dict[str, Any]) -> str:
    if row.get("category"):
        return str(row["category"])
    task_id = str(row.get("task_id", ""))
    parts = task_id.split("/")
    if len(parts) >= 3 and parts[0] == "mmlu_pro":
        return parts[1].replace("_", " ")
    return "unknown"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    esc = sum(1 for row in rows if escalated(str(row["arm"]), int(row.get("sub_calls", 0))))
    failures = n - passed
    failure_details = Counter(
        str(row.get("detail", ""))
        for row in rows
        if not row.get("passed")
    )
    total_cost = sum(cost_units(str(row["arm"]), int(row.get("sub_calls", 0))) for row in rows)
    return {
        "n": n,
        "pass_rate": passed / n if n else 0.0,
        "escalation_rate": esc / n if n else 0.0,
        "cost_units": total_cost,
        "cost_per_task": total_cost / n if n else 0.0,
        "failures": failures,
        "top_failure_details": dict(failure_details.most_common(3)),
    }


def by_arm(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["arm"])].append(row)
    return {arm: summarize_rows(arm_rows) for arm, arm_rows in sorted(groups.items())}


def by_category_arm(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(category_for(row), str(row["arm"]))].append(row)
    return {
        key: summarize_rows(group_rows)
        for key, group_rows in sorted(groups.items())
    }


def sonnet_gaps(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    summaries = by_category_arm(rows)
    sonnet_by_category = {
        category: summary["pass_rate"]
        for (category, arm), summary in summaries.items()
        if arm == "sonnet-only"
    }
    gaps: dict[tuple[str, str], float] = {}
    for (category, arm), summary in summaries.items():
        if category in sonnet_by_category and arm != "sonnet-only":
            gaps[(category, arm)] = summary["pass_rate"] - sonnet_by_category[category]
    return gaps


def oracle_diagnostics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task_arm: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["task_id"]), str(row["arm"])): row
        for row in rows
    }
    oracle_by_task = {
        str(row["task_id"]): escalated("echo-oracle", int(row.get("sub_calls", 0)))
        for row in rows
        if row.get("arm") == "echo-oracle"
    }
    if not oracle_by_task:
        return {}

    arms = sorted({str(row["arm"]) for row in rows if str(row["arm"]).startswith("echo-")})
    diagnostics: dict[str, dict[str, Any]] = {}
    for arm in arms:
        if arm == "echo-oracle":
            continue
        comparable = []
        for task_id, oracle_esc in oracle_by_task.items():
            row = by_task_arm.get((task_id, arm))
            if row is None:
                continue
            arm_esc = escalated(arm, int(row.get("sub_calls", 0)))
            comparable.append((row, arm_esc, oracle_esc))

        n = len(comparable)
        if not n:
            continue
        same = sum(1 for _, arm_esc, oracle_esc in comparable if arm_esc == oracle_esc)
        false_accepts = [
            row for row, arm_esc, oracle_esc in comparable
            if not arm_esc and oracle_esc
        ]
        false_escalations = [
            row for row, arm_esc, oracle_esc in comparable
            if arm_esc and not oracle_esc
        ]
        diagnostics[arm] = {
            "n": n,
            "oracle_alignment": same / n,
            "false_accept_rate": len(false_accepts) / n,
            "false_escalation_rate": len(false_escalations) / n,
            "false_accept_tasks": [row["task_id"] for row in false_accepts[:10]],
            "false_escalation_tasks": [row["task_id"] for row in false_escalations[:10]],
        }
    return diagnostics


def specialist_recommendations(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Recommend category-specialist routing candidates from observed gaps."""
    summaries = by_category_arm(rows)
    gaps = sonnet_gaps(rows)
    categories = sorted({category for category, _ in summaries})
    recommendations: dict[str, dict[str, Any]] = {}

    for category in categories:
        category_rows = {
            arm: summary
            for (cat, arm), summary in summaries.items()
            if cat == category
        }
        sonnet = category_rows.get("sonnet-only", {})
        haiku = category_rows.get("haiku-only", {})
        echo_arms = {
            arm: summary
            for arm, summary in category_rows.items()
            if arm.startswith("echo-") and arm != "echo-oracle"
        }
        best_echo_arm = None
        best_echo_summary = None
        if echo_arms:
            best_echo_arm, best_echo_summary = max(
                echo_arms.items(),
                key=lambda item: (item[1]["pass_rate"], -item[1]["cost_per_task"]),
            )

        haiku_gap = gaps.get((category, "haiku-only"))
        best_echo_gap = gaps.get((category, best_echo_arm)) if best_echo_arm else None
        needs_specialist = False
        reasons: list[str] = []

        if haiku_gap is not None and haiku_gap <= -0.10:
            needs_specialist = True
            reasons.append(f"haiku is {abs(100 * haiku_gap):.1f} points below sonnet")
        if best_echo_gap is not None and best_echo_gap <= -0.05:
            needs_specialist = True
            reasons.append(f"best Echo arm remains {abs(100 * best_echo_gap):.1f} points below sonnet")
        if best_echo_summary and best_echo_summary["escalation_rate"] >= 0.40:
            needs_specialist = True
            reasons.append(f"best Echo arm escalates {100 * best_echo_summary['escalation_rate']:.1f}% of tasks")
        if not reasons:
            reasons.append("current generalist routing looks adequate on this slice")

        hint = CATEGORY_SPECIALIST_HINTS.get(
            category,
            {
                "specialist_type": "category-specific specialist",
                "why": "category performance suggests domain-specific behavior may matter",
            },
        )
        recommendations[category] = {
            "needs_specialist_probe": needs_specialist,
            "specialist_type": hint["specialist_type"],
            "why_this_specialist": hint["why"],
            "reason": "; ".join(reasons),
            "sonnet_pass_rate": sonnet.get("pass_rate"),
            "haiku_pass_rate": haiku.get("pass_rate"),
            "best_echo_arm": best_echo_arm,
            "best_echo_pass_rate": best_echo_summary["pass_rate"] if best_echo_summary else None,
            "best_echo_cost_per_task": best_echo_summary["cost_per_task"] if best_echo_summary else None,
        }
    return recommendations


def print_overall(summary: dict[str, dict[str, Any]]) -> None:
    print("=== overall by arm ===")
    print(f"{'arm':<34} {'pass':>7} {'esc':>7} {'cost':>9} {'cost/task':>10} {'fail':>6}")
    print("-" * 80)
    for arm, data in summary.items():
        print(
            f"{arm:<34} "
            f"{100 * data['pass_rate']:>6.1f}% "
            f"{100 * data['escalation_rate']:>6.1f}% "
            f"{data['cost_units']:>9.1f} "
            f"{data['cost_per_task']:>10.2f} "
            f"{data['failures']:>6}"
        )


def print_category_table(
    summaries: dict[tuple[str, str], dict[str, Any]],
    gaps: dict[tuple[str, str], float],
) -> None:
    print("\n=== by category and arm ===")
    print(f"{'category':<18} {'arm':<34} {'pass':>7} {'sonnet_gap':>11} {'esc':>7} {'cost/task':>10}")
    print("-" * 95)
    for (category, arm), data in summaries.items():
        gap = gaps.get((category, arm))
        gap_text = "" if gap is None else f"{100 * gap:+.1f}%"
        print(
            f"{category:<18} {arm:<34} "
            f"{100 * data['pass_rate']:>6.1f}% "
            f"{gap_text:>11} "
            f"{100 * data['escalation_rate']:>6.1f}% "
            f"{data['cost_per_task']:>10.2f}"
        )


def print_oracle(diagnostics: dict[str, dict[str, Any]]) -> None:
    if not diagnostics:
        return
    print("\n=== oracle routing diagnostics ===")
    print(f"{'arm':<34} {'align':>8} {'false_accept':>14} {'false_escalate':>15}")
    print("-" * 78)
    for arm, data in diagnostics.items():
        print(
            f"{arm:<34} "
            f"{100 * data['oracle_alignment']:>7.1f}% "
            f"{100 * data['false_accept_rate']:>13.1f}% "
            f"{100 * data['false_escalation_rate']:>14.1f}%"
        )
        if data["false_accept_tasks"]:
            print(f"  false accepts: {', '.join(data['false_accept_tasks'])}")
        if data["false_escalation_tasks"]:
            print(f"  false escalations: {', '.join(data['false_escalation_tasks'])}")


def print_specialists(recommendations: dict[str, dict[str, Any]]) -> None:
    if not recommendations:
        return
    print("\n=== category specialist routing probes ===")
    print(f"{'category':<18} {'probe?':>7} {'specialist':<36} {'reason'}")
    print("-" * 100)
    for category, rec in recommendations.items():
        probe = "yes" if rec["needs_specialist_probe"] else "later"
        print(
            f"{category:<18} {probe:>7} "
            f"{rec['specialist_type']:<36} {rec['reason']}"
        )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    table = ["| " + " | ".join(headers) + " |"]
    table.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def build_markdown_report(
    *,
    source: Path,
    meta: dict[str, Any] | None,
    overall: dict[str, dict[str, Any]],
    category: dict[tuple[str, str], dict[str, Any]],
    gaps: dict[tuple[str, str], float],
    oracle: dict[str, dict[str, Any]],
    specialists: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# MMLU-Pro Echo Analysis",
        "",
        f"Source: `{source}`",
        "",
    ]
    if meta:
        lines.extend([
            "## Sweep Metadata",
            "",
            "```json",
            json.dumps({k: v for k, v in meta.items() if k != "_meta"}, indent=2),
            "```",
            "",
        ])

    overall_rows = [
        [
            arm,
            f"{100 * data['pass_rate']:.1f}%",
            f"{100 * data['escalation_rate']:.1f}%",
            f"{data['cost_units']:.1f}",
            f"{data['cost_per_task']:.2f}",
            str(data["failures"]),
        ]
        for arm, data in overall.items()
    ]
    lines.extend([
        "## Overall By Arm",
        "",
        markdown_table(["Arm", "Pass", "Esc", "Cost", "Cost/Task", "Failures"], overall_rows),
        "",
    ])

    category_rows = []
    for (category_name, arm), data in category.items():
        gap = gaps.get((category_name, arm))
        category_rows.append([
            category_name,
            arm,
            f"{100 * data['pass_rate']:.1f}%",
            "" if gap is None else f"{100 * gap:+.1f}%",
            f"{100 * data['escalation_rate']:.1f}%",
            f"{data['cost_per_task']:.2f}",
        ])
    lines.extend([
        "## Category Breakdown",
        "",
        markdown_table(["Category", "Arm", "Pass", "Gap vs Sonnet", "Esc", "Cost/Task"], category_rows),
        "",
    ])

    if oracle:
        oracle_rows = [
            [
                arm,
                f"{100 * data['oracle_alignment']:.1f}%",
                f"{100 * data['false_accept_rate']:.1f}%",
                f"{100 * data['false_escalation_rate']:.1f}%",
                ", ".join(data["false_accept_tasks"][:5]),
                ", ".join(data["false_escalation_tasks"][:5]),
            ]
            for arm, data in oracle.items()
        ]
        lines.extend([
            "## Oracle Routing Diagnostics",
            "",
            markdown_table(
                ["Arm", "Oracle Align", "False Accept", "False Escalate", "False Accept Tasks", "False Escalation Tasks"],
                oracle_rows,
            ),
            "",
        ])

    specialist_rows = [
        [
            category_name,
            "yes" if rec["needs_specialist_probe"] else "later",
            rec["specialist_type"],
            rec["reason"],
        ]
        for category_name, rec in specialists.items()
    ]
    lines.extend([
        "## Specialist Routing Probes",
        "",
        markdown_table(["Category", "Probe?", "Specialist Type", "Reason"], specialist_rows),
        "",
        "## Interpretation Guide",
        "",
        "- Prioritize pass rate first, then cost and escalation rate.",
        "- `false_accept_rate` is the dangerous error: the router accepted cheap output when oracle would escalate.",
        "- `false_escalation_rate` is a cost error: the router paid for Sonnet when cheap output was already good.",
        "- Categories marked `yes` are candidates for a small specialist model or specialist prompt in the next harness version.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MMLU-Pro Echo results.")
    parser.add_argument("jsonl", type=Path, help="Path to *_mmlu_pro_*.jsonl")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--report",
        nargs="?",
        const="auto",
        default=None,
        help="Write a Markdown report. Omit value to use <jsonl>_REPORT.md.",
    )
    args = parser.parse_args()

    meta, rows = load_records(args.jsonl)
    if not rows:
        print("No result rows found.", file=sys.stderr)
        sys.exit(1)

    overall = by_arm(rows)
    category = by_category_arm(rows)
    gaps = sonnet_gaps(rows)
    oracle = oracle_diagnostics(rows)
    specialists = specialist_recommendations(rows)

    if args.json:
        print(json.dumps({
            "meta": meta,
            "overall": overall,
            "by_category_arm": {
                f"{category_name}::{arm}": data
                for (category_name, arm), data in category.items()
            },
            "sonnet_gaps": {
                f"{category_name}::{arm}": gap
                for (category_name, arm), gap in gaps.items()
            },
            "oracle_diagnostics": oracle,
            "specialist_recommendations": specialists,
        }, indent=2))
        return

    if meta:
        print("=== sweep meta ===")
        print(json.dumps({k: v for k, v in meta.items() if k != "_meta"}, indent=2))
        print()
    print_overall(overall)
    print_category_table(category, gaps)
    print_oracle(oracle)
    print_specialists(specialists)

    if args.report:
        if args.report == "auto":
            report_path = args.jsonl.with_name(f"{args.jsonl.stem}_REPORT.md")
        else:
            report_path = Path(args.report)
        report = build_markdown_report(
            source=args.jsonl,
            meta=meta,
            overall=overall,
            category=category,
            gaps=gaps,
            oracle=oracle,
            specialists=specialists,
        )
        report_path.write_text(report)
        print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
