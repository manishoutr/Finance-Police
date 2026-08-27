#!/usr/bin/env python3
"""Run the complete system test suite and write a human-readable report."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def build_report(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    total = failures = errors = skipped = 0
    cases = []
    for suite in suites:
        total += int(suite.attrib.get("tests", 0))
        failures += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        skipped += int(suite.attrib.get("skipped", 0))
        for case in suite.findall("testcase"):
            status = "PASSED"
            detail = ""
            failure = case.find("failure")
            error = case.find("error")
            skip = case.find("skipped")
            if failure is not None:
                status = "FAILED"
                detail = (failure.attrib.get("message") or failure.text or "").strip()
            elif error is not None:
                status = "ERROR"
                detail = (error.attrib.get("message") or error.text or "").strip()
            elif skip is not None:
                status = "SKIPPED"
                detail = (skip.attrib.get("message") or skip.text or "").strip()
            cases.append({
                "test": f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}",
                "status": status,
                "detail": detail,
                "duration_seconds": float(case.attrib.get("time", 0.0)),
            })

    passed = total - failures - errors - skipped
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_tests": total,
        "passed": passed,
        "failed": failures + errors,
        "skipped": skipped,
        "failure_details": [c for c in cases if c["status"] in {"FAILED", "ERROR"}],
        "tests": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    base = args.base_dir.resolve()
    outputs = base / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    xml_path = outputs / "system_tests_junit.xml"

    command = [sys.executable, "-m", "pytest", "-q", "--junitxml", str(xml_path)]
    result = subprocess.run(command, cwd=base)

    report = build_report(xml_path)
    (outputs / "system_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    import csv
    with (outputs / "system_test_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test", "status", "detail", "duration_seconds"])
        writer.writeheader()
        writer.writerows(report["tests"])

    lines = [
        "# STEP 14 RELIABILITY TEST REPORT",
        "",
        f"Total tests: {report['total_tests']}",
        f"Passed: {report['passed']}",
        f"Failed: {report['failed']}",
        f"Skipped: {report['skipped']}",
        "",
    ]
    if report["failure_details"]:
        lines += ["## Failure details", ""]
        for item in report["failure_details"]:
            lines += [f"- **{item['test']}**", f"  - {item['detail']}"]
    else:
        lines += ["## Failure details", "", "None."]
    lines += ["", "## Reliability notes", "", "See the test output for warnings and skipped environment-dependent checks."]
    (outputs / "system_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n## STEP 14 RELIABILITY TEST REPORT\n")
    print(f"Total tests: {report['total_tests']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"Skipped: {report['skipped']}")
    if report["failure_details"]:
        print("\nFailure details:")
        for item in report["failure_details"]:
            print(f"- {item['test']}: {item['detail']}")
    else:
        print("\nFailure details: None")
    print(f"\nReports written to: {outputs}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
