"""Run the paper's canonical recompute scripts and assert they reproduce every number.

These wrap the standalone verify/analyze scripts (stdlib-only) as pytest so the full
paper reproduction is part of the test suite and the CI gate. No GPU, no evolution.
"""
import subprocess
import sys

import pytest


def _run(paper_dir, *args):
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=str(paper_dir), capture_output=True, text=True,
    )
    return proc


def test_verify_results_reproduces_all_numbers(paper_dir):
    proc = _run(paper_dir, "scripts/runners/verify_results.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 failed" in proc.stdout and "RESULT:" in proc.stdout


def test_verify_scaling_tables_reproduces(paper_dir):
    proc = _run(paper_dir, "scripts/runners/verify_scaling_tables.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 failed" in proc.stdout


def test_analyze_hshg_reports_zero_solve(paper_dir):
    proc = _run(paper_dir, "scripts/analysis/analyze_hshg_results.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0% solve rate" in proc.stdout
