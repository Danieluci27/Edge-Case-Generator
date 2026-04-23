"""Practical isolated execution helpers."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from edge_case_generator.types import ExecutionResult, ExecutionStatus


class CodeExecutor:
    """Run Python snippets in temporary subprocesses with timeouts."""

    def __init__(self, timeout_sec: float = 1.0, python_executable: str = "python") -> None:
        self.timeout_sec = timeout_sec
        self.python_executable = python_executable

    def run(self, code: str, candidate_input: str, valid_input: bool = True) -> ExecutionResult:
        """Execute code with candidate input and capture stdout and stderr."""

        if not valid_input:
            return ExecutionResult(
                status=ExecutionStatus.INVALID_INPUT,
                stdout="",
                stderr="Input validation failed before execution",
                return_code=None,
                runtime_sec=0.0,
                timed_out=False,
                valid_input=False,
                parsed_output=None,
            )

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="edge_case_exec_") as temp_dir:
            program_path = Path(temp_dir) / "program.py"
            program_path.write_text(code, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [self.python_executable, str(program_path)],
                    input=candidate_input,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    cwd=temp_dir,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                runtime = time.perf_counter() - started
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    return_code=None,
                    runtime_sec=runtime,
                    timed_out=True,
                    valid_input=True,
                    parsed_output=None,
                )

        runtime = time.perf_counter() - started
        stdout = completed.stdout
        stderr = completed.stderr
        normalized_output = stdout.strip()
        status = ExecutionStatus.SUCCESS

        if completed.returncode != 0:
            lowered = stderr.lower()
            if "assert" in lowered:
                status = ExecutionStatus.ASSERTION_FAILURE
            elif completed.returncode < 0:
                status = ExecutionStatus.CRASH
            else:
                status = ExecutionStatus.RUNTIME_ERROR

        return ExecutionResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            return_code=completed.returncode,
            runtime_sec=runtime,
            timed_out=False,
            valid_input=True,
            parsed_output=normalized_output if status == ExecutionStatus.SUCCESS else None,
        )

