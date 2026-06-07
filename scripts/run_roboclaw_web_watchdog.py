#!/usr/bin/env python3
"""Run RoboClaw Web UI with a small local watchdog.

This is intentionally tiny: local dev/demo deployments should not strand the
browser with "Failed to fetch" just because the web process exited.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RoboClaw Web UI and restart it if it exits.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8766")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--log", default="/private/tmp/roboclaw_web_8766.log")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    log_path = Path(args.log).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop = False
    child: subprocess.Popen[bytes] | None = None

    def _stop(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True
        if child and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    backoff_s = 1.0
    while not stop:
        with log_path.open("ab") as log_file:
            log_file.write(f"\n[watchdog] starting RoboClaw web on {args.host}:{args.port}\n".encode())
            log_file.flush()
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "roboclaw.http.server",
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                ],
                cwd=str(repo),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            exit_code = child.wait()
            log_file.write(f"\n[watchdog] RoboClaw web exited with {exit_code}\n".encode())
            log_file.flush()
        if stop:
            break
        time.sleep(backoff_s)
        backoff_s = min(backoff_s * 1.5, 15.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
