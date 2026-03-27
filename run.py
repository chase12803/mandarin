#!/usr/bin/env python3
"""Cross-platform setup and run entrypoint for Mandarin."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
APP_FILE = PROJECT_ROOT / "app.py"


def log_step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def run_command(command: list[str], cwd: Path) -> None:
    command_str = " ".join(command)
    print(f"$ {command_str}", flush=True)
    try:
        subprocess.run(command, cwd=str(cwd), check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing required command: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {command_str}") from exc


def validate_project_layout() -> None:
    if not FRONTEND_DIR.is_dir():
        raise RuntimeError(f"Missing frontend directory: {FRONTEND_DIR}")
    if not REQUIREMENTS_FILE.is_file():
        raise RuntimeError(f"Missing requirements file: {REQUIREMENTS_FILE}")
    if not APP_FILE.is_file():
        raise RuntimeError(f"Missing app entrypoint: {APP_FILE}")


def venv_python_path() -> Path:
    if sys.platform.startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_prerequisites() -> str:
    npm_cmd = shutil.which("npm") or shutil.which("npm.cmd")
    if npm_cmd is None:
        raise RuntimeError("npm is not available in PATH. Install Node.js and npm first.")
    return npm_cmd


def ensure_virtualenv() -> Path:
    python_path = venv_python_path()
    if python_path.exists():
        return python_path

    log_step("Creating virtual environment")
    run_command([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=PROJECT_ROOT)

    if not python_path.exists():
        raise RuntimeError(f"Virtualenv was created, but Python was not found at: {python_path}")
    return python_path


def main() -> int:
    try:
        npm_cmd = ensure_prerequisites()
        validate_project_layout()

        log_step("Installing frontend dependencies")
        run_command([npm_cmd, "install"], cwd=FRONTEND_DIR)

        log_step("Building frontend")
        run_command([npm_cmd, "run", "build"], cwd=FRONTEND_DIR)

        venv_python = ensure_virtualenv()

        log_step("Installing backend dependencies")
        run_command([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], cwd=PROJECT_ROOT)

        log_step("Starting application")
        run_command([str(venv_python), str(APP_FILE)], cwd=PROJECT_ROOT)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
