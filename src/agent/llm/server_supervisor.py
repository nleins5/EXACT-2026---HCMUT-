"""Manage the llama-server subprocess lifecycle (single-resident)."""
from __future__ import annotations

import atexit
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal, Optional

import httpx

from src.core.config import settings
from src.utils.logger import logger

Role = Literal["coder", "instruct"]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class LlamaServerSupervisor:
    """Spawn / kill / swap llama-server process."""

    def __init__(self) -> None:
        cfg = settings.llm.server
        self.binary: Path = self._resolve_path(cfg.binary, allow_which=True)
        self.host: str = cfg.host
        self.port: int = cfg.port
        self.base_url: str = cfg.base_url
        self.startup_timeout: int = cfg.startup_timeout_s
        self.shutdown_timeout: int = cfg.shutdown_timeout_s
        self.n_ctx: int = cfg.n_ctx
        self.n_gpu_layers: int = cfg.n_gpu_layers
        self.extra: list[str] = list(cfg.extra_args)

        self._proc: Optional[subprocess.Popen] = None
        self._role: Optional[Role] = None
        self._log_fh = None
        self._proc_lock = threading.RLock()
        self._abort_event = threading.Event()

        atexit.register(self.shutdown)

    # ── Public API ───────────────────────────────────────────────────

    @property
    def active_role(self) -> Optional[Role]:
        with self._proc_lock:
            return self._role

    def is_alive(self) -> bool:
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    def swap_to(self, role: Role) -> None:
        self._abort_event.clear()
        with self._proc_lock:
            if self._role == role and self._proc is not None and self._proc.poll() is None:
                logger.debug(f"llama-server already active for role={role}, skipping swap.")
                return

            self._kill()
            self._spawn(role)
        try:
            self._wait_ready(role)
            if self._abort_event.is_set():
                raise RuntimeError("llama-server startup was aborted.")
        except Exception:
            with self._proc_lock:
                self._kill()
            raise

    def shutdown(self) -> None:
        self._abort_event.set()
        with self._proc_lock:
            self._kill()

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_path(p: str, *, allow_which: bool = False) -> Path:
        path = Path(p)
        if not path.is_absolute():
            if allow_which and "/" not in p and "\\" not in p:
                import shutil
                found = shutil.which(p)
                if found:
                    return Path(found)
            path = _PROJECT_ROOT / path
        return path

    def _spawn(self, role: Role) -> None:
        cfg = settings.llm.coder if role == "coder" else settings.llm.instruct
        model_path = self._resolve_path(cfg.model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"GGUF not found: {model_path}")
        if not self.binary.exists():
            raise FileNotFoundError(f"llama-server binary not found: {self.binary}")

        cmd = [
            str(self.binary),
            "--model", str(model_path),
            "--alias", cfg.model_name,             # ten xuat hien o /v1/models
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.n_ctx),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--jinja",
            *self.extra,
        ]
        logger.info(
            f"Spawning llama-server (role={role}): "
            f"{' '.join(str(c) for c in cmd)}"
        )

        log_dir = _PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"llama-server-{role}.log"
        self._log_fh = open(log_path, "ab")

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        self._role = role

    def _kill(self) -> None:
        if self._proc is None:
            self._role = None
            self._close_log()
            return
        if self._proc.poll() is not None:
            self._proc = None
            self._role = None
            self._close_log()
            return

        logger.info(
            f"Killing llama-server (pid={self._proc.pid}, role={self._role})..."
        )
        try:
            self._proc.terminate()
            self._proc.wait(timeout=self.shutdown_timeout)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not respond to SIGTERM, sending SIGKILL.")
            self._proc.kill()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.error("llama-server still alive after SIGKILL.")
        finally:
            self._close_log()
            self._proc = None
            self._role = None

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def _wait_ready(self, role: Role) -> None:
        url = f"{self.base_url.rstrip('/')}/models"
        cfg = settings.llm.coder if role == "coder" else settings.llm.instruct
        expected_model = cfg.model_name
        deadline = time.time() + self.startup_timeout
        last_err: Exception | None = None

        while time.time() < deadline:
            if self._abort_event.is_set():
                raise RuntimeError("llama-server startup was aborted.")
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server died early (exit={self._proc.returncode}). "
                    f"Check logs/llama-server-{self._role}.log"
                )
            try:
                r = httpx.get(url, timeout=1.5)
                if r.status_code == 200:
                    payload = r.json()
                    if not isinstance(payload, dict):
                        raise ValueError("llama-server /models returned a non-object payload")
                    model_ids = {
                        str(item.get("id") or item.get("model") or item.get("name"))
                        for key in ("data", "models")
                        for item in payload.get(key, [])
                        if isinstance(item, dict)
                    }
                    if expected_model in model_ids:
                        return
                    last_err = RuntimeError(
                        f"HTTP 200 but expected model {expected_model!r} was not active"
                    )
                    time.sleep(0.5)
                    continue
                last_err = RuntimeError(f"HTTP {r.status_code}")
            except (httpx.HTTPError, ValueError, TypeError) as e:
                last_err = e
            time.sleep(0.5)

        raise TimeoutError(
            f"llama-server not ready after {self.startup_timeout}s "
            f"(role={self._role}). Last error: {last_err}"
        )
