"""Quan ly vong doi tien trinh llama-server (single-resident BTC Q3).

Co che hoat dong
----------------
- 1 instance `llama-server` chay tren port co dinh (settings.llm.server.port).
- Khi swap_to(role) duoc goi:
    1. Kill tien trinh dang chay (SIGTERM, fallback SIGKILL).
    2. Spawn binary moi voi GGUF cua role yeu cau.
    3. Poll GET /v1/models toi khi 200 OK (max startup_timeout_s).

Vi sao port co dinh: ChatOpenAI client se cache base_url; chi can update
header `model` trong request khi role doi.

BTC Q5: llama-server tu da expose /v1/chat/completions OpenAI-compatible.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import time
from pathlib import Path
from typing import Literal, Optional

import httpx

from src.core.config import settings
from src.utils.logger import logger

Role = Literal["coder", "instruct"]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class LlamaServerSupervisor:
    """Spawn / kill / swap tien trinh llama-server.

    Usage:
        supervisor = LlamaServerSupervisor()
        supervisor.swap_to("coder")    # spawn lan dau
        supervisor.swap_to("instruct") # kill coder + spawn instruct
        supervisor.shutdown()          # cleanup khi app exit
    """

    def __init__(self) -> None:
        cfg = settings.llm.server
        self.binary: Path = self._resolve_path(cfg.binary)
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

        # Dam bao subprocess duoc cleanup ngay ca khi app crash.
        atexit.register(self.shutdown)

    # ── Public API ───────────────────────────────────────────────────

    @property
    def active_role(self) -> Optional[Role]:
        """Role dang resident, hoac None neu chua spawn."""
        return self._role

    def is_alive(self) -> bool:
        """True neu tien trinh con song."""
        return self._proc is not None and self._proc.poll() is None

    def swap_to(self, role: Role) -> None:
        """Dam bao llama-server dang chay voi GGUF cua role yeu cau.

        - Neu role da active va process con song -> no-op.
        - Nguoc lai -> kill process cu, spawn moi, cho /v1/models san sang.

        Raises:
            FileNotFoundError: GGUF hoac binary khong ton tai.
            TimeoutError: server khong ready sau startup_timeout_s.
            RuntimeError: tien trinh chet som lam khi spawn.
        """
        if self._role == role and self.is_alive():
            logger.debug(f"llama-server da active role={role}, bo qua swap.")
            return

        self._kill()
        self._spawn(role)
        self._wait_ready()

    def shutdown(self) -> None:
        """Dung tien trinh llama-server. Goi tu dong khi Python exit."""
        self._kill()

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_path(p: str) -> Path:
        """Anchor relative path vao project root."""
        path = Path(p)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path

    def _spawn(self, role: Role) -> None:
        """Spawn tien trinh llama-server moi voi GGUF cua role."""
        cfg = settings.llm.coder if role == "coder" else settings.llm.instruct
        model_path = self._resolve_path(cfg.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF khong ton tai: {model_path}\n"
                f"Hay tai weights ve thu muc models/."
            )
        if not self.binary.exists():
            raise FileNotFoundError(
                f"llama-server binary khong ton tai: {self.binary}\n"
                f"Chay scripts/install_llama_server.ps1 de duoc huong dan tai."
            )

        cmd = [
            str(self.binary),
            "--model", str(model_path),
            "--alias", cfg.model_name,             # ten xuat hien o /v1/models
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.n_ctx),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--jinja",                              # bat ChatML template
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

        # Tren Windows, dung CREATE_NEW_PROCESS_GROUP de terminate sach.
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
        """Terminate tien trinh dang chay, dong file log."""
        if self._proc is None:
            self._role = None
            return
        if self._proc.poll() is not None:
            # Da chet san.
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
            logger.warning("llama-server khong chiu terminate trong thoi gian timeout, SIGKILL.")
            self._proc.kill()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.error("llama-server van khong chet sau SIGKILL.")
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

    def _wait_ready(self) -> None:
        """Poll GET /v1/models toi khi 200 OK hoac timeout."""
        url = f"{self.base_url.rstrip('/')}/models"
        deadline = time.time() + self.startup_timeout
        last_err: Exception | None = None

        while time.time() < deadline:
            # Phat hien chet som de fail-fast thay vi cho 60s.
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server tien trinh chet som "
                    f"(exit={self._proc.returncode}). "
                    f"Xem log tai logs/llama-server-{self._role}.log"
                )
            try:
                r = httpx.get(url, timeout=1.5)
                if r.status_code == 200:
                    return
                last_err = RuntimeError(f"HTTP {r.status_code}")
            except httpx.HTTPError as e:
                last_err = e
            time.sleep(0.5)

        raise TimeoutError(
            f"llama-server khong ready sau {self.startup_timeout}s "
            f"(role={self._role}). Last error: {last_err}"
        )
