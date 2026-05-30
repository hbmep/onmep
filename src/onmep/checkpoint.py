import os
import json
import pickle
import tempfile
from dataclasses import dataclass


@dataclass
class CheckpointManager:
    root: str   # e.g. /.../reports/run_XYZ/checkpoints

    def __post_init__(self):
        os.makedirs(self.root, exist_ok=True)
        self.latest_path = os.path.join(self.root, "latest.json")
        return

    def _step_path(self, step: int) -> str:
        return os.path.join(self.root, f"step_{step:03d}.pkl")

    def latest(self):
        try:
            with open(self.latest_path, "r") as f:
                meta = json.load(f)
            p = meta.get("path")
            if p and os.path.isfile(p):
                return meta
        except Exception:
            pass
        return None

    def save(
        self,
        *,
        step: int,
        agent_state: dict,
        env_state: dict,
        extra_meta: dict | None = None
    ):
        payload = {
            "step": int(step),
            "agent_state": agent_state,
            "env_state": env_state,
            "meta": (extra_meta or {}),
            "version": 1,
        }
        output_path = self._step_path(step)

        # Atomic write
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as tmp:
            pickle.dump(payload, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path = tmp.name
        os.replace(tmp_path, output_path)

        # Update latest pointer
        meta = {"path": output_path, "step": int(step)}
        with tempfile.NamedTemporaryFile(dir=self.root, mode="w", delete=False) as tmp:
            json.dump(meta, tmp)
            tmp_path = tmp.name
        os.replace(tmp_path, self.latest_path)
        return

    def load_latest(self) -> dict[str] | None:
        """Load last successful checkpoint."""
        meta = self.latest()
        if not meta:
            return None
        with open(meta["path"], "rb") as f:
            return pickle.load(f)

    def load(self, path: str):
        """Load an arbitrary checkpoint file by absolute path."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_step(self, step: int):
        """Load a checkpoint saved at a specific step number."""
        p = self._step_path(step)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Checkpoint not found for step={step}: {p}")
        with open(p, "rb") as f:
            return pickle.load(f)
