"""RoboClaw VLA RLinf launcher — experimental worker orchestration.

Mirrors dexbotic.rl._embodied_cli. Not validated against a live RLinf
environment; treat as structural reference until integration testing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)  # [experimental]
except Exception:  # pragma: no cover - torch is optional for import/help tests
    mp = None  # type: ignore[assignment]

from roboclaw_vla.rl import registry


def _resolve_config_name(argv: list[str]) -> tuple[str, list[str]]:
    """Extract --config-name from argv, return (config_name, remaining_argv)."""

    config_name = "libero_10_grpo_roboclaw"
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--config-name="):
            config_name = argv[i].split("=", 1)[1]
        elif argv[i] == "--config-name" and i + 1 < len(argv):
            config_name = argv[i + 1]
            i += 1
        else:
            remaining.append(argv[i])
        i += 1
    return config_name, remaining


def run_roboclaw_rl(cfg: Any) -> None:
    # [experimental] Ray workers load this module via RLINF_EXT_MODULE and call register().
    os.environ.setdefault("RLINF_EXT_MODULE", "roboclaw_vla.rl.registry")
    registry.register_all()

    print("[RoboClaw RL] Launching with RLinf backend.")

    from omegaconf import OmegaConf
    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    loss_type = str(getattr(getattr(cfg, "algorithm", None), "loss_type", "grpo"))
    actor_placement = component_placement.get_strategy("actor")
    if loss_type == "embodied_sac":
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    else:
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

        actor_worker_cls = EmbodiedFSDPActor

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )
    runner.init_workers()
    runner.run()


def main() -> None:
    config_name, remaining = _resolve_config_name(sys.argv[1:])
    if "--help" in remaining or "-h" in remaining:
        parser = argparse.ArgumentParser(
            description="RoboClaw VLA RLinf launcher — experimental worker orchestration."
        )
        parser.add_argument("--config-name", default=config_name)
        parser.add_argument("--dataset_path", default=os.environ.get("RLINF_DATASET_PATH", ""))
        parser.add_argument("--checkpoint_path", default=os.environ.get("RLINF_CHECKPOINT_PATH", ""))
        parser.add_argument("--artifact_path", default=os.environ.get("RLINF_ARTIFACT_DIR", "outputs"))
        parser.print_help()
        return
    sys.argv[1:] = remaining

    import hydra
    from omegaconf import OmegaConf

    @hydra.main(
        version_base="1.1",
        config_path="../config/rl",
        config_name=config_name,
    )
    def _main(cfg: Any) -> None:
        # [experimental] Inject CLI dataset/checkpoint/artifact paths into cfg.
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dataset_path", default=os.environ.get("RLINF_DATASET_PATH", ""))
        parser.add_argument("--checkpoint_path", default=os.environ.get("RLINF_CHECKPOINT_PATH", ""))
        parser.add_argument("--artifact_path", default=os.environ.get("RLINF_ARTIFACT_DIR", "outputs"))
        known, _ = parser.parse_known_args()
        if known.dataset_path:
            OmegaConf.update(cfg, "env.train.dataset_path", known.dataset_path, merge=True, force_add=True)
        if known.checkpoint_path:
            OmegaConf.update(cfg, "actor.model.model_path", known.checkpoint_path, merge=True, force_add=True)
        if known.artifact_path:
            OmegaConf.update(cfg, "runner.logger.log_path", known.artifact_path, merge=True, force_add=True)
        run_roboclaw_rl(cfg)

    _main()


if __name__ == "__main__":
    main()
