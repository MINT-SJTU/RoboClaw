"""RoboClaw VLA eval loop — experimental.

Runs policy in eval env and writes success_rate to eval_info.json.
Mirrors RLinf only_eval mode. Not validated on live hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def run_eval(cfg: Any) -> dict[str, Any]:
    # [experimental]
    from omegaconf import OmegaConf
    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    cfg = validate_cfg(cfg)
    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=component_placement.get_strategy("actor"),
    )
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=component_placement.get_strategy("rollout"),
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=component_placement.get_strategy("env"),
    )
    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )
    runner.init_workers()
    OmegaConf.update(cfg, "runner.only_eval", True, merge=True)
    runner.run()
    log_path = Path(cfg.runner.logger.log_path)
    for candidate in ("eval_results.json", "metrics.json", "eval_info.json"):
        p = log_path / candidate
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {"success_rate": None, "log_path": str(log_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--artifact_path", required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--config_name", default="libero_10_grpo_roboclaw")
    args = parser.parse_args()

    artifact_path = Path(args.artifact_path)
    artifact_path.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "checkpointPath": args.checkpoint_path,
        "suite": args.suite,
        "success_rate": None,
        "implemented": True,
        "experimental": True,
    }
    try:
        import hydra
        from omegaconf import OmegaConf

        with hydra.initialize_config_dir(
            config_dir=str(Path(__file__).parent.parent / "config" / "rl"),
            job_name="roboclaw_eval",
            version_base="1.1",
        ):
            cfg = hydra.compose(config_name=args.config_name)
            if args.checkpoint_path:
                OmegaConf.update(cfg, "actor.model.model_path", args.checkpoint_path, merge=True, force_add=True)
            OmegaConf.update(cfg, "runner.logger.log_path", args.artifact_path, merge=True, force_add=True)
            eval_result = run_eval(cfg)
            result["success_rate"] = eval_result.get("success_rate")
            result["eval_result"] = eval_result
    except Exception as exc:
        result["error"] = str(exc)

    (artifact_path / "eval_info.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
