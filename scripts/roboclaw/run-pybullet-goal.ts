#!/usr/bin/env -S node --import tsx

import path from "node:path";
import { fileURLToPath } from "node:url";
import { ForwardOnlyTaskPlanner } from "../../src/roboclaw/v0/control/forward-task-planner.ts";
import { PyBulletHuskyBodyAdapter } from "../../src/roboclaw/v0/embodiment/pybullet-husky-body.ts";
import { PyBulletExecutionAdapter } from "../../src/roboclaw/v0/execution/pybullet-execution-adapter.ts";
import { runGoal } from "../../src/roboclaw/v0/runtime.ts";

function parseInstruction(argv: string[]): string {
  const flagIndex = argv.indexOf("--instruction");
  if (flagIndex >= 0) {
    const value = argv[flagIndex + 1];
    if (value) {
      return value;
    }
  }

  return "move forward 0.5 meters";
}

async function main(argv: string[]): Promise<number> {
  const instruction = parseInstruction(argv);
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
  const pythonBin =
    process.env.ROBOCLAW_PYBULLET_PYTHON ??
    path.join(
      repoRoot,
      process.platform === "win32"
        ? ".venv-pybullet/Scripts/python.exe"
        : ".venv-pybullet/bin/python",
    );

  const result = await runGoal(
    {
      goalId: "round2-pybullet-goal",
      instruction,
    },
    {
      body: new PyBulletHuskyBodyAdapter(),
      planner: new ForwardOnlyTaskPlanner(),
      executor: new PyBulletExecutionAdapter({ pythonBin }),
    },
  );

  console.log(
    JSON.stringify(
      {
        goal: result.plan.goal,
        skill: result.plan.skill,
        audit: result.audit,
        receipt: result.receipt,
      },
      null,
      2,
    ),
  );

  return result.receipt.status === "completed" ? 0 : 1;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main(process.argv.slice(2))
    .then((code) => {
      process.exitCode = code;
    })
    .catch((error: unknown) => {
      const detail = error instanceof Error ? error.message : String(error);
      console.error(detail);
      process.exitCode = 1;
    });
}
