import { randomUUID } from "node:crypto";
import { So101WaveTaskPlanner } from "./control/so101-wave-task-planner.ts";
import type { ExecutionSupervisor } from "./control/supervisor.ts";
import type { TaskPlanner } from "./control/task-planner.ts";
import type { BodyAdapter } from "./embodiment/body-adapter.ts";
import { PyBulletSo101BodyAdapter } from "./embodiment/pybullet-so101-body.ts";
import type { ExecutionAdapter, ExecutionReceipt } from "./execution/execution-adapter.ts";
import {
  PyBulletSo101ExecutionAdapter,
  type PyBulletSo101ExecutionAdapterOptions,
} from "./execution/pybullet-so101-execution-adapter.ts";
import { runGoal, type GoalRunResult } from "./runtime.ts";

export type So101RuntimeOverrides = {
  body?: BodyAdapter;
  planner?: TaskPlanner;
  executor?: ExecutionAdapter;
  supervisor?: ExecutionSupervisor;
};

export type RunSo101GoalInput = {
  instruction: string;
  goalId?: string;
  executorOptions?: PyBulletSo101ExecutionAdapterOptions;
};

export type RunSo101GoalResult = GoalRunResult & {
  receipt: ExecutionReceipt;
};

export async function runSo101Goal(
  input: RunSo101GoalInput,
  overrides: So101RuntimeOverrides = {},
): Promise<RunSo101GoalResult> {
  const goalId = input.goalId?.trim() || randomUUID();

  return (await runGoal(
    {
      goalId,
      instruction: input.instruction,
    },
    {
      body: overrides.body ?? new PyBulletSo101BodyAdapter(),
      planner: overrides.planner ?? new So101WaveTaskPlanner(),
      executor: overrides.executor ?? new PyBulletSo101ExecutionAdapter(input.executorOptions),
      supervisor: overrides.supervisor,
    },
  )) as RunSo101GoalResult;
}
