import { ExecutionSupervisor, type SupervisorAuditRecord } from "./control/supervisor.ts";
import type { TaskPlan, TaskPlanner } from "./control/task-planner.ts";
import type { BodyAdapter, BodyDescriptor } from "./embodiment/body-adapter.ts";
import type { ExecutionAdapter, ExecutionReceipt } from "./execution/execution-adapter.ts";
import type { GoalRequest } from "./types.ts";

export type RoboClawV0RuntimeDeps = {
  body: BodyAdapter;
  planner: TaskPlanner;
  executor: ExecutionAdapter;
  supervisor?: ExecutionSupervisor;
};

export type GoalRunResult = {
  body: BodyDescriptor;
  plan: TaskPlan;
  audit: SupervisorAuditRecord;
  receipt: ExecutionReceipt;
};

export class SupervisorRejectionError extends Error {
  constructor(readonly audit: SupervisorAuditRecord) {
    super(audit.reasons.join("; "));
    this.name = "SupervisorRejectionError";
  }
}

export async function runGoal(
  goal: GoalRequest,
  deps: RoboClawV0RuntimeDeps,
): Promise<GoalRunResult> {
  const body = await deps.body.describeBody();
  const plan = await deps.planner.plan({ goal, body });
  const supervisor = deps.supervisor ?? new ExecutionSupervisor();
  const audit = supervisor.review({ body, skill: plan.skill });

  if (!audit.approved) {
    throw new SupervisorRejectionError(audit);
  }

  const receipt = await deps.executor.execute({ goal, body, skill: plan.skill });

  return {
    body,
    plan,
    audit,
    receipt,
  };
}
