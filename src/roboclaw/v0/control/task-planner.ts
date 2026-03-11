import type { BodyDescriptor } from "../embodiment/body-adapter.ts";
import type { SkillContract } from "../execution/execution-adapter.ts";
import type { GoalRequest } from "../types.ts";

export type PlanRequest = {
  goal: GoalRequest;
  body: BodyDescriptor;
};

export type TaskPlan = {
  goal: GoalRequest;
  skill: SkillContract;
  rationale: string;
};

export interface TaskPlanner {
  plan(input: PlanRequest): Promise<TaskPlan>;
}
