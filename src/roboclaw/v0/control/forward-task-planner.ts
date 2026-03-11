import type { BodyDescriptor } from "../embodiment/body-adapter.ts";
import type { SkillContract } from "../execution/execution-adapter.ts";
import type { GoalRequest } from "../types.ts";
import type { PlanRequest, TaskPlan, TaskPlanner } from "./task-planner.ts";

const MOVE_FORWARD_PATTERN =
  /\bmove\s+forward\s+(?<value>\d+(?:\.\d+)?)\s*(?<unit>m|meter|meters|cm|centimeter|centimeters)\b/i;

export function parseForwardInstruction(instruction: string): number | null {
  const match = instruction.match(MOVE_FORWARD_PATTERN);
  const value = match?.groups?.value;
  const unit = match?.groups?.unit?.toLowerCase();

  if (value === undefined || unit === undefined) {
    return null;
  }

  const parsedValue = Number(value);
  if (!Number.isFinite(parsedValue) || parsedValue <= 0) {
    return null;
  }

  if (unit === "cm" || unit === "centimeter" || unit === "centimeters") {
    return parsedValue / 100;
  }

  return parsedValue;
}

export function buildForwardSkill(goal: GoalRequest, meters: number): SkillContract {
  return {
    skillId: "simulator-move-forward",
    summary: `Move the simulated mobile base forward ${meters.toFixed(2)} meters`,
    semanticAction: "base.move.forward",
    requiredCapabilities: ["base.move.forward"],
    supportedBackends: ["simulator"],
    parameters: {
      meters,
      goalId: goal.goalId,
    },
    preconditions: ["body_is_online", "simulator_runtime_available"],
    successCriterion: `simulator reports at least ${meters.toFixed(2)} meters of forward travel`,
    failureCriterion: "simulator rejects the action or does not reach the requested distance",
  };
}

export class ForwardOnlyTaskPlanner implements TaskPlanner {
  constructor(private readonly maxMeters = 2) {}

  async plan(input: PlanRequest): Promise<TaskPlan> {
    return buildForwardPlan(input.goal, input.body, this.maxMeters);
  }
}

export function buildForwardPlan(
  goal: GoalRequest,
  body: BodyDescriptor,
  maxMeters: number,
): TaskPlan {
  const meters = parseForwardInstruction(goal.instruction);

  if (meters === null) {
    throw new Error(
      "Round 2 planner only supports instructions like `move forward 0.5 meters` or `move forward 25 cm`.",
    );
  }

  if (meters > maxMeters) {
    throw new Error(`Round 2 planner only supports forward distances up to ${maxMeters} meters.`);
  }

  return {
    goal,
    skill: buildForwardSkill(goal, meters),
    rationale: `Planner matched a bounded forward-motion instruction for ${body.bodyId}.`,
  };
}
