import type { BodyDescriptor } from "../embodiment/body-adapter.ts";
import type { SkillContract } from "../execution/execution-adapter.ts";
import type { GoalRequest } from "../types.ts";
import type { PlanRequest, TaskPlan, TaskPlanner } from "./task-planner.ts";

const WAVE_PATTERN = /\b(wave|hello|greet|greeting)\b/i;

export function matchesSo101WaveInstruction(instruction: string): boolean {
  return WAVE_PATTERN.test(instruction);
}

export function buildSo101WaveSkill(goal: GoalRequest): SkillContract {
  return {
    skillId: "so101-wave",
    summary: "Make the simulated SO101 execute a bounded greeting wave",
    semanticAction: "arm.wave",
    requiredCapabilities: ["arm.wave"],
    supportedBackends: ["simulator"],
    parameters: {
      sequenceName: "greeting_wave",
      goalId: goal.goalId,
    },
    preconditions: ["body_is_online", "simulator_runtime_available", "so101_assets_installed"],
    successCriterion:
      "simulator completes the wave sequence and reports joint motion plus end-effector travel",
    failureCriterion: "simulator cannot load the SO101 assets or the wave motion does not complete",
  };
}

export function buildSo101WavePlan(goal: GoalRequest, body: BodyDescriptor): TaskPlan {
  if (!matchesSo101WaveInstruction(goal.instruction)) {
    throw new Error(
      "Round 3 planner only supports bounded SO101 greeting instructions like `wave`, `say hello`, or `greet`.",
    );
  }

  return {
    goal,
    skill: buildSo101WaveSkill(goal),
    rationale: `Planner mapped a bounded greeting instruction onto the SO101 wave sequence for ${body.bodyId}.`,
  };
}

export class So101WaveTaskPlanner implements TaskPlanner {
  async plan(input: PlanRequest): Promise<TaskPlan> {
    return buildSo101WavePlan(input.goal, input.body);
  }
}
