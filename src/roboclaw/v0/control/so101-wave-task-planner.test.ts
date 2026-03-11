import { describe, expect, it } from "vitest";
import { So101WaveTaskPlanner, matchesSo101WaveInstruction } from "./so101-wave-task-planner.ts";

describe("matchesSo101WaveInstruction", () => {
  it("matches bounded greeting intents and rejects unrelated instructions", () => {
    expect(matchesSo101WaveInstruction("wave")).toBe(true);
    expect(matchesSo101WaveInstruction("say hello with the arm")).toBe(true);
    expect(matchesSo101WaveInstruction("move forward 0.5 meters")).toBe(false);
  });
});

describe("So101WaveTaskPlanner", () => {
  it("maps a greeting instruction into an SO101 wave skill", async () => {
    const planner = new So101WaveTaskPlanner();
    const plan = await planner.plan({
      goal: {
        goalId: "goal-wave-1",
        instruction: "wave hello",
      },
      body: {
        bodyId: "pybullet-so101",
        displayName: "PyBullet SO101",
        embodimentKind: "manipulator_arm",
        backendKind: "simulator",
        capabilities: ["arm.wave"],
      },
    });

    expect(plan.skill.semanticAction).toBe("arm.wave");
    expect(plan.skill.parameters).toEqual({
      sequenceName: "greeting_wave",
      goalId: "goal-wave-1",
    });
  });
});
