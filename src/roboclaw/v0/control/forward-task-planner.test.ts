import { describe, expect, it } from "vitest";
import { ForwardOnlyTaskPlanner, parseForwardInstruction } from "./forward-task-planner.ts";

describe("parseForwardInstruction", () => {
  it("parses meter and centimeter instructions deterministically", () => {
    expect(parseForwardInstruction("move forward 0.5 meters")).toBe(0.5);
    expect(parseForwardInstruction("move forward 25 cm")).toBe(0.25);
    expect(parseForwardInstruction("pick up the cube")).toBeNull();
  });
});

describe("ForwardOnlyTaskPlanner", () => {
  it("maps a bounded forward instruction into a semantic skill contract", async () => {
    const planner = new ForwardOnlyTaskPlanner();
    const plan = await planner.plan({
      goal: {
        goalId: "goal-5",
        instruction: "move forward 0.5 meters",
      },
      body: {
        bodyId: "pybullet-husky",
        displayName: "PyBullet Husky",
        embodimentKind: "mobile_base",
        backendKind: "simulator",
        capabilities: ["base.move.forward"],
      },
    });

    expect(plan.skill.semanticAction).toBe("base.move.forward");
    expect(plan.skill.parameters).toEqual({
      meters: 0.5,
      goalId: "goal-5",
    });
  });
});
