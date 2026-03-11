import { describe, expect, it } from "vitest";
import type { BodyDescriptor } from "./embodiment/body-adapter.ts";
import type { SkillContract } from "./execution/execution-adapter.ts";
import { runGoal, SupervisorRejectionError } from "./runtime.ts";
import { StaticBodyAdapter, StaticTaskPlanner, StubExecutionAdapter } from "./stubs.ts";

const simulatorBody: BodyDescriptor = {
  bodyId: "sim-base-1",
  displayName: "Sim Base 1",
  embodimentKind: "mobile_base",
  backendKind: "simulator",
  capabilities: ["base.move.forward", "base.stop"],
};

function buildMoveForwardSkill(
  supportedBackends: readonly ("hardware" | "simulator")[],
): SkillContract {
  return {
    skillId: "move-forward",
    summary: "Move the mobile base forward a short distance",
    semanticAction: "base.move.forward",
    requiredCapabilities: ["base.move.forward"],
    supportedBackends,
    parameters: { meters: 0.25 },
    preconditions: ["body_is_online"],
    successCriterion: "base reaches the requested forward offset",
    failureCriterion: "movement times out or is interrupted",
  };
}

describe("runGoal", () => {
  it("executes the smallest validated path", async () => {
    const executor = new StubExecutionAdapter();

    const result = await runGoal(
      {
        goalId: "goal-1",
        instruction: "move forward 25 cm",
      },
      {
        body: new StaticBodyAdapter(simulatorBody),
        planner: new StaticTaskPlanner(buildMoveForwardSkill(["simulator", "hardware"])),
        executor,
      },
    );

    expect(result.plan.skill.semanticAction).toBe("base.move.forward");
    expect(result.audit.approved).toBe(true);
    expect(result.audit.checkedConstraints).toEqual([
      "required_capabilities",
      "supported_backends",
    ]);
    expect(result.receipt).toMatchObject({
      status: "completed",
      backendKind: "simulator",
    });
    expect(executor.requests).toHaveLength(1);
  });

  it("rejects a planned skill that targets the wrong backend", async () => {
    const executor = new StubExecutionAdapter();
    let rejection: unknown;

    try {
      await runGoal(
        {
          goalId: "goal-2",
          instruction: "move forward 25 cm",
        },
        {
          body: new StaticBodyAdapter(simulatorBody),
          planner: new StaticTaskPlanner(buildMoveForwardSkill(["hardware"])),
          executor,
        },
      );
    } catch (error) {
      rejection = error;
    }

    expect(rejection).toBeInstanceOf(SupervisorRejectionError);

    const typedRejection = rejection as SupervisorRejectionError;

    expect(typedRejection.audit.approved).toBe(false);
    expect(typedRejection.audit.backendApproved).toBe(false);
    expect(typedRejection.audit.reasons).toEqual([
      "Backend simulator is not supported by skill move-forward",
    ]);
    expect(executor.requests).toHaveLength(0);
  });

  it("rejects a planned skill when the body lacks the required capability", async () => {
    const executor = new StubExecutionAdapter();
    let rejection: unknown;

    try {
      await runGoal(
        {
          goalId: "goal-3",
          instruction: "move forward 25 cm",
        },
        {
          body: new StaticBodyAdapter({
            ...simulatorBody,
            bodyId: "sim-base-2",
            capabilities: ["base.stop"],
          }),
          planner: new StaticTaskPlanner(buildMoveForwardSkill(["simulator", "hardware"])),
          executor,
        },
      );
    } catch (error) {
      rejection = error;
    }

    expect(rejection).toBeInstanceOf(SupervisorRejectionError);

    const typedRejection = rejection as SupervisorRejectionError;

    expect(typedRejection.audit.approved).toBe(false);
    expect(typedRejection.audit.backendApproved).toBe(true);
    expect(typedRejection.audit.missingCapabilities).toEqual(["base.move.forward"]);
    expect(typedRejection.audit.reasons).toEqual([
      "Missing required capabilities: base.move.forward",
    ]);
    expect(executor.requests).toHaveLength(0);
  });
});
