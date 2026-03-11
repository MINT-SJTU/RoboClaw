import { describe, expect, it, vi } from "vitest";
import { createRoboClawHandlers } from "./roboclaw.js";

describe("roboclawHandlers", () => {
  it("validates instruction params before invoking RoboClaw runtime", async () => {
    const runSo101Goal = vi.fn();
    const respond = vi.fn();
    const handlers = createRoboClawHandlers({ runSo101Goal });

    await handlers["roboclaw.so101.run"]({
      req: { type: "req", id: "req-1", method: "roboclaw.so101.run" },
      params: {},
      client: null,
      isWebchatConnect: () => false,
      respond,
      context: {} as never,
    });

    expect(runSo101Goal).not.toHaveBeenCalled();
    expect(respond).toHaveBeenCalledWith(
      false,
      undefined,
      expect.objectContaining({
        code: "INVALID_REQUEST",
      }),
    );
  });

  it("routes a bounded instruction through the RoboClaw runtime", async () => {
    const runSo101Goal = vi.fn(async () => ({
      body: {
        bodyId: "pybullet-so101",
        displayName: "PyBullet SO101",
        embodimentKind: "manipulator_arm",
        backendKind: "simulator" as const,
        capabilities: ["arm.wave"],
      },
      plan: {
        goal: {
          goalId: "goal-wave-3",
          instruction: "wave hello",
        },
        skill: {
          skillId: "so101-wave",
          summary: "Wave the simulated SO101",
          semanticAction: "arm.wave",
          requiredCapabilities: ["arm.wave"],
          supportedBackends: ["simulator" as const],
          parameters: { sequenceName: "greeting_wave" },
          preconditions: ["body_is_online"],
          successCriterion: "wave completes",
          failureCriterion: "wave fails",
        },
        rationale: "planner matched a bounded wave instruction",
      },
      audit: {
        approved: true,
        backendApproved: true,
        checkedConstraints: ["required_capabilities", "supported_backends"],
        missingCapabilities: [],
        reasons: [],
      },
      receipt: {
        executionId: "pybullet-so101-goal-wave-3",
        status: "completed" as const,
        backendKind: "simulator" as const,
        detail: "wave completed",
        observation: {
          robotModel: "so101",
        },
      },
    }));
    const respond = vi.fn();
    const handlers = createRoboClawHandlers({ runSo101Goal });

    await handlers["roboclaw.so101.run"]({
      req: { type: "req", id: "req-2", method: "roboclaw.so101.run" },
      params: { instruction: "wave hello", goalId: "goal-wave-3" },
      client: null,
      isWebchatConnect: () => false,
      respond,
      context: {} as never,
    });

    expect(runSo101Goal).toHaveBeenCalledWith({
      instruction: "wave hello",
      goalId: "goal-wave-3",
    });
    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        goalId: "goal-wave-3",
        receipt: expect.objectContaining({
          status: "completed",
        }),
      }),
      undefined,
    );
  });
});
