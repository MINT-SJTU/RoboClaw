import { describe, expect, it } from "vitest";
import {
  PyBulletExecutionAdapter,
  parsePyBulletSimulatorOutput,
} from "./pybullet-execution-adapter.ts";

describe("parsePyBulletSimulatorOutput", () => {
  it("extracts the marked JSON payload from noisy stdout", () => {
    const result = parsePyBulletSimulatorOutput(`
pybullet build time: Mar 11 2026 16:54:59
ROBOCLAW_PYBULLET_RESULT={"success":true,"simulator":"pybullet","robotModel":"husky","semanticAction":"base.move.forward","requestedMeters":0.5,"observedForwardMeters":0.61,"lateralDriftMeters":0.01,"startPosition":[0,0,0],"endPosition":[0.61,0.01,0],"steps":180,"detail":"movement completed"}
`);

    expect(result.success).toBe(true);
    expect(result.observedForwardMeters).toBe(0.61);
    expect(result.semanticAction).toBe("base.move.forward");
  });
});

describe("PyBulletExecutionAdapter", () => {
  it("maps simulator JSON into an execution receipt", async () => {
    const adapter = new PyBulletExecutionAdapter({
      pythonBin: "/tmp/fake-python",
      simulatorScript: "/tmp/fake-script.py",
      runner: async () => ({
        stdout: [
          "pybullet build time: Mar 11 2026 16:54:59",
          'ROBOCLAW_PYBULLET_RESULT={"success":true,"simulator":"pybullet","robotModel":"husky","semanticAction":"base.move.forward","requestedMeters":0.25,"observedForwardMeters":0.31,"lateralDriftMeters":0.0,"startPosition":[0,0,0],"endPosition":[0.31,0,0],"steps":120,"detail":"movement completed"}',
        ].join("\n"),
        stderr: "",
        code: 0,
      }),
    });

    const receipt = await adapter.execute({
      goal: {
        goalId: "goal-4",
        instruction: "move forward 25 cm",
      },
      body: {
        bodyId: "pybullet-husky",
        displayName: "PyBullet Husky",
        embodimentKind: "mobile_base",
        backendKind: "simulator",
        capabilities: ["base.move.forward"],
      },
      skill: {
        skillId: "simulator-move-forward",
        summary: "Move the simulated mobile base forward",
        semanticAction: "base.move.forward",
        requiredCapabilities: ["base.move.forward"],
        supportedBackends: ["simulator"],
        parameters: { meters: 0.25 },
        preconditions: ["body_is_online"],
        successCriterion: "simulator reports forward movement",
        failureCriterion: "simulator fails",
      },
    });

    expect(receipt.status).toBe("completed");
    expect(receipt.command).toEqual([
      "/tmp/fake-python",
      "/tmp/fake-script.py",
      "execute",
      '{"semanticAction":"base.move.forward","parameters":{"meters":0.25},"bodyId":"pybullet-husky","goalId":"goal-4"}',
    ]);
    expect(receipt.observation).toMatchObject({
      simulator: "pybullet",
      observedForwardMeters: 0.31,
      exitCode: 0,
    });
  });
});
