import { describe, expect, it } from "vitest";
import {
  PyBulletSo101ExecutionAdapter,
  parsePyBulletSo101SimulatorOutput,
} from "./pybullet-so101-execution-adapter.ts";

describe("parsePyBulletSo101SimulatorOutput", () => {
  it("extracts the marked SO101 simulator payload from noisy stdout", () => {
    const result = parsePyBulletSo101SimulatorOutput(`
pybullet build time: Mar 11 2026 16:54:59
ROBOCLAW_SO101_RESULT={"success":true,"simulator":"pybullet","robotModel":"so101","semanticAction":"arm.wave","sequenceName":"greeting_wave","jointOrder":["shoulder_pan","shoulder_lift"],"startJointPositions":{"shoulder_pan":0},"finalJointPositions":{"shoulder_pan":0.4},"peakJointExcursions":{"shoulder_pan":0.8},"endEffectorStart":[0,0,0],"endEffectorFinal":[0.02,0.01,0.18],"endEffectorPathMeters":0.22,"endEffectorLiftMeters":0.18,"steps":640,"assetSource":"SO101-Classic-Control@583899971f978b1b03664a1fa25dd377cfe429c6","detail":"wave completed"}
`);

    expect(result.success).toBe(true);
    expect(result.robotModel).toBe("so101");
    expect(result.endEffectorPathMeters).toBe(0.22);
  });
});

describe("PyBulletSo101ExecutionAdapter", () => {
  it("maps simulator JSON into a typed execution receipt", async () => {
    const adapter = new PyBulletSo101ExecutionAdapter({
      pythonBin: "/tmp/fake-python",
      simulatorScript: "/tmp/fake-script.py",
      urdfPath: "/tmp/so101_new_calib.urdf",
      traceDir: "/tmp/roboclaw-traces",
      runner: async () => ({
        stdout: [
          "pybullet build time: Mar 11 2026 16:54:59",
          'ROBOCLAW_SO101_RESULT={"success":true,"simulator":"pybullet","robotModel":"so101","semanticAction":"arm.wave","sequenceName":"greeting_wave","jointOrder":["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"],"startJointPositions":{"shoulder_pan":0},"finalJointPositions":{"shoulder_pan":0.4},"peakJointExcursions":{"shoulder_pan":0.8},"endEffectorStart":[0,0,0],"endEffectorFinal":[0.02,0.01,0.18],"endEffectorPathMeters":0.22,"endEffectorLiftMeters":0.18,"steps":640,"assetSource":"SO101-Classic-Control@583899971f978b1b03664a1fa25dd377cfe429c6","tracePath":"/tmp/roboclaw-traces/goal-wave-2.json","detail":"wave completed"}',
        ].join("\n"),
        stderr: "",
        code: 0,
      }),
      validateCommandPaths: false,
    });

    const receipt = await adapter.execute({
      goal: {
        goalId: "goal-wave-2",
        instruction: "wave hello",
      },
      body: {
        bodyId: "pybullet-so101",
        displayName: "PyBullet SO101",
        embodimentKind: "manipulator_arm",
        backendKind: "simulator",
        capabilities: ["arm.wave"],
      },
      skill: {
        skillId: "so101-wave",
        summary: "Wave the simulated SO101",
        semanticAction: "arm.wave",
        requiredCapabilities: ["arm.wave"],
        supportedBackends: ["simulator"],
        parameters: { sequenceName: "greeting_wave" },
        preconditions: ["body_is_online"],
        successCriterion: "wave completes",
        failureCriterion: "wave fails",
      },
    });

    expect(receipt.status).toBe("completed");
    expect(receipt.command).toEqual([
      "/tmp/fake-python",
      "/tmp/fake-script.py",
      "execute",
      '{"semanticAction":"arm.wave","parameters":{"sequenceName":"greeting_wave"},"goalId":"goal-wave-2","bodyId":"pybullet-so101","urdfPath":"/tmp/so101_new_calib.urdf","tracePath":"/tmp/roboclaw-traces/goal-wave-2.json"}',
    ]);
    expect(receipt.observation).toMatchObject({
      simulator: "pybullet",
      robotModel: "so101",
      endEffectorPathMeters: 0.22,
      exitCode: 0,
    });
  });
});
