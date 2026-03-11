import { access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileUtf8 } from "../../../daemon/exec-file.ts";
import type { ExecutionAdapter, ExecutionReceipt, ExecutionRequest } from "./execution-adapter.ts";

const RESULT_MARKER = "ROBOCLAW_PYBULLET_RESULT=";

export type PyBulletCommandResult = {
  stdout: string;
  stderr: string;
  code: number;
};

export type PyBulletCommandRunner = (
  command: string,
  args: readonly string[],
) => Promise<PyBulletCommandResult>;

export type PyBulletSimulatorOutput = {
  success: boolean;
  simulator: "pybullet";
  robotModel: "husky";
  semanticAction: string;
  requestedMeters: number;
  observedForwardMeters: number;
  lateralDriftMeters: number;
  startPosition: [number, number, number];
  endPosition: [number, number, number];
  steps: number;
  detail: string;
};

export type PyBulletExecutionAdapterOptions = {
  pythonBin?: string;
  simulatorScript?: string;
  runner?: PyBulletCommandRunner;
  validateCommandPaths?: boolean;
};

function defaultPythonBin(): string {
  if (process.env.ROBOCLAW_PYBULLET_PYTHON) {
    return process.env.ROBOCLAW_PYBULLET_PYTHON;
  }

  if (process.platform === "win32") {
    return path.resolve(process.cwd(), ".venv-pybullet", "Scripts", "python.exe");
  }

  return path.resolve(process.cwd(), ".venv-pybullet", "bin", "python");
}

function defaultSimulatorScript(): string {
  return fileURLToPath(
    new URL("../../../../scripts/roboclaw/pybullet_husky_sim.py", import.meta.url),
  );
}

export function parsePyBulletSimulatorOutput(stdout: string): PyBulletSimulatorOutput {
  const markerIndex = stdout.lastIndexOf(RESULT_MARKER);
  if (markerIndex < 0) {
    throw new Error("PyBullet simulator output did not contain a RoboClaw result marker.");
  }

  const payload = stdout.slice(markerIndex + RESULT_MARKER.length).trim();
  return JSON.parse(payload) as PyBulletSimulatorOutput;
}

async function runPyBulletCommand(
  command: string,
  args: readonly string[],
): Promise<PyBulletCommandResult> {
  const result = await execFileUtf8(command, [...args], {
    cwd: process.cwd(),
    maxBuffer: 10 * 1024 * 1024,
  });

  return {
    stdout: result.stdout,
    stderr: result.stderr,
    code: result.code,
  };
}

export class PyBulletExecutionAdapter implements ExecutionAdapter {
  private readonly pythonBin: string;
  private readonly simulatorScript: string;
  private readonly runner: PyBulletCommandRunner;
  private readonly validateCommandPaths: boolean;

  constructor(options: PyBulletExecutionAdapterOptions = {}) {
    this.pythonBin = options.pythonBin ?? defaultPythonBin();
    this.simulatorScript =
      options.simulatorScript ??
      process.env.ROBOCLAW_PYBULLET_SIM_SCRIPT ??
      defaultSimulatorScript();
    this.runner = options.runner ?? runPyBulletCommand;
    this.validateCommandPaths = options.validateCommandPaths ?? options.runner === undefined;
  }

  async execute(request: ExecutionRequest): Promise<ExecutionReceipt> {
    if (request.skill.semanticAction !== "base.move.forward") {
      return {
        executionId: `pybullet-unsupported-${request.goal.goalId}`,
        status: "failed",
        backendKind: request.body.backendKind,
        detail: `PyBulletExecutionAdapter does not support semantic action ${request.skill.semanticAction}.`,
      };
    }

    if (this.validateCommandPaths) {
      await access(this.pythonBin);
      await access(this.simulatorScript);
    }

    const command = [
      this.pythonBin,
      this.simulatorScript,
      "execute",
      JSON.stringify({
        semanticAction: request.skill.semanticAction,
        parameters: request.skill.parameters,
        bodyId: request.body.bodyId,
        goalId: request.goal.goalId,
      }),
    ] as const;

    const result = await this.runner(command[0], command.slice(1));
    const simulation = parsePyBulletSimulatorOutput(result.stdout);

    return {
      executionId: `pybullet-${request.goal.goalId}`,
      status: simulation.success && result.code === 0 ? "completed" : "failed",
      backendKind: request.body.backendKind,
      detail: simulation.detail,
      command,
      observation: {
        ...simulation,
        exitCode: result.code,
        stderr: result.stderr.trim() || undefined,
      },
    };
  }
}
