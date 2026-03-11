import { access, mkdir } from "node:fs/promises";
import path from "node:path";
import { execFileUtf8 } from "../../../daemon/exec-file.ts";
import { resolveOpenClawPackageRootSync } from "../../../infra/openclaw-root.ts";
import type { ExecutionAdapter, ExecutionReceipt, ExecutionRequest } from "./execution-adapter.ts";

const RESULT_MARKER = "ROBOCLAW_SO101_RESULT=";

function resolvePackageRootPath(relativePath: string): string {
  const packageRoot =
    resolveOpenClawPackageRootSync({
      cwd: process.cwd(),
      argv1: process.argv[1],
      moduleUrl: import.meta.url,
    }) ?? process.cwd();
  return path.resolve(packageRoot, relativePath);
}

const DEFAULT_ASSET_ROOT = resolvePackageRootPath(".local/roboclaw/so101-classic-control");
const DEFAULT_TRACE_DIR = resolvePackageRootPath(".local/roboclaw/so101-traces");

export type PyBulletSo101CommandResult = {
  stdout: string;
  stderr: string;
  code: number;
};

export type PyBulletSo101CommandRunner = (
  command: string,
  args: readonly string[],
) => Promise<PyBulletSo101CommandResult>;

export type PyBulletSo101SimulatorOutput = {
  success: boolean;
  simulator: "pybullet";
  robotModel: "so101";
  semanticAction: string;
  sequenceName: string;
  jointOrder: string[];
  startJointPositions: Record<string, number>;
  finalJointPositions: Record<string, number>;
  peakJointExcursions: Record<string, number>;
  endEffectorStart: [number, number, number];
  endEffectorFinal: [number, number, number];
  endEffectorPathMeters: number;
  endEffectorLiftMeters: number;
  steps: number;
  assetSource: string;
  tracePath?: string;
  detail: string;
};

export type PyBulletSo101ExecutionAdapterOptions = {
  pythonBin?: string;
  simulatorScript?: string;
  urdfPath?: string;
  traceDir?: string;
  runner?: PyBulletSo101CommandRunner;
  validateCommandPaths?: boolean;
};

function defaultPythonBin(): string {
  if (process.env.ROBOCLAW_PYBULLET_PYTHON) {
    return process.env.ROBOCLAW_PYBULLET_PYTHON;
  }

  if (process.platform === "win32") {
    return resolvePackageRootPath(".venv-pybullet/Scripts/python.exe");
  }

  return resolvePackageRootPath(".venv-pybullet/bin/python");
}

function defaultSimulatorScript(): string {
  return resolvePackageRootPath("scripts/roboclaw/pybullet_so101_sim.py");
}

function defaultUrdfPath(): string {
  return path.resolve(DEFAULT_ASSET_ROOT, "so101_new_calib.urdf");
}

export function parsePyBulletSo101SimulatorOutput(stdout: string): PyBulletSo101SimulatorOutput {
  const markerIndex = stdout.lastIndexOf(RESULT_MARKER);
  if (markerIndex < 0) {
    throw new Error("SO101 simulator output did not contain a RoboClaw result marker.");
  }

  const payload = stdout.slice(markerIndex + RESULT_MARKER.length).trim();
  return JSON.parse(payload) as PyBulletSo101SimulatorOutput;
}

async function runPyBulletSo101Command(
  command: string,
  args: readonly string[],
): Promise<PyBulletSo101CommandResult> {
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

export class PyBulletSo101ExecutionAdapter implements ExecutionAdapter {
  private readonly pythonBin: string;
  private readonly simulatorScript: string;
  private readonly urdfPath: string;
  private readonly traceDir: string;
  private readonly runner: PyBulletSo101CommandRunner;
  private readonly validateCommandPaths: boolean;

  constructor(options: PyBulletSo101ExecutionAdapterOptions = {}) {
    this.pythonBin = options.pythonBin ?? defaultPythonBin();
    this.simulatorScript =
      options.simulatorScript ?? process.env.ROBOCLAW_SO101_SIM_SCRIPT ?? defaultSimulatorScript();
    this.urdfPath = options.urdfPath ?? process.env.ROBOCLAW_SO101_URDF_PATH ?? defaultUrdfPath();
    this.traceDir = options.traceDir ?? process.env.ROBOCLAW_SO101_TRACE_DIR ?? DEFAULT_TRACE_DIR;
    this.runner = options.runner ?? runPyBulletSo101Command;
    this.validateCommandPaths = options.validateCommandPaths ?? options.runner === undefined;
  }

  async execute(request: ExecutionRequest): Promise<ExecutionReceipt> {
    if (request.skill.semanticAction !== "arm.wave") {
      return {
        executionId: `pybullet-so101-unsupported-${request.goal.goalId}`,
        status: "failed",
        backendKind: request.body.backendKind,
        detail: `PyBulletSo101ExecutionAdapter does not support semantic action ${request.skill.semanticAction}.`,
      };
    }

    if (this.validateCommandPaths) {
      await access(this.pythonBin);
      await access(this.simulatorScript);
      await access(this.urdfPath);
    }

    await mkdir(this.traceDir, { recursive: true });
    const tracePath = path.resolve(this.traceDir, `${request.goal.goalId}.json`);

    const command = [
      this.pythonBin,
      this.simulatorScript,
      "execute",
      JSON.stringify({
        semanticAction: request.skill.semanticAction,
        parameters: request.skill.parameters,
        goalId: request.goal.goalId,
        bodyId: request.body.bodyId,
        urdfPath: this.urdfPath,
        tracePath,
      }),
    ] as const;

    const result = await this.runner(command[0], command.slice(1));
    const simulation = parsePyBulletSo101SimulatorOutput(result.stdout);

    return {
      executionId: `pybullet-so101-${request.goal.goalId}`,
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
