import type { BodyDescriptor } from "../embodiment/body-adapter.ts";
import type { BackendKind, CapabilityId, GoalRequest } from "../types.ts";

export type SkillContract = {
  skillId: string;
  summary: string;
  semanticAction: string;
  requiredCapabilities: readonly CapabilityId[];
  supportedBackends: readonly BackendKind[];
  parameters: Record<string, unknown>;
  preconditions: readonly string[];
  successCriterion: string;
  failureCriterion: string;
};

export type ExecutionRequest = {
  goal: GoalRequest;
  body: BodyDescriptor;
  skill: SkillContract;
};

export type ExecutionReceipt = {
  executionId: string;
  status: "completed" | "failed";
  backendKind: BackendKind;
  detail: string;
  command?: readonly string[];
  observation?: Record<string, unknown>;
};

export interface ExecutionAdapter {
  execute(request: ExecutionRequest): Promise<ExecutionReceipt>;
}
