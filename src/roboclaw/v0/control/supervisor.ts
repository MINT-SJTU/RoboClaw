import type { BodyDescriptor } from "../embodiment/body-adapter.ts";
import type { SkillContract } from "../execution/execution-adapter.ts";
import type { CapabilityId } from "../types.ts";

export type SupervisorAuditRecord = {
  approved: boolean;
  checkedConstraints: readonly string[];
  missingCapabilities: CapabilityId[];
  backendApproved: boolean;
  reasons: string[];
};

export type SupervisorReviewInput = {
  body: BodyDescriptor;
  skill: SkillContract;
};

export class ExecutionSupervisor {
  review(input: SupervisorReviewInput): SupervisorAuditRecord {
    const missingCapabilities = input.skill.requiredCapabilities.filter(
      (capability) => !input.body.capabilities.includes(capability),
    );
    const backendApproved = input.skill.supportedBackends.includes(input.body.backendKind);
    const reasons: string[] = [];

    if (missingCapabilities.length > 0) {
      reasons.push(`Missing required capabilities: ${missingCapabilities.join(", ")}`);
    }

    if (!backendApproved) {
      reasons.push(
        `Backend ${input.body.backendKind} is not supported by skill ${input.skill.skillId}`,
      );
    }

    return {
      approved: reasons.length === 0,
      checkedConstraints: ["required_capabilities", "supported_backends"],
      missingCapabilities,
      backendApproved,
      reasons,
    };
  }
}
