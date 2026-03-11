import type { PlanRequest, TaskPlan, TaskPlanner } from "./control/task-planner.ts";
import type { BodyAdapter, BodyDescriptor } from "./embodiment/body-adapter.ts";
import type {
  ExecutionAdapter,
  ExecutionReceipt,
  ExecutionRequest,
  SkillContract,
} from "./execution/execution-adapter.ts";

export class StaticBodyAdapter implements BodyAdapter {
  constructor(private readonly body: BodyDescriptor) {}

  async describeBody(): Promise<BodyDescriptor> {
    return this.body;
  }
}

export class StaticTaskPlanner implements TaskPlanner {
  constructor(
    private readonly skill: SkillContract,
    private readonly rationale = "Round 1 stub planner selected a fixed skill contract",
  ) {}

  async plan(input: PlanRequest): Promise<TaskPlan> {
    return {
      goal: input.goal,
      skill: this.skill,
      rationale: this.rationale,
    };
  }
}

export class StubExecutionAdapter implements ExecutionAdapter {
  readonly requests: ExecutionRequest[] = [];

  async execute(request: ExecutionRequest): Promise<ExecutionReceipt> {
    this.requests.push(request);

    return {
      executionId: `stub-${this.requests.length}`,
      status: "completed",
      backendKind: request.body.backendKind,
      detail: `Stub executed ${request.skill.skillId} for ${request.body.bodyId}`,
    };
  }
}
