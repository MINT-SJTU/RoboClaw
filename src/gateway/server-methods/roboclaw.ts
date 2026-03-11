import { runSo101Goal, type RunSo101GoalResult } from "../../roboclaw/v0/so101-runtime.js";
import { ErrorCodes, errorShape } from "../protocol/index.js";
import type { GatewayRequestHandlers } from "./types.js";

export type RoboClawGatewayRunner = (input: {
  instruction: string;
  goalId?: string;
}) => Promise<RunSo101GoalResult>;

export function createRoboClawHandlers(deps: {
  runSo101Goal: RoboClawGatewayRunner;
}): GatewayRequestHandlers {
  return {
    "roboclaw.so101.run": async ({ params, respond }) => {
      const instruction = typeof params.instruction === "string" ? params.instruction.trim() : "";
      const goalId = typeof params.goalId === "string" ? params.goalId.trim() : undefined;

      if (!instruction) {
        respond(
          false,
          undefined,
          errorShape(
            ErrorCodes.INVALID_REQUEST,
            "invalid roboclaw.so101.run params: instruction (string) required",
          ),
        );
        return;
      }

      try {
        const result = await deps.runSo101Goal({ instruction, goalId });
        respond(
          true,
          {
            goalId: result.plan.goal.goalId,
            body: result.body,
            rationale: result.plan.rationale,
            skill: result.plan.skill,
            audit: result.audit,
            receipt: result.receipt,
          },
          undefined,
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, message));
      }
    },
  };
}

export const roboclawHandlers = createRoboClawHandlers({
  runSo101Goal,
});
