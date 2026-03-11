export type BackendKind = "hardware" | "simulator";

export type CapabilityId = string;

export type GoalRequest = {
  goalId: string;
  instruction: string;
};
