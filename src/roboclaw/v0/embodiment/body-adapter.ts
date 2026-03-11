import type { BackendKind, CapabilityId } from "../types.ts";

export type BodyDescriptor = {
  bodyId: string;
  displayName: string;
  embodimentKind: string;
  backendKind: BackendKind;
  capabilities: readonly CapabilityId[];
};

export interface BodyAdapter {
  describeBody(): Promise<BodyDescriptor>;
}
