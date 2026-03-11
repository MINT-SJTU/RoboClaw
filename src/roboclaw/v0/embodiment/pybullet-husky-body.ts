import type { BodyAdapter, BodyDescriptor } from "./body-adapter.ts";

export const PYBULLET_HUSKY_BODY: BodyDescriptor = {
  bodyId: "pybullet-husky",
  displayName: "PyBullet Husky",
  embodimentKind: "mobile_base",
  backendKind: "simulator",
  capabilities: ["base.move.forward", "base.stop"],
};

export class PyBulletHuskyBodyAdapter implements BodyAdapter {
  async describeBody(): Promise<BodyDescriptor> {
    return PYBULLET_HUSKY_BODY;
  }
}
