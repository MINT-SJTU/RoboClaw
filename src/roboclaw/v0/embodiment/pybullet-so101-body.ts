import type { BodyAdapter, BodyDescriptor } from "./body-adapter.ts";

const PYBULLET_SO101_BODY: BodyDescriptor = {
  bodyId: "pybullet-so101",
  displayName: "PyBullet SO101",
  embodimentKind: "manipulator_arm",
  backendKind: "simulator",
  capabilities: ["arm.wave"],
};

export class PyBulletSo101BodyAdapter implements BodyAdapter {
  async describeBody(): Promise<BodyDescriptor> {
    return PYBULLET_SO101_BODY;
  }
}

export function getPyBulletSo101BodyDescriptor(): BodyDescriptor {
  return PYBULLET_SO101_BODY;
}
