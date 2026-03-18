"""Standard ROS2 adapter implementation for stage-1 embodied execution."""

from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import replace
from typing import Any

from roboclaw.embodied.definition.systems.deployments.model import DeploymentProfile
from roboclaw.embodied.execution.integration.adapters.model import (
    AdapterBinding,
    AdapterHealthMode,
    AdapterOperation,
    AdapterOperationResult,
    AdapterStateSnapshot,
    CompatibilityCheckItem,
    CompatibilityCheckResult,
    CompatibilityComponent,
    DebugSnapshotResult,
    DependencyCheckItem,
    DependencyCheckResult,
    DependencyKind,
    EnvironmentProbeResult,
    HealthReport,
    PrimitiveExecutionResult,
    ReadinessReport,
    SensorCaptureResult,
)
from roboclaw.embodied.execution.integration.adapters.ros2.profiles import (
    Ros2EmbodimentProfile,
    get_ros2_profile,
)
from roboclaw.embodied.execution.integration.carriers.model import ExecutionTarget
from roboclaw.embodied.execution.integration.transports.ros2.contracts import (
    Ros2ActionSpec,
    Ros2InterfaceBundle,
    Ros2ServiceSpec,
    Ros2TopicSpec,
)


class Ros2ActionServiceAdapter:
    """Framework ROS2 adapter that shells out to the ROS2 CLI on declared interfaces only."""

    def __init__(
        self,
        *,
        binding: AdapterBinding,
        deployment: DeploymentProfile,
        assembly: Any,
        tools: Any,
        profile: Ros2EmbodimentProfile | None = None,
    ) -> None:
        self.binding = binding
        self.deployment = deployment
        self.assembly = assembly
        self.tools = tools
        self.adapter_id = binding.id
        self.assembly_id = binding.assembly_id
        self._active_target_id: str | None = deployment.target_id
        self._launched = False
        self._profile = profile or self._resolve_profile()
        self._service_type_cache: dict[str, str] = {}
        self._action_type_cache: dict[str, str] = {}

    def probe_env(self) -> EnvironmentProbeResult:
        bundle = self._bundle_for_target(self._active_target_id or self.deployment.target_id)
        dependencies = self._declared_dependency_ids(bundle)
        return EnvironmentProbeResult(
            adapter_id=self.adapter_id,
            assembly_id=self.assembly_id,
            transport=self.binding.transport,
            available_targets=self.binding.supported_targets,
            detected_dependencies=dependencies,
            notes=(f"profile={self._profile.id}" if self._profile is not None else "profile=unresolved",),
            details={
                "deployment_id": self.deployment.id,
                "target_id": self._active_target_id or self.deployment.target_id,
                "namespace": bundle.namespace if bundle is not None else None,
            },
        )

    def check_dependencies(self) -> DependencyCheckResult:
        target_id = self._active_target_id or self.deployment.target_id
        bundle = self._bundle_for_target(target_id)
        if bundle is None:
            item = DependencyCheckItem(
                dependency_id="target:ros2_bundle",
                kind=self.binding.lifecycle.dependencies[0].kind if self.binding.lifecycle.dependencies else DependencyKind.OTHER,
                required=True,
                available=False,
                message=f"Target '{target_id}' does not expose a ROS2 interface bundle.",
            )
            return DependencyCheckResult(
                adapter_id=self.adapter_id,
                ok=False,
                items=(item,),
                checked_dependencies=(item.dependency_id,),
                missing_required=(item.dependency_id,),
            )

        items: list[DependencyCheckItem] = []
        missing_required: list[str] = []
        for dep_id, kind, available, required, message in self._declared_dependency_checks(bundle):
            if required and not available:
                missing_required.append(dep_id)
            items.append(
                DependencyCheckItem(
                    dependency_id=dep_id,
                    kind=kind,
                    required=required,
                    available=available,
                    message=message,
                )
            )
        return DependencyCheckResult(
            adapter_id=self.adapter_id,
            ok=not missing_required,
            items=tuple(items),
            checked_dependencies=tuple(item.dependency_id for item in items),
            missing_required=tuple(missing_required),
            notes=(f"profile={self._profile.id}" if self._profile is not None else "profile=unresolved",),
        )

    async def connect(
        self,
        *,
        target_id: str,
        config: dict[str, Any] | None = None,
    ) -> AdapterOperationResult:
        if target_id not in self.binding.supported_targets:
            return AdapterOperationResult(
                operation=AdapterOperation.CONNECT,
                ok=False,
                target_id=target_id,
                error_code="TARGET_UNSUPPORTED",
                message=f"Adapter '{self.adapter_id}' does not support target '{target_id}'.",
            )

        self._active_target_id = target_id
        await self._maybe_launch_runtime(config=config)
        service = self._service_spec("connect", required=True)
        result = await self._call_service(
            service,
            payload={"deployment_id": self.deployment.id},
            operation=AdapterOperation.CONNECT,
        )
        if not result.ok and self._launched:
            for _ in range(2):
                await asyncio.sleep(1.0)
                result = await self._call_service(
                    service,
                    payload={"deployment_id": self.deployment.id},
                    operation=AdapterOperation.CONNECT,
                )
                if result.ok:
                    break
        if result.ok:
            readiness = await self.ready()
            if not readiness.ready:
                return AdapterOperationResult(
                    operation=AdapterOperation.CONNECT,
                    ok=False,
                    target_id=target_id,
                    error_code="DEPENDENCY_UNAVAILABLE",
                    message=readiness.message or "ROS2 control surface is not ready.",
                    details=readiness.details,
                )
        return result

    async def disconnect(self) -> AdapterOperationResult:
        result = await self._call_service(
            self._service_spec("disconnect", required=False),
            payload={"deployment_id": self.deployment.id},
            operation=AdapterOperation.DISCONNECT,
        )
        if result.ok:
            self._active_target_id = None
        return result

    async def ready(self) -> ReadinessReport:
        dep_result = self.check_dependencies()
        if not dep_result.ok:
            return ReadinessReport(
                ready=False,
                target_id=self._active_target_id or self.deployment.target_id,
                blocked_operations=(
                    AdapterOperation.CONNECT,
                    AdapterOperation.RESET,
                    AdapterOperation.STOP,
                ),
                message="Required ROS2 interfaces are missing for this adapter.",
                details={"missing_required": dep_result.missing_required},
            )
        if self._active_target_id is None:
            return ReadinessReport(
                ready=False,
                target_id=self.deployment.target_id,
                blocked_operations=(AdapterOperation.CONNECT,),
                message="Adapter is not connected to an active target yet.",
            )
        runtime_missing = await self._runtime_missing_required()
        if runtime_missing:
            return ReadinessReport(
                ready=False,
                target_id=self._active_target_id,
                blocked_operations=(
                    AdapterOperation.CONNECT,
                    AdapterOperation.STOP,
                    AdapterOperation.RESET,
                    AdapterOperation.RECOVER,
                ),
                message="Declared ROS2 interfaces are not currently available in the ROS graph.",
                details={"missing_required": runtime_missing},
            )
        return ReadinessReport(
            ready=True,
            target_id=self._active_target_id,
            details={"deployment_id": self.deployment.id},
        )

    async def health(self) -> HealthReport:
        readiness = await self.ready()
        if readiness.ready:
            return HealthReport(
                mode=AdapterHealthMode.READY,
                healthy=True,
                details=readiness.details,
            )
        return HealthReport(
            mode=AdapterHealthMode.UNAVAILABLE,
            healthy=False,
            error_codes=("DEPENDENCY_UNAVAILABLE",),
            blocked_operations=readiness.blocked_operations,
            message=readiness.message,
            details=readiness.details,
        )

    async def check_compatibility(self) -> CompatibilityCheckResult:
        checks: list[CompatibilityCheckItem] = []
        blocking_failures: list[str] = []
        for constraint in self.binding.compatibility.constraints:
            satisfied = True
            message: str | None = None
            if constraint.component == CompatibilityComponent.TRANSPORT:
                satisfied = constraint.target == "ros2"
                if not satisfied:
                    message = f"Expected ROS2 transport, got '{constraint.target}'."
            elif constraint.component == CompatibilityComponent.BRIDGE:
                if self.binding.bridge_id is None:
                    satisfied = not constraint.required
                    if not satisfied:
                        message = "Adapter binding does not declare a bridge id."
                else:
                    satisfied = constraint.target == self.binding.bridge_id
                    if not satisfied:
                        message = (
                            f"Constraint expects bridge '{constraint.target}', "
                            f"binding uses '{self.binding.bridge_id}'."
                        )
            if constraint.required and not satisfied:
                blocking_failures.append(f"{constraint.component.value}:{constraint.target}")
            checks.append(
                CompatibilityCheckItem(
                    component=constraint.component,
                    target=constraint.target,
                    requirement=constraint.requirement,
                    satisfied=satisfied,
                    required=constraint.required,
                    detected_version="stage1",
                    message=message,
                )
            )
        return CompatibilityCheckResult(
            adapter_api_version=self.binding.compatibility.adapter_api_version,
            compatible=not blocking_failures,
            checks=tuple(checks),
            blocking_failures=tuple(blocking_failures),
        )

    async def stop(self, *, scope: str = "all") -> AdapterOperationResult:
        return await self._call_service(
            self._service_spec("stop", required=True),
            payload={"scope": scope},
            operation=AdapterOperation.STOP,
        )

    async def reset(self, *, mode: str = "home") -> AdapterOperationResult:
        resolved_mode = mode or (self._profile.default_reset_mode if self._profile is not None else "home")
        return await self._call_service(
            self._service_spec("reset", required=True),
            payload={"mode": resolved_mode},
            operation=AdapterOperation.RESET,
        )

    async def recover(self, *, strategy: str | None = None) -> AdapterOperationResult:
        return await self._call_service(
            self._service_spec("recover", required=True),
            payload={"strategy": strategy or "default"},
            operation=AdapterOperation.RECOVER,
        )

    async def get_state(self) -> AdapterStateSnapshot:
        topic = self._topic_spec("state", required=False)
        if topic is None:
            return AdapterStateSnapshot(
                source="ros2",
                target_id=self._active_target_id or self.deployment.target_id,
                notes=("State topic is not declared for this target.",),
            )
        output = await self._run_ros2_command(f"ros2 topic echo --once --field data {shlex.quote(topic.path)}")
        parsed = self._parse_state_topic_output(output)
        if not parsed:
            legacy_output = await self._run_ros2_command(f"ros2 topic echo --once {shlex.quote(topic.path)}")
            legacy_parsed = self._parse_state_topic_output(legacy_output)
            if legacy_parsed:
                output = legacy_output
                parsed = legacy_parsed
        values = {"raw": output}
        updated_fields: tuple[str, ...] = ("raw",)
        if parsed:
            values.update(parsed)
            updated_fields = tuple(values.keys())
        return AdapterStateSnapshot(
            source="ros2",
            target_id=self._active_target_id or self.deployment.target_id,
            values=values,
            updated_fields=updated_fields,
            notes=(f"topic={topic.path}",),
        )

    async def execute_primitive(
        self,
        name: str,
        args: dict[str, Any] | None = None,
    ) -> PrimitiveExecutionResult:
        resolved_args = args or {}
        payload = {
            "primitive_name": name,
            "arguments_json": json.dumps(resolved_args, ensure_ascii=False, sort_keys=True),
        }
        action = self._action_spec("execute_primitive", required=True)
        available_actions = await self._list_runtime_interfaces("action")
        if action is not None and action.path in available_actions:
            action_type = await self._runtime_action_type(action.path) or action.action_type
            output = await self._run_ros2_command(
                f"ros2 action send_goal {shlex.quote(action.path)} "
                f"{shlex.quote(action_type)} {shlex.quote(json.dumps(payload, ensure_ascii=False))}"
            )
            if self._looks_like_failure(output):
                return PrimitiveExecutionResult(
                    primitive_name=name,
                    accepted=False,
                    completed=False,
                    status="failed",
                    error_code="COMMAND_FAILED",
                    message=output.strip() or "ROS2 action call failed.",
                    output={"raw": output},
                )
            lowered = output.lower()
            accepted = "accepted" in lowered or "goal accepted" in lowered or not output.startswith("Error")
            completed = "succeeded" in lowered or "result:" in lowered
            status = "succeeded" if completed else "accepted"
            return PrimitiveExecutionResult(
                primitive_name=name,
                accepted=accepted,
                completed=completed,
                status=status,
                message=output.strip() or None,
                output={"raw": output},
            )

        primitive_service = self._primitive_service_spec(name, resolved_args)
        if primitive_service is not None:
            ok, output, error_code = await self._invoke_service(primitive_service, payload=resolved_args)
            if not ok:
                return PrimitiveExecutionResult(
                    primitive_name=name,
                    accepted=False,
                    completed=False,
                    status="failed",
                    error_code=error_code or "SERVICE_UNAVAILABLE",
                    message=output,
                    output={"raw": output},
                )
            return PrimitiveExecutionResult(
                primitive_name=name,
                accepted=True,
                completed=True,
                status="succeeded",
                message=output,
                output={"raw": output},
            )

        if action is None:
            message = "Declared execute_primitive action is unavailable."
        else:
            message = f"Declared execute_primitive action `{action.path}` is unavailable in the ROS graph."
        return PrimitiveExecutionResult(
            primitive_name=name,
            accepted=False,
            completed=False,
            status="failed",
            error_code="ACTION_UNAVAILABLE",
            message=message,
        )

    async def capture_sensor(self, sensor_id: str, mode: str = "latest") -> SensorCaptureResult:
        service = self._service_spec("sensor_snapshot", required=False)
        if service is None:
            return SensorCaptureResult(
                sensor_id=sensor_id,
                mode=mode,
                captured=False,
                message="Declared sensor snapshot service is unavailable.",
            )
        output = await self._run_ros2_command(
            f"ros2 service call {shlex.quote(service.path)} {shlex.quote(service.service_type)} "
            f"{shlex.quote(json.dumps({'sensor_id': sensor_id, 'mode': mode}, ensure_ascii=False))}"
        )
        return SensorCaptureResult(
            sensor_id=sensor_id,
            mode=mode,
            captured=not self._looks_like_failure(output),
            payload_ref=None,
            metadata={"raw": output},
            message=output.strip() or None,
        )

    async def debug_snapshot(self) -> DebugSnapshotResult:
        service = self._service_spec("debug_snapshot", required=True)
        if service is None:
            return DebugSnapshotResult(
                captured=False,
                summary="Declared debug snapshot service is unavailable.",
                message="Declared debug snapshot service is unavailable.",
            )
        output = await self._run_ros2_command(
            f"ros2 service call {shlex.quote(service.path)} {shlex.quote(service.service_type)} "
            f"{shlex.quote('{}')}"
        )
        if self._looks_like_failure(output):
            return DebugSnapshotResult(
                captured=False,
                summary="ROS2 debug snapshot failed.",
                payload={"raw": output},
                message=output.strip() or "ROS2 debug snapshot failed.",
            )
        return DebugSnapshotResult(
            captured=True,
            summary="Collected ROS2 debug snapshot.",
            payload={"raw": output},
            message=output.strip() or None,
        )

    def _resolve_profile(self) -> Ros2EmbodimentProfile | None:
        profile_hint = str(self.deployment.connection.get("profile_id", "")).strip().lower()
        if profile_hint:
            profile = get_ros2_profile(profile_hint)
            if profile is not None:
                return profile
        robots = getattr(self.assembly, "robots", ())
        if robots:
            primary_robot_id = getattr(robots[0], "robot_id", None)
            return get_ros2_profile(primary_robot_id)
        return None

    def _bundle_for_target(self, target_id: str) -> Ros2InterfaceBundle | None:
        target = self._target(target_id)
        if target is None or target.ros2 is None:
            return None
        namespace_override = str(self.deployment.connection.get("namespace", "")).strip()
        if not namespace_override or namespace_override == target.ros2.namespace:
            bundle = target.ros2
        else:
            bundle = Ros2InterfaceBundle(
                namespace=namespace_override,
                topics=tuple(
                    self._replace_namespace(topic, target.ros2.namespace, namespace_override)
                    for topic in target.ros2.topics
                ),
                services=tuple(
                    self._replace_namespace(service, target.ros2.namespace, namespace_override)
                    for service in target.ros2.services
                ),
                actions=tuple(
                    self._replace_namespace(action, target.ros2.namespace, namespace_override)
                    for action in target.ros2.actions
                ),
                frames=target.ros2.frames,
            )
        return self._augment_bundle_with_profile_services(bundle)

    def _target(self, target_id: str) -> ExecutionTarget | None:
        for target in getattr(self.assembly, "execution_targets", ()):
            if target.id == target_id:
                return target
        return None

    @staticmethod
    def _replace_namespace(item: Any, current: str, new: str) -> Any:
        return replace(item, path=item.path.replace(current, new, 1))

    def _declared_dependency_ids(self, bundle: Ros2InterfaceBundle | None) -> tuple[str, ...]:
        if bundle is None:
            return tuple()
        ids = [f"service:{name}" for name in self._profile.required_services] if self._profile is not None else []
        if self._profile is not None:
            ids.extend(f"action:{name}" for name in self._profile.required_actions)
            ids.extend(f"topic:{name}" for name in self._profile.optional_topics)
        return tuple(ids)

    def _augment_bundle_with_profile_services(self, bundle: Ros2InterfaceBundle) -> Ros2InterfaceBundle:
        profile = self._profile
        if profile is None or not profile.primitive_services:
            return bundle
        existing = {service.name for service in bundle.services}
        extra = tuple(item for item in profile.extra_service_specs(bundle.namespace) if item.name not in existing)
        if not extra:
            return bundle
        return Ros2InterfaceBundle(
            namespace=bundle.namespace,
            topics=bundle.topics,
            services=tuple((*bundle.services, *extra)),
            actions=bundle.actions,
            frames=bundle.frames,
        )

    def _primitive_service_spec(self, primitive_name: str, args: dict[str, Any]) -> Ros2ServiceSpec | None:
        bundle = self._bundle_for_target(self._active_target_id or self.deployment.target_id)
        profile = self._profile
        if bundle is None or profile is None:
            return None
        service = profile.primitive_service_for(primitive_name, args)
        if service is None:
            return None
        return bundle.service(service.service_name)

    def _declared_dependency_checks(
        self,
        bundle: Ros2InterfaceBundle,
    ) -> tuple[tuple[str, DependencyKind, bool, bool, str | None], ...]:
        checks: list[tuple[str, DependencyKind, bool, bool, str | None]] = []
        profile = self._profile
        if profile is None:
            return tuple()
        for name in profile.required_services:
            service = bundle.service(name)
            dep_id = f"service:{name}"
            available = service is not None
            message = None if available else f"Declared service '{name}' is missing from target bundle."
            checks.append((dep_id, DependencyKind.ROS2_SERVICE, available, True, message))
        for name in profile.required_actions:
            action = bundle.action(name)
            dep_id = f"action:{name}"
            available = action is not None
            message = None if available else f"Declared action '{name}' is missing from target bundle."
            checks.append((dep_id, DependencyKind.ROS2_ACTION, available, True, message))
        for name in profile.optional_topics:
            topic = bundle.topic(name)
            dep_id = f"topic:{name}"
            available = topic is not None
            message = None if available else f"Declared topic '{name}' is missing from target bundle."
            checks.append((dep_id, DependencyKind.ROS2_TOPIC, available, False, message))
        return tuple(checks)

    async def _runtime_missing_required(self) -> tuple[str, ...]:
        bundle = self._bundle_for_target(self._active_target_id or self.deployment.target_id)
        profile = self._profile
        if bundle is None or profile is None:
            return ("target:ros2_bundle",)

        available_services = await self._list_runtime_interfaces("service")
        available_actions = await self._list_runtime_interfaces("action")
        missing: list[str] = []
        for name in profile.required_services:
            service = bundle.service(name)
            if service is None or service.path not in available_services:
                missing.append(f"service:{name}")
        for name in profile.required_actions:
            action = bundle.action(name)
            if action is None or action.path not in available_actions:
                missing.append(f"action:{name}")
        return tuple(missing)

    async def _list_runtime_interfaces(self, kind: str) -> set[str]:
        if kind not in {"service", "action", "topic"}:
            return set()
        output = await self._run_ros2_command(f"ros2 {kind} list")
        if self._looks_like_failure(output):
            return set()
        return {
            line.strip()
            for line in output.splitlines()
            if line.strip() and not line.lower().startswith("stderr:")
        }

    def _service_spec(self, name: str, *, required: bool) -> Ros2ServiceSpec | None:
        bundle = self._bundle_for_target(self._active_target_id or self.deployment.target_id)
        if bundle is None:
            return None
        service = bundle.service(name)
        if service is None and required:
            return None
        return service

    def _action_spec(self, name: str, *, required: bool) -> Ros2ActionSpec | None:
        bundle = self._bundle_for_target(self._active_target_id or self.deployment.target_id)
        if bundle is None:
            return None
        action = bundle.action(name)
        if action is None and required:
            return None
        return action

    def _topic_spec(self, name: str, *, required: bool) -> Ros2TopicSpec | None:
        bundle = self._bundle_for_target(self._active_target_id or self.deployment.target_id)
        if bundle is None:
            return None
        topic = bundle.topic(name)
        if topic is None and required:
            return None
        return topic

    async def _call_service(
        self,
        service: Ros2ServiceSpec | None,
        *,
        payload: dict[str, Any],
        operation: AdapterOperation,
    ) -> AdapterOperationResult:
        if service is None:
            return AdapterOperationResult(
                operation=operation,
                ok=False,
                target_id=self._active_target_id or self.deployment.target_id,
                error_code="SERVICE_UNAVAILABLE",
                message=f"Declared ROS2 service for '{operation.value}' is unavailable.",
            )
        ok, output, error_code = await self._invoke_service(service, payload=payload)
        if not ok:
            return AdapterOperationResult(
                operation=operation,
                ok=False,
                target_id=self._active_target_id or self.deployment.target_id,
                error_code=error_code or "COMMAND_FAILED",
                message=output.strip() or f"ROS2 service call '{operation.value}' failed.",
                details={"raw": output},
            )
        return AdapterOperationResult(
            operation=operation,
            ok=True,
            target_id=self._active_target_id or self.deployment.target_id,
            message=output.strip() or None,
            details={"raw": output},
        )

    async def _invoke_service(
        self,
        service: Ros2ServiceSpec,
        *,
        payload: dict[str, Any],
    ) -> tuple[bool, str, str | None]:
        available_services = await self._list_runtime_interfaces("service")
        if service.path not in available_services:
            return False, f"Declared ROS2 service `{service.path}` is unavailable in the ROS graph.", "SERVICE_UNAVAILABLE"
        service_type = await self._runtime_service_type(service.path) or service.service_type
        serialized = self._serialize_service_payload(service_type, payload)
        output = await self._run_ros2_command(
            f"ros2 service call {shlex.quote(service.path)} {shlex.quote(service_type)} "
            f"{shlex.quote(serialized)}"
        )
        if self._looks_like_failure(output) or self._service_output_indicates_failure(service_type, output):
            return False, output, "COMMAND_FAILED"
        return True, output.strip() or output, None

    async def _maybe_launch_runtime(self, *, config: dict[str, Any] | None = None) -> None:
        launch_command = str((config or {}).get("launch_command") or self.deployment.connection.get("launch_command") or "").strip()
        if not launch_command or self._launched:
            return
        connect_service = self._service_spec("connect", required=False)
        if connect_service is not None:
            available_services = await self._list_runtime_interfaces("service")
            if connect_service.path in available_services:
                self._launched = True
                return
        if connect_service is None:
            wait_clause = "sleep 4"
        else:
            wait_clause = (
                f"for _ in 1 2 3 4 5 6 7 8; do "
                f"ros2 service list | grep -Fx {shlex.quote(connect_service.path)} >/dev/null 2>&1 && break; "
                "sleep 1; "
                "done"
            )
        await self._run_ros2_command(
            f"nohup bash -lc {shlex.quote(launch_command)} >/tmp/{self.adapter_id}_launch.log 2>&1 & {wait_clause}"
        )
        self._launched = True

    async def _run_ros2_command(self, command: str) -> str:
        if self.tools is None or not hasattr(self.tools, "execute"):
            return "Error: Tool registry with exec support is required."
        prefix_parts: list[str] = []
        distro = str(self.deployment.connection.get("ros_distro", "")).strip()
        if distro:
            prefix_parts.append(
                f"source /opt/ros/{distro}/setup.bash >/dev/null 2>&1 || "
                f"source /opt/ros/{distro}/setup.zsh >/dev/null 2>&1"
            )
        else:
            prefix_parts.append(
                "for setup in /opt/ros/*/setup.bash; do "
                "[ -f \"$setup\" ] || continue; "
                "source \"$setup\" >/dev/null 2>&1 && break; "
                "done; "
            )
        source_command = str(self.deployment.connection.get("source_command", "")).strip()
        if source_command:
            prefix_parts.append(source_command)
        inner = "; ".join((*prefix_parts, command)) if prefix_parts else command
        return await self.tools.execute("exec", {"command": f"bash -lc {shlex.quote(inner)}"})

    async def _runtime_service_type(self, path: str) -> str | None:
        cached = self._service_type_cache.get(path)
        if cached is not None:
            return cached
        output = await self._run_ros2_command(f"ros2 service type {shlex.quote(path)}")
        value = self._extract_runtime_type(output)
        if value is not None:
            self._service_type_cache[path] = value
        return value

    async def _runtime_action_type(self, path: str) -> str | None:
        cached = self._action_type_cache.get(path)
        if cached is not None:
            return cached
        output = await self._run_ros2_command(f"ros2 action type {shlex.quote(path)}")
        value = self._extract_runtime_type(output)
        if value is not None:
            self._action_type_cache[path] = value
        return value

    @staticmethod
    def _extract_runtime_type(output: str) -> str | None:
        if not output or Ros2ActionServiceAdapter._looks_like_failure(output):
            return None
        for line in output.splitlines():
            stripped = line.strip()
            if stripped and not stripped.lower().startswith("stderr:") and "/" in stripped:
                return stripped
        return None

    @staticmethod
    def _serialize_service_payload(service_type: str, payload: dict[str, Any]) -> str:
        if service_type == "std_srvs/srv/Trigger":
            return "{}"
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _service_output_indicates_failure(service_type: str, output: str) -> bool:
        lowered = output.lower()
        if service_type == "std_srvs/srv/Trigger":
            return "success: false" in lowered or "success=false" in lowered
        return False

    @staticmethod
    def _parse_state_topic_output(output: str) -> dict[str, Any]:
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped or stripped == "---" or stripped.lower().startswith("stderr:"):
                continue
            payload = stripped
            if stripped.startswith("data:"):
                payload = stripped.removeprefix("data:").strip()
            if not payload:
                continue
            if payload and payload[0] in {"'", '"'} and payload[-1] == payload[0]:
                payload = payload[1:-1]
            payload = payload.replace("\\'", "'").replace('\\"', '"')
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _looks_like_failure(output: str) -> bool:
        lowered = output.strip().lower()
        return lowered.startswith("error") or "exit code:" in lowered or "stderr:" in lowered


__all__ = ["Ros2ActionServiceAdapter"]
