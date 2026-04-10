"""Lightweight tool base for isolated embodied slices.

This mirrors the JSON-schema helper behavior from the main agent tool base
without importing the full roboclaw.agent package tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StandaloneTool(ABC):
    """Minimal tool interface compatible with ToolRegistry expectations."""

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calls."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str | list:
        """Execute the tool with given parameters."""

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obj, dict):
            return obj

        props = schema.get("properties", {})
        result = {}
        for key, value in obj.items():
            if key in props:
                result[key] = self._cast_value(value, props[key])
            else:
                result[key] = value
        return result

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        target_type = schema.get("type")

        if target_type == "boolean" and isinstance(val, bool):
            return val
        if target_type == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if target_type in self._TYPE_MAP and target_type not in ("boolean", "integer", "array", "object"):
            expected = self._TYPE_MAP[target_type]
            if isinstance(val, expected):
                return val

        if target_type == "integer" and isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return val

        if target_type == "number" and isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return val

        if target_type == "string":
            return val if val is None else str(val)

        if target_type == "boolean" and isinstance(val, str):
            lowered = val.lower()
            if lowered in ("true", "1", "yes"):
                return True
            if lowered in ("false", "0", "no"):
                return False
            return val

        if target_type == "array" and isinstance(val, list):
            item_schema = schema.get("items")
            return [self._cast_value(item, item_schema) for item in val] if item_schema else val

        if target_type == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)

        return val

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return self._validate(params, {**schema, "type": "object"}, "")

    def _validate(self, val: Any, schema: dict[str, Any], path: str) -> list[str]:
        target_type, label = schema.get("type"), path or "parameter"
        if target_type == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            return [f"{label} should be integer"]
        if target_type == "number" and (
            not isinstance(val, self._TYPE_MAP[target_type]) or isinstance(val, bool)
        ):
            return [f"{label} should be number"]
        if (
            target_type in self._TYPE_MAP
            and target_type not in ("integer", "number")
            and not isinstance(val, self._TYPE_MAP[target_type])
        ):
            return [f"{label} should be {target_type}"]

        errors: list[str] = []
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if target_type in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if target_type == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if target_type == "object":
            props = schema.get("properties", {})
            for key in schema.get("required", []):
                if key not in val:
                    errors.append(f"missing required {path + '.' + key if path else key}")
            for key, value in val.items():
                child_path = path + "." + key if path else key
                if key in props:
                    errors.extend(self._validate(value, props[key], child_path))
                    continue
                if schema.get("additionalProperties") is False:
                    errors.append(f"{child_path} is not allowed")
                    continue
                if isinstance(schema.get("additionalProperties"), dict):
                    errors.extend(self._validate(value, schema["additionalProperties"], child_path))
        if target_type == "array" and "items" in schema:
            for index, item in enumerate(val):
                errors.extend(
                    self._validate(item, schema["items"], f"{path}[{index}]" if path else f"[{index}]")
                )
        return errors

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
