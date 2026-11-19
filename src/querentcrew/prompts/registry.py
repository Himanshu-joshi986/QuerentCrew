"""Versioned prompt registry.

Prompts live in YAML files next to this module (one file per agent). Each file
holds several versions so they can be compared in the evaluation harness:

    name: sql_writer
    default: v3
    versions:
      v1:
        description: ...
        system: |
          ...
        user: |
          ... $question ...

Templates use `string.Template` ($var) syntax, so SQL braces and JSON in
prompts never need escaping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

PROMPT_DIR = Path(__file__).parent


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    description: str
    system: str
    user: str

    def render(self, **variables) -> tuple[str, str]:
        return (
            Template(self.system).safe_substitute(**variables).strip(),
            Template(self.user).safe_substitute(**variables).strip(),
        )


class PromptRegistry:
    def __init__(self, directory: str | Path = PROMPT_DIR):
        self.directory = Path(directory)
        self._prompts: dict[str, dict[str, PromptVersion]] = {}
        self._defaults: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        for file in sorted(self.directory.glob("*.yaml")):
            data = yaml.safe_load(file.read_text(encoding="utf-8"))
            name = data["name"]
            self._defaults[name] = data.get("default") or next(iter(data["versions"]))
            self._prompts[name] = {
                version: PromptVersion(
                    name=name,
                    version=version,
                    description=spec.get("description", ""),
                    system=spec.get("system", ""),
                    user=spec.get("user", ""),
                )
                for version, spec in data["versions"].items()
            }

    def names(self) -> list[str]:
        return sorted(self._prompts)

    def versions(self, name: str) -> list[str]:
        return list(self._prompts[name])

    def get(self, name: str, version: str | None = None) -> PromptVersion:
        if name not in self._prompts:
            raise KeyError(f"Unknown prompt '{name}'. Known: {', '.join(self.names())}")
        version = version or self._defaults[name]
        if version not in self._prompts[name]:
            raise KeyError(f"Prompt '{name}' has no version '{version}'. Known: {self.versions(name)}")
        return self._prompts[name][version]

    def render(self, name: str, version: str | None = None, **variables) -> tuple[str, str]:
        return self.get(name, version).render(**variables)
