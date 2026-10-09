"""The skills catalog — WZRD's progressive-disclosure layer.

A skill is a folder under `wzrd_mcp/skills/<name>/` in the Agent Skills
format: a `SKILL.md` with YAML frontmatter (`name`, `description`, optional
`metadata.kind: workflow`) and, optionally, deeper `references/*.md`.

Three levels of disclosure, so the always-on context stays small:

1. **Index** — one line per skill (name + description), rendered into the
   server `instructions` and the `load_skill` tool description.
2. **Body** — `SKILL.md`, fetched on demand via `load_skill(name)` (any
   client) or the `skill://<name>/SKILL.md` resource (clients that read MCP
   resources).
3. **References** — `load_skill(name, file="references/x.md")`, only when the
   body points there.

Pure module (no FastMCP server import) so `server.py` can build its
instructions from it before the server exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from fastmcp.server.providers.skills._common import parse_frontmatter

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    kind: str  # "workflow" (an end-to-end job, offered as an MCP prompt) | "guide"
    path: Path

    def files(self) -> list[str]:
        """Supporting files, relative to the skill folder (SKILL.md excluded)."""
        return sorted(
            str(p.relative_to(self.path))
            for p in self.path.rglob("*")
            if p.is_file() and p.name != "SKILL.md"
        )


@cache
def catalog() -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for folder in sorted(SKILLS_DIR.iterdir()):
        main = folder / "SKILL.md"
        if not main.is_file():
            continue
        meta, _ = parse_frontmatter(main.read_text(encoding="utf-8"))
        name = meta.get("name") or folder.name
        if name != folder.name:
            raise ValueError(f"skill {folder.name!r}: frontmatter name {name!r} must match its folder")
        kind = (meta.get("metadata") or {}).get("kind", "guide")
        skills[name] = Skill(name, " ".join(meta.get("description", "").split()), kind, folder)
    return skills


def index_lines() -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in catalog().values())


def read(name: str, file: str | None = None) -> str:
    """SKILL.md body (frontmatter stripped), or one supporting file."""
    skill = catalog().get(name)
    if skill is None:
        raise ValueError(f"unknown skill {name!r}; available: {', '.join(catalog())}")
    if not file or file == "SKILL.md":
        _, body = parse_frontmatter((skill.path / "SKILL.md").read_text(encoding="utf-8"))
        files = skill.files()
        tail = f"\n\n---\nFiles in this skill (load with file=...): {', '.join(files)}" if files else ""
        return body.strip() + tail
    target = (skill.path / file).resolve()
    if skill.path not in target.parents or not target.is_file():
        raise ValueError(f"skill {name!r} has no file {file!r}; files: {', '.join(skill.files()) or '(none)'}")
    return target.read_text(encoding="utf-8")
