"""Serve the skills catalog three ways (see `skill_catalog.py` for the model):

- `load_skill` tool — works on every client, including tools-only ones.
- MCP resources `skill://<name>/SKILL.md` (+ `_manifest` and supporting files)
  via FastMCP's `SkillsDirectoryProvider` — the standard skills-over-MCP shape,
  for clients that read resources (Claude Code's resource tools, @-mentions).
- One MCP prompt per workflow skill — user-facing entry points (Claude Code
  shows them as `/mcp__wzrd__<name>`), each telling the agent to load the
  skill and start.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from pydantic import Field

from . import skill_catalog
from .server import mcp, tool

mcp.add_provider(SkillsDirectoryProvider(skill_catalog.SKILLS_DIR))


async def load_skill(name: str, file: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    """Load a WZRD skill: the workflow, prompt formulas and settings for a kind of job. Call it before starting that job.

    Returns the skill's SKILL.md; it lists any deeper reference files, which
    you load with `file` only when the body points you there.

    Args:
        name: Skill name, from the index in the server instructions.
        file: A supporting file inside the skill, e.g. "references/anti-slop.md".
    """
    try:
        return skill_catalog.read(name, file)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


# The skill index rides in the tool description too: some clients never show
# server instructions to the model, and the tool is where it looks.
load_skill.__doc__ = load_skill.__doc__.replace(
    "    Args:", "    Skills:\n" + "\n".join("    " + line for line in skill_catalog.index_lines().splitlines()) + "\n\n    Args:"
)
load_skill = tool(title="Load skill", tags={"skills"}, read_only=True, idempotent=True)(load_skill)


def _register_prompt(skill: skill_catalog.Skill) -> None:
    def prompt(goal: Annotated[str, Field(description="What the user wants, in their words (optional).")] = "") -> str:
        ask = f" The user's goal: {goal}" if goal else ""
        return (
            f"Load the WZRD skill `{skill.name}` with load_skill(name=\"{skill.name}\") and follow it.{ask} "
            "Start by confirming the active project (set_project) and asking for anything the skill "
            "says must be locked in before generating."
        )

    mcp.prompt(
        name=skill.name.replace("-", "_"),
        title=skill.name.replace("-", " ").capitalize(),
        description=skill.description,
        tags={"skills"},
    )(prompt)


for _skill in skill_catalog.catalog().values():
    if _skill.kind == "workflow":
        _register_prompt(_skill)
