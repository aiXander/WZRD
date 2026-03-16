# WZRD Skills

Each subdirectory is a **skill** — a composable pipeline of WZRD tool calls that the agent can execute as a sequence. Skills are loaded at runtime by the orchestrator (Mastra) and injected into the agent's context.

## Structure

```
skills/
  <skill_name>/
    skill.json       # Metadata, tool sequence, and parameter mappings
    prompt.md        # Natural language instructions for the agent
```

## Skill JSON Schema

```json
{
  "name": "skill_name",
  "description": "What this pipeline does",
  "tags": ["projection", "animation", ...],
  "inputs": {
    "param_name": { "type": "string", "description": "...", "required": true }
  },
  "steps": [
    {
      "tool": "tool_name",
      "description": "What this step does",
      "params": { "param": "{input.param_name}" },
      "output_key": "step_result_name"
    }
  ]
}
```

Parameter values can reference:
- `{input.param_name}` — skill-level inputs
- `{steps.step_name.output_field}` — outputs from prior steps
