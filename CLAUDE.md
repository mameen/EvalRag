# EvalRAG

Read [docs/guide_llm_agents.md](docs/guide_llm_agents.md) for full project context, structure, and conventions.

## Key Rules

- **No mocking** — test real implementations with test data
- **Versioning** — `major.minor.build(YYYYMMDDHHmmSS)` — only the maintainer bumps minor+
- **Run with** `PYTHONPATH=src` (editable install via hatchling is broken)
- **Tests**: `PYTHONPATH=src python -m pytest tests/ -v`
- **Demo**: `PYTHONPATH=src python examples/hello_world.py`
- **Template**: copy `examples/experiment_template.py` for new experiments
- **Reports**: timestamped HTML+JSON in `examples/reports/`, self-contained with inline D3.js
