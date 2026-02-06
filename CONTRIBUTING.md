# Contributing to Aye Chef

Thanks for your interest in contributing!

## Development Setup

### Docker (recommended)

```bash
git clone https://github.com/deepseekcoder2/ayechef.git
cd aye-chef
docker compose up -d
```

The web panel runs at http://localhost:8080 with hot-reload enabled.

### Local (without Docker)

Requires Python 3.11+.

```bash
pip install -r requirements.txt

# Terminal 1: web panel
flask --app panel.app run --port 8080

# Terminal 2: background worker
huey_consumer panel.jobs.huey_config.huey -w 2 -k thread
```

## Running Tests

```bash
# All tests (excludes slow/network-dependent)
pytest

# Specific file
pytest tests/test_unit.py -v

# Include slow tests
pytest -m "slow" -v
```

## Project Structure

- `panel/` — Flask web UI (routes, templates, jobs)
- `panel/tools/registry.py` — Tool definitions (add new tools here)
- CLI scripts in project root — Backend logic (orchestrator, importers, parsers)
- `config.py` — Centralized configuration
- `mealie_client.py` — Mealie API client
- `tests/` — Test suite

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Ensure tests pass: `pytest`
4. Open a pull request with a clear description of what and why

## Adding a New Maintenance Tool

1. Add a `Tool` entry in `panel/tools/registry.py`
2. That's it — the tool auto-appears in the web UI

## Code Style

- Python: Follow existing patterns in the codebase
- Templates: Tailwind CSS utility classes
- Keep it simple — this is a home-network tool, not enterprise software
