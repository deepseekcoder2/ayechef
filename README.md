# Aye Chef - AI Meal Planning System

AI-powered weekly meal planning with vector search, automatic shopping lists, and smart recipe imports.

**Stack:** Flask + Huey (web panel) • OpenRouter (LLM) • sentence-transformers (local embeddings) • Mealie (recipe storage)

---

## What This Does

A complete AI meal planning system that:

1. **Plans weekly meals** using an agentic AI chef that considers variety, complexity, and protein balance
2. **Personalizes for your household** - dietary restrictions, cooking constraints, and per-session special instructions
3. **Generates shopping lists** automatically from meal plans with LLM-powered refinement
4. **Imports recipes** from supported websites with full processing (parsing, tagging, indexing)
5. **Generates shareable summaries** — copy-paste-ready meal plans and shopping lists with prep warnings and organized categories
6. **Uses vector search** for semantic recipe matching (finds recipes by meaning, not just keywords)
7. **Tracks your collection** with insights showing cuisine distribution, exploration coverage, and cooking patterns

### Limitations

- **English only.** Recipe importing, ingredient parsing, tagging, search, and meal planning all assume English-language content. Non-English recipes may fail to parse correctly, receive incorrect tags, or be removed during quality checks. Multilingual support is not currently planned.

---

## Prerequisites

Before setting up Aye Chef, you need:

### 1. A running Mealie instance

Aye Chef is a companion tool for [Mealie](https://mealie.io), an open-source recipe manager. Mealie stores your recipes; Aye Chef adds AI meal planning on top.

If you don't have Mealie yet, set it up first: [Mealie Installation Guide](https://docs.mealie.io/documentation/getting-started/installation/)

### 2. Hardware requirements

Aye Chef runs a local embedding model (`nvidia/llama-embed-nemotron-8b`, 8B parameters) for semantic recipe search. This model loads in bfloat16 and requires significant memory:

| Platform | Minimum RAM | Notes |
|----------|-------------|-------|
| **Apple Silicon** (M1/M2/M3/M4) | 32GB unified memory | 16GB is too tight with OS overhead |
| **NVIDIA GPU** | 16GB VRAM | RTX 4080, A5000, A100, etc. |
| **CPU-only** | 24GB system RAM | Works but embedding generation is slow |

The model downloads ~16GB on first use and stays cached in a Docker volume.

> **Why such a large model?** Smaller embedding models were tested but couldn't perform the semantic search tasks required by the AI meal planner (e.g., finding recipes by cooking style, protein type, or complexity). The 8B model's 4096-dimensional embeddings provide the retrieval quality the agentic planner depends on.

### 3. Docker & Docker Compose

Aye Chef runs as two Docker containers (web panel + background worker).

- [Install Docker](https://docs.docker.com/get-docker/)
- Docker Compose V2 is included with modern Docker Desktop

### 4. An OpenRouter API key

Aye Chef uses [OpenRouter](https://openrouter.ai) to access LLMs (GPT-4o-mini by default) for meal planning, recipe tagging, ingredient parsing, and shopping list generation.

1. Create an account at https://openrouter.ai
2. Add credits (meal planning typically costs $0.01-0.05 per plan)
3. Generate an API key at https://openrouter.ai/keys

### 5. A Mealie API token

This lets Aye Chef read and write recipes in your Mealie instance.

1. Log into your Mealie server
2. Go to **User Settings** (click your profile icon → Settings)
3. Scroll to **API Tokens**
4. Create a new token and copy it

### 6. (Optional) Brave Search API key

Only needed if you want automatic image backfill for recipes that don't have photos. Get one at https://brave.com/search/api/ — the free tier is sufficient.

---

## Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/deepseekcoder2/ayechef.git
cd ayechef
```

### Step 2: Start the containers

```bash
docker compose up -d
```

The first build takes several minutes — it builds the Docker image and installs Python dependencies.

> **Note:** The embedding model (~16GB) downloads on first use, not during build. This happens automatically when you first import recipes or run maintenance tools. The download is cached in a Docker volume so it only happens once.

### Step 3: Open the web panel

Go to **http://localhost:8080**

You'll see a **"Setup required"** banner at the top. This is expected.

### Step 4: Configure your Mealie connection

1. Click the **"Setup required"** banner — it takes you to the Status page
2. Click **"Go to Settings"** to jump to the Settings section on the dashboard
3. Under **Connection**, enter your Mealie server URL (e.g., `http://192.168.1.100:9000`)
4. Click **Save**

### Step 5: Add your API credentials

Still in the Settings section:

1. Expand **API Credentials**
2. Enter your **Mealie API Token** (from Prerequisites step 5)
3. Enter your **OpenRouter API Key** (from Prerequisites step 4)
4. Click **Save**

> You can verify the connection using the **"Test Connection"** button. It will confirm whether Aye Chef can reach your Mealie server with the provided token.

### Step 6: Check system status

Go to **Status** (via the nav bar or the banner link) and confirm all checks are green:

| Check | What it means |
|-------|---------------|
| Config file | `data/config.yaml` exists and is readable |
| Mealie connection | Can reach your Mealie server with the token |
| OpenRouter API | API key is valid |
| Recipe index | Local search database is built (empty is OK for now) |
| Embedding model | Model is downloaded and loadable |

The recipe index and embedding model checks will resolve after you import recipes (next step).

### Step 7: Import your first recipes

Aye Chef needs recipes in your Mealie instance to plan meals from. If you already have recipes in Mealie, skip to Step 8.

**Option A: Import from a recipe website**
1. Go to **Import** in the nav bar
2. Enter a URL from a supported recipe site
3. Click **Import** — this scrapes, parses, tags, and indexes the recipe

**Option B: Bulk import from a site**
1. Go to **Import → Bulk Import**
2. Enter a recipe website URL
3. Select how many recipes to import
4. The system discovers recipe URLs and imports them in the background

**Option C: Teach Aye Chef a new site**
1. Go to **Import → Learn New Site**
2. Enter any recipe website URL
3. The LLM analyzes the site structure and generates a custom scraper

### Step 8: Build the search index

After importing recipes, build the vector search index:

1. On the **Dashboard**, scroll to **Maintenance**
2. Click **Repair & Optimize** under "Routine"
3. This tags untagged recipes, parses ingredients, and builds the search index

The first run may take a few minutes if you imported many recipes. The embedding model downloads automatically if not already cached (~16GB, one-time).

### Step 9: Plan your first meal

1. On the **Dashboard**, click **Plan Week**
2. Optionally add special instructions (e.g., "Use up leftover chicken", "No fish this week")
3. Click **Generate** — the AI plans your week and produces a shopping list
4. View the result in **Jobs** once it completes

---

## Important: How Aye Chef Works with Mealie

Aye Chef maintains its own search index of your recipes. Understanding this will save you confusion:

**Always import recipes through Aye Chef, not Mealie.** Aye Chef's import pipeline does much more than just adding a recipe — it scrapes the page, parses ingredients into structured data, assigns cuisine/protein/method tags, and indexes the recipe for vector search. Recipes added directly through Mealie's UI skip all of this, so the AI meal planner won't know they exist.

**If you already have recipes in Mealie** (or add some directly later), run **Repair & Optimize** from the Dashboard → Maintenance section. This syncs your Mealie library into Aye Chef's index by tagging, parsing, and indexing any recipes it doesn't know about yet.

**After any bulk changes in Mealie** — manual adds, edits, deletes — run Repair & Optimize again to keep the search index in sync.

**Meal plans draw from your indexed recipes only.** If the AI seems to have a limited pool to pick from, go to **Status** and check the recipe index count. That number is how many recipes the planner can actually see. If it's lower than your Mealie total, run Repair & Optimize.

---

## Configuration

All configuration happens through the web UI. The underlying files are:

### data/config.yaml

Created automatically on first boot from `config.yaml.example`. You can edit it through the web UI (Settings panel) or directly.

```yaml
connection:
  mealie_url: "http://your-mealie-server:9000"

mealie:
  use_direct_db: false  # Enable DB mode for faster reads (requires volume mount) WARNING: Experimental

household:
  servings: 4
  meal_types: [lunch, dinner]
  description: ""  # Who you're cooking for (e.g., "Family of 4")

preferences:
  cuisines: []              # Favorite cuisines (empty = AI varies)
  dietary_restrictions: []  # Hard restrictions (e.g., vegetarian, gluten-free)

# AI context (optional) - natural language guidance for the AI
personal:
  dietary: []   # Dietary notes, one per line (e.g., "Kids are picky eaters")
  cooking: ""   # Kitchen constraints (e.g., "30 min weeknights")

pantry:
  staples:
    - salt
    - pepper
    - olive oil
    # Items you always have (excluded from shopping lists)

llm:
  chat_model: "openai/gpt-4o-mini"
  embedding_model: "nvidia/llama-embed-nemotron-8b"

# Optional: Image search for recipes without photos
image_search:
  use_vision_validation: true  # AI verifies images are food (slower but accurate)
```

See `config.yaml.example` for the full reference with all options documented.

### API Tokens

Configure via the web UI: **Settings → API Credentials**

| Token | Required | Where to Get |
|-------|----------|--------------|
| Mealie Token | Yes | Mealie → User Settings → API Tokens |
| OpenRouter API Key | Yes | https://openrouter.ai/keys |
| Brave Search API Key | No | https://brave.com/search/api/ (for image backfill) |

Credentials are stored in `data/secrets.yaml`. Environment variables (`MEALIE_TOKEN`, `OPENROUTER_API_KEY`, `BRAVE_API_KEY`) take priority if set.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐        ┌─────────────────────────┐    │
│  │   web (Flask)   │        │   worker (Huey)         │    │
│  │   Port 8080     │        │   Background Jobs       │    │
│  │                 │        │                         │    │
│  │  Control Panel  │───────▶│  Executes CLI tools     │    │
│  │  - Meal Planning│  Jobs  │  via subprocess         │    │
│  │  - Recipe Import│        │                         │    │
│  │  - Maintenance  │        │  - orchestrator.py      │    │
│  │  - Diagnostics  │        │  - import_site.py       │    │
│  └─────────────────┘        │  - mealie_parse.py      │    │
│           │                 │  - automatic_tagger.py  │    │
│           │                 │  - etc.                 │    │
│           │                 └─────────────────────────┘    │
│           │                            │                   │
│           ▼                            ▼                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Shared Volume: ./data                  │   │
│  │  - config.yaml     - job_status.db                  │   │
│  │  - recipe_index.db - job_outputs/                   │   │
│  │  - huey.db         - logs/                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Mealie API    │
                    │  (your server)  │
                    └─────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Web Panel** (`panel/`) | Flask UI for all operations |
| **Tool Registry** (`panel/tools/registry.py`) | Maps UI actions to CLI commands |
| **Job System** (`panel/jobs/`) | Huey task queue for background execution |
| **CLI Backend** | Python scripts that implement all logic |
| **Recipe URLs** (`recipe_urls/`) | URL discovery library with auto-discovery |

### MealieClient

`MealieClient` (in `mealie_client.py`) is the unified client for all Mealie API operations. It supports two data access modes:

| Mode | Description | Performance |
|------|-------------|-------------|
| **API Mode** (default) | All reads/writes go through Mealie's REST API | Standard |
| **DB Mode** (experimental) | Reads directly from Mealie's SQLite database | ~500x faster reads |

**How it works:**
- **Reads**: In DB mode, recipe data is read directly from Mealie's SQLite database at `/mealie-data/mealie.db`
- **Writes**: Always go through the API (Mealie must handle its own schema and side effects)

**When to use DB mode:**
- You have Mealie's data directory mounted as a Docker volume
- You're doing bulk operations (imports, maintenance, indexing)
- You want faster response times for read-heavy workloads

**Enabling DB mode:**

DB mode requires mounting Mealie's data volume into the Aye Chef containers. A separate compose file is provided for this:

```bash
# 1. Find your Mealie data volume
docker volume ls | grep mealie

# 2. Create .env with your volume name
echo "MEALIE_VOLUME_NAME=your_mealie_volume_name" > .env

# 3. Enable in data/config.yaml (or via Settings → Advanced in the web UI)
#    mealie:
#      use_direct_db: true

# 4. Start with both compose files
docker compose -f docker-compose.yml -f docker-compose.dbmode.yml up -d
```

---

## Web Panel Features

### Dashboard (`/`)
- Connection status indicator
- Quick meal plan generation with customization (cuisines, dietary restrictions, special instructions)
- **Special instructions** - One-shot prompts for each planning session (e.g., "Use up leftover chicken", "Skip fish this week")
- Active job monitoring
- Maintenance tools organized by category (Routine, Cleanup, Advanced)
- Settings panel with theme selector (18+ themes including dark modes)
- **Personal context** settings for dietary needs, household details, and cooking constraints
- API credentials management (Mealie token, OpenRouter, Brave Search)

### Insights (`/insights`)
- **Collection Personality** - MBTI-style archetype for your recipe collection (e.g., "S-P-W" = Specialist, Poultry-Centric, Wok-driven)
- **Exploration Stats** - Track how many recipes you've actually cooked vs. your total collection
- **This Week** - Summary of current meal plan with cuisine distribution
- **Top Cuisines** - Visual breakdown of your recipe collection by cuisine

### Import Recipes (`/import`)
- Single recipe URL import with full processing (uses the [recipe-scrapers](https://github.com/hhursev/recipe-scrapers) library, which supports hundreds of sites)
- Bulk site import with category/collection selection
- "Learn New Site" - LLM-powered URL discovery for any recipe website

### Jobs (`/jobs`)
- Real-time output streaming (Server-Sent Events)
- Job history with status indicators
- Cancel running jobs
- Sharable Direct Message export for completed meal plans

### System Status (`/status`)
- Health checks for all prerequisites (config, Mealie, OpenRouter, index)
- Cuisine readiness check with link to Insights
- Recipe quality metrics (tagged vs untagged)
- Mealie tags seeding for new installations
- Embedding model status

---

## Registered Tools

The panel executes these CLI tools via background jobs:

### Planning
| Tool | CLI Command | Description |
|------|-------------|-------------|
| `plan_week` | `python orchestrator.py` | Generate personalized meal plan and shopping list (supports `--temp-prompt` for one-shot instructions) |

### Import
| Tool | CLI Command | Description |
|------|-------------|-------------|
| `import_recipe` | `python import_recipe.py` | Import single recipe |
| `import_site` | `python import_site.py --yes` | Import from website |
| `add_site` | `python add_site.py --yes` | Learn new site |
| `test_import` | `python test_import.py` | Test import |

### Maintenance

Tools are organized into groups for easier navigation:

**Routine** (safe to run regularly)
| Tool | CLI Command | When to Use |
|------|-------------|-------------|
| `repair_optimize` | `python utils/recipe_maintenance.py` | After importing recipes, or if search seems off |
| `health_check` | `python utils/recipe_maintenance.py --quick` | To diagnose issues before taking action |
| `backfill_images` | `python recipe_images.py --backfill` | After imports or for AI-generated recipes |

**Cleanup** (deletes data ⚠️)
| Tool | CLI Command | When to Use |
|------|-------------|-------------|
| `cleanup_recipes` | `python utils/cleanup_duplicates.py --all --confirm` | After bulk imports that created duplicates |
| `cleanup_meals` | `python utils/cleanup_meal_data.py --confirm` | To start fresh with meal planning |

**Advanced** (recovery and debugging)
| Tool | CLI Command | When to Use |
|------|-------------|-------------|
| `rebuild_search` | `python utils/rebuild_search_index.py` | Only if search returns wrong results or errors |
| `clear_cache` | `python tools/invalidate_cache.py` | If suggestions seem stuck on old patterns |
| `fix_equipment` | `python utils/label_equipment.py --apply` | If shopping lists include kitchen equipment |
| `diagnose_recipe` | `python automatic_tagger.py --diagnose=<slug>` | To troubleshoot a specific recipe |

*⚠️ = Destructive operation (shows confirmation before running)*

---

### Adding Sites

No site scrapers are included by default. Use the "Learn New Site" feature to generate scrapers for any recipe website:

**Via Web UI:**
1. Go to Import → "Learn New Site"
2. Enter any recipe website URL
3. The LLM analyzes the site and generates a scraper

**Via CLI:**
```bash
python add_site.py https://example-recipe-site.com
```

Generated scrapers are saved to `recipe_urls/sites/` and auto-discovered on next use.

---

## Docker Services

### docker-compose.yml (default)

Runs the web panel and background worker. No additional configuration required.

Two services:
- **`web`** — Gunicorn serving the Flask panel on port 8080
- **`worker`** — Huey consumer (2 threads, keeps the embedding model in memory)

### docker-compose.dbmode.yml (optional)

Adds read-only access to Mealie's SQLite database for ~500x faster reads. See [Enabling DB mode](#enabling-db-mode) above.

```bash
docker compose -f docker-compose.yml -f docker-compose.dbmode.yml up -d
```

---

## Project Structure

```
ayechef/
├── panel/                      # Flask web panel
│   ├── app.py                  # Application factory
│   ├── routes/                 # Route blueprints
│   │   ├── main.py             # Dashboard routes
│   │   ├── import_recipes.py   # Import routes
│   │   ├── jobs.py             # Job management
│   │   ├── diagnostics.py      # Health checks
│   │   └── insights.py         # Collection insights
│   ├── insights/               # Insights data layer
│   │   ├── __init__.py
│   │   └── queries.py          # Collection analytics queries
│   ├── jobs/                   # Background job system
│   │   ├── huey_config.py      # Huey SQLite setup
│   │   ├── runner.py           # Generic job runner
│   │   └── status.py           # Job status tracking
│   ├── tools/
│   │   └── registry.py         # Tool definitions → CLI mapping
│   └── templates/              # Jinja2 templates
│       ├── base.html           # Base layout with theme support
│       ├── index.html          # Dashboard
│       ├── insights/           # Collection insights
│       ├── import/             # Recipe import
│       ├── jobs/               # Job management
│       └── diagnostics/        # System status
│
├── recipe_urls/                # URL discovery library
│   ├── __init__.py             # Auto-discovery API
│   ├── _abstract.py            # Base scraper class
│   └── sites/                  # User-generated scrapers (gitignored)
│       └── __init__.py         # Package marker only
│
├── orchestrator.py             # Full meal planning pipeline
├── chef_agentic.py             # Agentic AI meal planner
├── import_site.py              # Website recipe scraper
├── import_recipe.py            # Single recipe import
├── add_site.py                 # LLM scraper generator
├── bulk_import_smart.py        # Bulk import with processing
├── mealie_parse.py             # Ingredient parsing
├── automatic_tagger.py         # Recipe tagging
├── recipe_images.py            # Image backfill via web search
├── config.py                   # Centralized configuration
├── mealie_client.py            # Unified Mealie API client
├── recipe_rag.py               # Vector embeddings & search
├── recipe_ann_index.py         # USearch ANN index
├── batch_llm_processor.py      # LLM caching & batching
├── prompts.py                  # All LLM prompts
├── shopping_*.py               # Shopping list processing
│
├── utils/                      # Utility scripts
│   ├── cleanup_meal_data.py
│   ├── cleanup_duplicates.py
│   ├── rebuild_search_index.py
│   ├── recipe_maintenance.py
│   ├── label_equipment.py
│   └── ...
│
├── tools/                      # Development tools
│   ├── logging_utils.py        # Centralized logging
│   ├── invalidate_cache.py     # Cache management
│   └── seed_mealie_tags.py     # Tag seeding utility
│
├── Dockerfile                  # Container build
├── docker-compose.yml          # Service definitions (default)
├── docker-compose.dbmode.yml   # Optional: Mealie direct DB access
├── docker-entrypoint.sh        # Container startup script
│
└── data/                       # Runtime data (Docker volume)
    ├── config.yaml             # Main configuration (auto-created on first boot)
    ├── secrets.yaml            # API credentials (managed via web UI)
    ├── recipe_index.db         # Local recipe database
    ├── huey.db                 # Job queue database
    ├── logs/
    │   └── aye_chef.log
    └── job_outputs/            # Job output logs
```

---

## Development

### Running Without Docker

Requires Python 3.11+.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web panel
flask --app panel.app run --port 8080

# Start the background worker (separate terminal)
huey_consumer panel.jobs.huey_config.huey -w 2 -k thread
```

On first boot, `data/config.yaml` is created automatically from `config.yaml.example`. Configure your Mealie URL and API credentials through the web UI at http://localhost:8080, or edit `data/config.yaml` and `data/secrets.yaml` directly.

### Running CLI Tools Directly

```bash
# Test configuration
python config.py

# Plan meals
python orchestrator.py --week-start 2026-02-02

# Plan meals with special instructions
python orchestrator.py --week-start 2026-02-02 --temp-prompt "Use up leftover chicken"

# Import from site
python import_site.py https://thewoksoflife.com --sitemap --dry-run

# Parse ingredients
python mealie_parse.py --scan-unparsed --yes

# Quick health check
python utils/recipe_maintenance.py --quick
```

---

## Security

Aye Chef is designed for **trusted home networks**. It has no built-in user authentication — anyone who can reach the panel can use it.

### Recommendations

- **Home network (default):** No action needed. Your router's firewall keeps the panel local.
- **Exposing beyond LAN:** Put Aye Chef behind a reverse proxy with authentication (e.g., Traefik, Caddy, Authelia, nginx proxy manager). Do **not** expose port 8080 directly to the internet.
- **API credentials** are stored in `data/secrets.yaml`. Environment variables (`MEALIE_TOKEN`, `OPENROUTER_API_KEY`) take priority if set, if you prefer not to store credentials in a file.

### Reporting Vulnerabilities

If you discover a security issue, please report it privately via [GitHub Security Advisories](../../security/advisories/new) rather than opening a public issue.

---

## Troubleshooting

### "Setup required" banner won't go away

Go to **Status** and check which items are failing:
- **Config file** — Should resolve automatically. If not, verify `data/config.yaml` exists.
- **Mealie connection** — Check that `mealie_url` is correct and the Mealie server is reachable from the Aye Chef container. Use **Test Connection** in Settings.
- **OpenRouter API** — Verify your API key is valid and has credits.
- **Recipe index / Embedding model** — Run **Repair & Optimize** from Maintenance after importing recipes.

### Panel shows "Not connected"

1. Check `data/config.yaml` has correct `mealie_url`
2. Verify Mealie is running and accessible from where Aye Chef is running
3. Verify token is set via Settings → API Credentials
4. If running in Docker, use the Docker network hostname (e.g., `http://mealie:9000`) or the host IP — `localhost` inside a container refers to the container itself, not your host machine

### Jobs stuck in "pending"

1. Check worker is running: `docker compose logs worker`
2. Restart worker: `docker compose restart worker`

### Jobs hang after cancellation

If cancelling a job causes the worker to hang:
1. Use "Reset Worker" in Dashboard → Maintenance
2. Or restart: `docker compose restart worker`

### Embedding model not downloading

The model downloads automatically on first use. If it hangs or fails, pre-download manually:
```bash
docker compose exec worker python -c \
  "from sentence_transformers import SentenceTransformer; SentenceTransformer('nvidia/llama-embed-nemotron-8b')"
```

If you're behind a corporate proxy or firewall, ensure the container has internet access to reach Hugging Face (`huggingface.co`).

### Import fails for unsupported site

Use "Learn New Site" in the panel or:
```bash
python add_site.py https://new-recipe-site.com
```

### Docker: "localhost" doesn't reach Mealie

Inside Docker, `localhost` refers to the container, not your host machine. Use one of:
- Your host's LAN IP: `http://192.168.1.x:9000`
- Docker's host gateway: `http://host.docker.internal:9000` (Docker Desktop only)
- If Mealie is also in Docker on the same network: `http://mealie:9000`

---

## Credits

- **[Mealie](https://mealie.io)** - Recipe management system
- **[recipe-scrapers](https://github.com/hhursev/recipe-scrapers)** - Recipe parsing library
- **[mkayeterry/recipe-urls](https://github.com/mkayeterry/recipe-urls)** - URL discovery patterns
- **[nlynch31/Mealie-script](https://github.com/nlynch31/Mealie-script)** - Ingredient parsing approach

---

## License

MIT License - See [LICENSE](LICENSE) for details.
