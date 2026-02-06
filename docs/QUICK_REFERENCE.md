# Aye Chef Quick Reference

## Docker (Recommended)

### Start/Stop
```bash
docker compose up -d      # Start
docker compose down       # Stop
docker compose logs -f    # View logs
docker compose restart    # Restart both services
```

### Access
- **Web Panel:** http://localhost:8080
- **Mealie:** Your configured URL (check Settings)

### Common Tasks via Panel

| Task | Location |
|------|----------|
| Plan a week | Dashboard → "Generate Meal Plan" |
| Import recipes | Dashboard → "Import Recipes" |
| Check job status | Top-right → "History" or `/jobs` |
| Run maintenance | Dashboard → Maintenance section |
| Cleanup recipes | Dashboard → Maintenance (or CLI below) |
| View diagnostics | Maintenance → "System Status" |

---

## CLI Reference (Advanced)

### End-to-End Pipeline
```bash
# Full pipeline: Plan → Shopping → WhatsApp
python orchestrator.py --week-start 2026-02-03

# With tuning
python orchestrator.py --week-start 2026-02-03 --candidate-k 100 --max-refines 2

# Dry run (no writes)
python orchestrator.py --dry-run --week-start 2026-02-03
```

### Recipe Import
```bash
# Single recipe
python import_recipe.py https://example.com/recipe-url

# Bulk from supported site
python import_site.py https://thewoksoflife.com --sitemap

# Preview categories
python import_site.py https://thewoksoflife.com --list-categories

# Import specific categories
python import_site.py https://thewoksoflife.com --categories "Chinese,Japanese"

# Learn new site (LLM analysis)
python add_site.py https://new-recipe-site.com
```

### Maintenance
```bash
# Quick health check
python utils/recipe_maintenance.py --quick

# Full maintenance
python utils/recipe_maintenance.py

# Re-parse unparsed ingredients
python mealie_parse.py --scan-unparsed --yes

# Tag untagged recipes
python utils/bulk_tag.py --all --yes

# Rebuild vector index
python utils/rebuild_ann_index.py

# Clean old meal plans
python utils/cleanup_meal_data.py --confirm

# Remove duplicate recipes (with (1), (2) in names)
python utils/cleanup_duplicates.py --duplicates --confirm

# Remove invalid recipes (no ingredients/instructions)
python utils/cleanup_duplicates.py --invalid --confirm

# Full cleanup (both duplicates AND invalid)
python utils/cleanup_duplicates.py --all --confirm
```

### Diagnostics
```bash
# Test configuration
python config.py

# Diagnose specific recipe
python automatic_tagger.py --diagnose=recipe-slug-here
```

---

## Parameters

### orchestrator.py
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--week-start` | Next Monday | Start date (YYYY-MM-DD) |
| `--candidate-k` | 25 | Candidates per role (↑ = better, slower) |
| `--max-refines` | 1 | Retry attempts per role |
| `--dry-run` | false | Test mode (no writes) |
| `--skip-validation` | false | Skip pre-flight checks |
| `--skip-parsing` | false | Skip ingredient scan |
| `--cuisines` | — | Comma-separated preferred cuisines |
| `--restrictions` | — | Comma-separated dietary restrictions |

### import_site.py
| Parameter | Default | Description |
|-----------|---------|-------------|
| `url` | (required) | Website URL |
| `--sitemap` | false | Use sitemap for full coverage |
| `--categories` | — | Comma-separated categories |
| `--list-categories` | false | List available categories |
| `--dry-run` | false | Preview without saving |
| `--validate-urls` | false | Check for 404s |

---

## Performance Guide

### Tuning Meal Planning

| Mode | Command | Time | Quality |
|------|---------|------|---------|
| Fast | `--candidate-k 10 --max-refines 0` | 5-10 min | Basic |
| Balanced | `--candidate-k 25 --max-refines 1` | 15-20 min | Good |
| High Quality | `--candidate-k 100 --max-refines 2` | 20-30 min | Best |

### Docker Resource Usage
- **Web container:** ~100MB RAM
- **Worker container:** ~2-4GB RAM (embedding model)
- **Embedding model download:** ~2GB (first run)

---

## File Locations

### Docker Volumes
| Path | Contents |
|------|----------|
| `./data/` | Config, databases, secrets, job outputs |
| `huggingface_cache` | Embedding model cache |

### Key Files
| File | Purpose |
|------|---------|
| `data/config.yaml` | Main configuration |
| `data/secrets.yaml` | API credentials (managed via web UI) |
| `data/recipe_index.db` | Recipe database + embeddings |
| `data/huey.db` | Job queue database |
| `data/job_status.db` | Job status tracking |
| `data/job_outputs/` | Job output logs |

---

## Troubleshooting

### Panel shows "Not connected"
1. Check `data/config.yaml` has correct `mealie_url`
2. Test with: Settings → "Test" button
3. Verify token is set via Settings → API Credentials

### Jobs stuck in "pending"
```bash
docker compose logs worker        # Check worker logs
docker compose restart worker     # Restart worker
```

### Jobs hang after cancellation
Use Dashboard → Maintenance → "Reset Worker" or:
```bash
docker compose restart worker
```

### Embedding model issues
```bash
# Pre-download model
docker compose exec worker python -c \
  "from sentence_transformers import SentenceTransformer; SentenceTransformer('nvidia/llama-embed-nemotron-8b')"
```

### View job output
```bash
# In container
docker compose exec web cat data/job_outputs/<job-id>.txt

# Or directly
cat data/job_outputs/<job-id>.txt
```

### Clear all jobs
```bash
# Via panel: Jobs → "Clear Completed"
# Or manually:
rm data/job_outputs/*.txt
rm data/job_status.db
```

---

## URLs

| Service | Default URL |
|---------|-------------|
| Web Panel | http://localhost:8080 |
| Jobs | http://localhost:8080/jobs |
| Import | http://localhost:8080/import |
| Status | http://localhost:8080/status |
| OpenRouter API | https://openrouter.ai/api/v1 |

---

## Supported Sites

No sites are pre-configured. Use **Learn New Site** (via the web panel or `python add_site.py <url>`) to add support for any recipe website. The system will analyze the site's HTML structure and generate a scraper automatically.

Check your configured sites:
```bash
python import_site.py --list-sites
```
