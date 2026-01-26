# Quick Start - LabTree

## 3-Minute Setup

### Step 1: Start Services
```bash
cd "/Users/hayriyan/Desktop/Code/AI 2.2/Syllabus/Picsart/Web"
cp .env.example .env
docker-compose up -d
```

### Step 2: Wait 30 Seconds
Services will be healthy when you see logs like:
- `Uvicorn running on http://0.0.0.0:8000`
- `postgres ready for connections`

### Step 3: Open in Browser
- **Web**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

## Create Test User

1. Click "Sign Up" on landing page
2. Register: `student@example.com` / `password123`
3. Log in and explore!

## Common Commands

```bash
docker compose up -d      # Start
docker compose down        # Stop
docker compose logs -f     # View logs
docker compose ps          # Status
make help                  # Show all make commands
```

See README.md for full documentation.
