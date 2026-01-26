# How to Run LabTree Locally

## Option 1: Using Docker (Recommended - but docker-compose needs fixing)

Your Docker is installed but `docker-compose` is missing. Use `docker compose` instead (newer syntax):

```bash
# Install docker-compose if not available
brew install docker-compose

# Then run:
docker compose up -d
```

Or use the newer integrated version (Docker Desktop 20.10+):

```bash
docker compose up -d
```

---

## Option 2: Run Without Docker (Easiest - Local Development)

### Prerequisites
- Python 3.11+ 
- Node.js 18+
- PostgreSQL 15 (or SQLite for testing)
- Redis (optional, for async jobs)

### Step 1: Setup Backend

```bash
cd backend

# Install Python dependencies
pip install poetry
poetry install

# Create .env file
cat > .env << 'ENV'
DATABASE_URL=sqlite:///lab_platform.db
REDIS_URL=redis://localhost:6379
JWT_SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
ENV

# Run migrations/create database
poetry run python -c "from app.database import create_db_and_tables; create_db_and_tables()"

# Start backend
poetry run uvicorn app.main:app --reload
```

Backend will be at: **http://localhost:8000**
API Docs: **http://localhost:8000/docs**

### Step 2: Setup Frontend (in another terminal)

```bash
cd frontend

# Install Node dependencies
npm install

# Start dev server
npm run dev
```

Frontend will be at: **http://localhost:5173**

---

## Option 3: Quick Test (Without Database)

If you just want to test the API without a database:

```bash
cd backend
pip install fastapi uvicorn
python -m uvicorn app.main:app --reload
```

Then visit: **http://localhost:8000/docs**

---

## Troubleshooting

### `docker-compose` not found
```bash
# Install it
brew install docker-compose

# Or use newer syntax
docker compose up -d
```

### Python version issues
```bash
# Check your Python version
python3 --version

# Should be 3.11+. If not:
# Use Conda: conda install python=3.11
# Or Homebrew: brew install python@3.11
```

### Port already in use
```bash
# Backend (8000)
lsof -i :8000
kill -9 <PID>

# Frontend (5173)
lsof -i :5173
kill -9 <PID>
```

### poetry not found
```bash
pip install poetry
poetry install
```

---

## Quick Start Commands

**Preferred (Docker + docker-compose):**
```bash
docker compose up -d
# Visit http://localhost:5173
```

**Fallback (Local Python + Node):**
```bash
# Terminal 1 - Backend
cd backend && poetry install && poetry run uvicorn app.main:app --reload

# Terminal 2 - Frontend  
cd frontend && npm install && npm run dev

# Visit http://localhost:5173
```

**Minimal (Just API):**
```bash
cd backend && pip install fastapi uvicorn && python -m uvicorn app.main:app --reload
# Visit http://localhost:8000/docs
```

