# LabTree - AI/ML Lab Platform

A comprehensive, full-stack web application for managing a 15-month ML/DL curriculum with automated testing and points-based progression.

## Features

✅ **Interactive Lab Tree** - Visualize all labs across 3 phases with dependencies and progress tracking  
✅ **Auto-Grading** - Submit GitHub repos and get instant test results  
✅ **Team Collaboration** - Form teams for team-based labs, share scores  
✅ **Progress Tracking** - Real-time points, phase progression, and dashboards  
✅ **Instructor Tools** - Create labs, configure tests, monitor cohort progress  
✅ **Local Testing** - Run tests in sandboxed Docker containers

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, React Router, Zustand |
| **Backend** | FastAPI, SQLModel, PostgreSQL |
| **Job Queue** | Celery + Redis |
| **Testing** | pytest, Docker |
| **DevOps** | Docker, docker-compose |

## Project Structure

```
├── README.md                 # Main documentation
├── QUICKSTART.md             # Quick start guide
├── docker-compose.yml        # Local dev setup
├── Makefile                  # Common commands
├── .env.example              # Environment template
│
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── main.py          # FastAPI app entry
│   │   ├── database.py      # DB connection
│   │   ├── config.py        # Settings
│   │   ├── security.py      # JWT auth
│   │   ├── models/schemas.py # ORM models
│   │   ├── api/             # Route handlers
│   │   ├── workers/         # Celery tasks
│   │   └── services/        # Business logic
│   ├── labs/                # Lab configurations
│   └── tests/               # Unit tests
│
└── frontend/                # React TypeScript SPA
    ├── src/
    │   ├── pages/           # Full├── QUICKSTART.   ├── components/      # Reusable components
    │   ├── hooks/           # React hooks
    │   ├── api/             # API client
    │   ├── App.tsx
    │   └── main.tsx
    └── package.json
```

## Quick Start

### With Docker (Recommended)

```bash
cd /path/to/Web
cp .env.example .env
docker-compose up -d
```

Access:
- **Frontend**: http://localhost:5173
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Without Docker

```bash
# Backend
cd backend
poetry install
poetry run uvicorn app.main:app --reload

# Frontend (in another terminal)
cd frontend
npm install
npm run dev
```

## Create Test User

1. Visit http://localhost:5173
2. Sign up: email=test@example.com, password=test123
3. Log in with same credentials
4. Browse labs → Submit solutions → Auto-grading

## Database Models

- **User**: Accounts, auth, progress tracking
- **Lab**: Course structure (code 1.1, 2.8, etc.)
- **Submission**: Student submissions with test results
- **TestCase**: Individual test specs per lab
- **Team**: Collaborative learning groups
- **Score**: Points tracking across phases

## API Documentation

Visit **http://localhost:8000/docs** for interactive Swagger UI with all endpoints.

### Key Endpoints

```http
POST   /api/auth/register
POST   /api/auth/login
GET    /api/auth/me

GET    /api/labs?phase=phase_1
GET    /api/labs/{lab_code}

POST   /api/submissions/{lab_code}
GET    /api/submissions/{submission_id}
```

## Lab Config Format

Example Lab 1.1 test configuration:

```python
LAB_1_1_CONFIG = {
    "code": "1.1",
    "phase": "phase_1",
    "points": 12,
    "test_cases": [
        {
            "name": "Data Structures",
            "points": 3,
            "command": "python -m pytest tests/test_data_structures.py -v",
            "timeout": 30
        },
        # ... more test cases
    ]
}
```

## Environment Variables

```env
DATABASE_URL=postgresql://labuser:labpass@localhost:5432/lab_platform
REDIS_URL=redis://localhost:6379
JWT_SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
VITE_API_BASE_URL=http://localhost:8000
```

## Common Commands

```bash
make help           # Show all commands
make up             # Start services
make down           # Stop services
make logs           # View logs
make test           # Run backend tests
make backend-shell  # SSH into backend
```

## Features Overview

### 🌳 Lab Tree
- Visual representation of 32+ labs across 3 phases
- Color-coded phases (blue, orange, purple)
- Progress badges (not started, in progress, passed)
- Interactive hover panels

### 📝 Lab Submission
- GitHub URL submission
- Automatic repo cloning
- Test execution in Docker sandbox
- Points calculation
- Result feedback

### 📊 Dashboards
- Personal progress per phase
- Team statistics
- Submission history
- Instructor cohort analytics

### 👥 Teams
- Create or join learning teams (2-4 people)
- Shared submissions for team labs
- Distributed points to team members

## Deployment

For production deployment, see README.md for full setup instructions covering:
- Environment separation
- Database backups
- SSL/TLS certificates
- Monitoring & logging
- Scaling considerations

## Future Enhancements

- [ ] D3.js animated tree visualization
- [ ] Real-time WebSocket updates
- [ ] GitHub OAuth integration
- [ ] Advanced analytics dashboards
- [ ] Slack notifications
- [ ] Email digest reports
- [ ] Mobile app

## License

MIT License - see LICENSE file for details

---

**Built for the 15-Month ML/DL Lab Program** 🚀
