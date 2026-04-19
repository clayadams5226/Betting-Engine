# MLB HR Prediction Engine

A web app that predicts which MLB batters are most likely to hit a home run on a given day. Uses Statcast batter/pitcher stats with platoon splits, PCA decorrelation to avoid double-counting correlated metrics, and environmental factors (weather, park dimensions) to produce calibrated probabilities.

**Stack:** Python (FastAPI) backend + React (Vite + TypeScript) frontend

## Prerequisites

- [Python 3.11+](https://www.python.org/downloads/)
- [Node.js 18+](https://nodejs.org/)
- An [OpenWeatherMap API key](https://openweathermap.org/api) (free tier works)

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/clayadams5226/Betting-Engine.git
cd Betting-Engine
```

### 2. Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv

# Activate it:
#   Windows (Command Prompt):
venv\Scripts\activate
#   Windows (Git Bash):
source venv/Scripts/activate
#   macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
# Still in the backend/ directory
cp .env.example .env
```

Open `backend/.env` and replace the placeholder with your actual API key:

```
OPENWEATHERMAP_API_KEY=your_actual_key_here
```

### 4. Frontend

Open a **second terminal** and run:

```bash
cd frontend

# Install dependencies
npm install
```

## Running the App

You need **two terminals** running simultaneously.

### Terminal 1 -- Backend

```bash
cd backend
source venv/Scripts/activate   # or your OS-specific activation command
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. You can verify it's running by visiting `http://localhost:8000/health`.

### Terminal 2 -- Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## Usage

1. Open `http://localhost:5173` in your browser
2. Select a date using the date picker
3. The app fetches that day's MLB schedule, lineups, and probable pitchers
4. Predictions are displayed grouped by game/team, sorted by HR probability

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/predictions/{date}` | HR probabilities for all batters on a given date (format: `YYYY-MM-DD`) |
| `GET /api/games/{date}` | Game schedule with probable pitchers for a given date |
| `GET /health` | Health check |

## Project Structure

```
backend/
  app/
    main.py              # FastAPI entry point
    api/routes.py        # API endpoint definitions
    data/
      loader.py          # Statcast + MLB API data ingestion
      blender.py         # Time-window blending (30-day + season)
      normalizer.py      # Z-score normalization
      env_score.py       # Park + weather factor scoring
      weather.py         # OpenWeatherMap integration
      stadiums.json      # Stadium coordinates, roof types, park factors
    models/
      pca.py             # PCA decorrelation (batter + pitcher)
      combiner.py        # Logistic regression combining pillars
      calibrator.py      # Isotonic regression calibration
      pipeline.py        # End-to-end prediction pipeline
      saved/             # Serialized model weights
    training/
      train.py           # Historical data training pipeline
      evaluate.py        # Backtesting + calibration curves
    cache/
      db.py              # SQLite caching for daily stats

frontend/
  src/
    App.tsx              # Main app component
    api/predictions.ts   # API client
    components/
      DatePicker.tsx     # Date selector
      TeamCard.tsx       # Team grouping with batter rows
      BatterRow.tsx      # Individual batter prediction display
      GameHeader.tsx     # Matchup + weather + park info
    types/index.ts       # TypeScript type definitions
```

## How It Works

1. **Platoon splits** -- Stats are filtered by batter/pitcher handedness matchup
2. **Time-window blending** -- Recent (30-day) and season stats are blended with stat-specific weights
3. **Z-score normalization** -- Stats normalized against league population
4. **PCA decorrelation** -- Separate PCA for batter stats (6 -> 2 components) and pitcher stats (6 -> 2 components) to avoid double-counting correlated metrics
5. **Environmental scoring** -- Park HR factor, temperature, wind, humidity, and roof type
6. **Logistic combination** -- Batter score, pitcher score, and environment combined via logistic regression with an interaction term
7. **Isotonic calibration** -- Post-processing to correct probability tail bias
