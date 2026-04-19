# MLB Home Run Prediction App

## Context
Build a web app that predicts which MLB batters are most likely to hit a home run on a given day. Uses Statcast batter/pitcher stats (with L/R platoon splits), environmental factors, and PCA decorrelation to avoid double-counting correlated stats. Results displayed in a React dashboard grouped by team, sorted by probability.

**Stack:** Python (FastAPI) backend + React frontend  
**Data:** pybaseball + MLB Stats API + OpenWeatherMap (all free)  
**Model:** Weighted scoring with PCA decorrelation + logistic combination layer

---

## Architecture

```
React Frontend (Vite + TypeScript)
    |
FastAPI Backend
    |
    +-- Data Layer (pybaseball, MLB Stats API, Weather API)
    +-- Model Layer (PCA, Logistic Combiner, Calibrator)
    +-- Cache Layer (SQLite for daily stat caching)
```

---

## Data Sources

| Source | Data | Notes |
|--------|------|-------|
| **pybaseball** | Statcast pitch-level data | Barrel%, Hard-Hit%, EV, Launch Angle, Sweet Spot%, xSLG, xwOBA. Must compute aggregates from pitch-level. L/R splits via `stand`/`p_throws` columns |
| **MLB Stats API** | Schedules, lineups, probable pitchers | Free, no auth. `statsapi.mlb.com/api/v1/schedule?hydrate=probablePitcher,lineup` |
| **OpenWeatherMap** | Game-time weather | Free tier: 1000 calls/day. Temperature, wind, humidity |
| **Baseball Savant** | Park HR factors | Statcast park factors (index=100 is avg). Scraped or cached seasonally |
| **Static JSON** | Stadium coordinates + roof type | For weather API lookups. ~30 entries, hardcoded |

---

## Model Pipeline (7 Stages)

### Stage 1: Platoon Split Selection
- Identify batter hand (L/R) vs opposing pitcher hand (L/R)
- Pull batter stats **vs that pitcher hand** and pitcher stats **vs that batter hand**
- When platoon split has <50 PA, apply Bayesian shrinkage toward overall stats:
  `X_split = (PA_split / (PA_split + 50)) * X_platoon + (50 / (PA_split + 50)) * X_overall`

### Stage 2: Time-Window Blending (30-day + season)
`X_blended = w_recent * X_30day + (1 - w_recent) * X_season`

Stat-specific weights based on stabilization rates:

| Stat | w_recent | Rationale |
|------|----------|-----------|
| Avg Exit Velocity | 0.50 | Most stable batted ball metric |
| Max Exit Velocity | 0.55 | Raw power; stabilizes fast |
| Hard-Hit% | 0.45 | Stabilizes relatively quickly |
| Avg Launch Angle | 0.40 | Approach-driven |
| Sweet Spot% | 0.38 | Moderate stability |
| Barrel% | 0.35 | Barrels are rare events |
| HR/9 | 0.25 | Very slow to stabilize |
| HR/FB% | 0.20 | Extremely noisy |
| Barrel% allowed | 0.35 | Mirrors batter Barrel% |
| Hard-Hit% allowed | 0.45 | Mirrors batter Hard-Hit% |
| xSLG against | 0.40 | Expected stat; moderate |
| xwOBA against | 0.40 | Expected stat; moderate |

**Early-season fallback** (pre-May): blend 30% current season + 70% prior season regressed 25% toward league average.

### Stage 3: Z-Score Normalization
`Z_i = (X_blended_i - mu_league_i) / sigma_league_i`

Computed against league population for same handedness split, updated daily.

### Stage 4: PCA Decorrelation (separate per pillar)

**Why separate PCA?** Batter and pitcher stats are from different populations. No cross-pillar correlation exists to remove.

**Batter PCA (6 stats -> 2 components):**
- PC1 (~60% variance): Raw batted-ball power (Barrel%, Hard-Hit%, Avg EV, Max EV, Sweet Spot%)
- PC2 (~18% variance): Launch approach (Avg Launch Angle, Sweet Spot%)

**Pitcher PCA (6 stats -> 2 components):**
- PC1 (~65% variance): HR vulnerability (HR/9, HR/FB%, Barrel% allowed, Hard-Hit% allowed)
- PC2 (~17% variance): Expected quality-of-contact (xSLG, xwOBA against)

**Retention rule:** eigenvalue > 1.0 or cumulative variance >= 80%.

**Pillar score:** variance-weighted composite:
`Batter_Score = (var_ratio_1 * PC1 + var_ratio_2 * PC2) / (var_ratio_1 + var_ratio_2)`

### Stage 5: Environmental Score (physics-informed, no PCA)
```
Env_Score = Park_adj * (1 + Temp_adj + Wind_adj + Humidity_adj + Roof_adj)

Park_adj  = ParkHRFactor / 100
Temp_adj  = 0.002 * (Temp_F - 72)           # ~2% per 10 degrees
Wind_adj  = wind_to_CF_mph * 0.003          # ~3% per mph blowing out
Humidity_adj = 0.001 * (Humidity% - 50)     # small effect
Roof_adj  = 0 (open) | -0.02 (closed) | -0.03 (dome)
```

### Stage 6: Logistic Combination
```
logit(P_HR) = beta_0 + beta_1 * Batter_Score + beta_2 * Pitcher_Score
            + beta_3 * ln(Env_Score) + beta_4 * (Batter_Score * Pitcher_Score)

P_HR = 1 / (1 + exp(-logit))
```
- Interaction term captures non-additive matchup effects
- Sigmoid bounds output to [0, 1]
- Train on 2-3 seasons of historical matchup data
- Base rate anchor: league avg ~3.5-4.5% per batter per game

### Stage 7: Isotonic Calibration
Post-processing step to correct systematic bias in probability tails. Fit on held-out 20% of training data.

---

## Project Structure

```
backend/
    app/
        main.py                 # FastAPI app entry point
        api/
            routes.py           # /predictions/{date}, /games/{date}
        data/
            loader.py           # pybaseball + MLB API data ingestion
            blender.py          # Time-window blending logic
            normalizer.py       # Z-score normalization
            env_score.py        # Environmental factor computation
            weather.py          # OpenWeatherMap integration
            stadiums.json       # Stadium coords, roof type, park factors
        models/
            pca.py              # PCA fitting + transform (batter & pitcher)
            combiner.py         # Logistic regression combining pillars
            calibrator.py       # Isotonic regression calibration
            pipeline.py         # End-to-end daily prediction pipeline
        training/
            train.py            # Historical data training pipeline
            evaluate.py         # Backtesting + calibration curves
        cache/
            db.py               # SQLite caching for daily stats
    requirements.txt
    
frontend/
    src/
        App.tsx
        components/
            DatePicker.tsx      # Date selector
            TeamCard.tsx        # Team grouping with batter rows
            BatterRow.tsx       # Individual batter prediction display
            GameHeader.tsx      # Matchup + weather + park info
        api/
            predictions.ts      # API client
        types/
            index.ts            # TypeScript types
    package.json
```

---

## Implementation Order

### Phase 1: Data Pipeline
1. **Static data setup** - `stadiums.json` with coords, roof type, park HR factors for all 30 stadiums
2. **`loader.py`** - Pull Statcast data via pybaseball (batter + pitcher, with platoon filtering). Pull schedules + lineups via MLB Stats API
3. **`weather.py`** - OpenWeatherMap integration for game-time weather at stadium coords
4. **`blender.py`** - Time-window blending with stat-specific weights + early-season fallback
5. **`normalizer.py`** - Z-score normalization against league population
6. **`env_score.py`** - Environmental score computation
7. **`cache/db.py`** - SQLite cache to avoid re-pulling same-day stats

### Phase 2: Model
8. **`pca.py`** - Separate PCA for batter (6 stats -> 2 PCs) and pitcher (6 stats -> 2 PCs)
9. **`combiner.py`** - Logistic regression with interaction term
10. **`calibrator.py`** - Isotonic regression post-processing
11. **`pipeline.py`** - End-to-end: date in -> sorted predictions out
12. **`training/train.py`** - Train on historical data (2022-2024 seasons)

### Phase 3: API
13. **`main.py` + `routes.py`** - FastAPI endpoints: `GET /api/predictions/{date}`, `GET /api/games/{date}`

### Phase 4: Frontend
14. **React app** - Date picker -> fetch predictions -> display grouped by team, sorted by probability
15. **Styling** - Clean dashboard with confidence indicators, matchup details, weather badges

---

## API Endpoints

```
GET /api/predictions/{date}
  -> { games: [{ home_team, away_team, venue, weather, batters: [{ name, team, hr_probability, batter_score, pitcher_score, env_score, confidence, opposing_pitcher }] }] }

GET /api/games/{date}
  -> { games: [{ game_id, home, away, time, venue, probable_pitchers }] }
```

---

## Verification Plan
1. **Unit tests** for each pipeline stage (blender, normalizer, PCA, combiner, env_score)
2. **Backtest** on 2024 season: run predictions for each game day, compare predicted probabilities vs actual HR outcomes
3. **Calibration curve**: plot predicted vs observed HR rates across probability bins
4. **Smoke test**: run full pipeline for today's games, verify output format and reasonable probabilities (expect range ~1-15%)
5. **Frontend**: select a date, verify teams display correctly grouped and sorted
