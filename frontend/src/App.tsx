import { useEffect, useState } from "react";
import "./App.css";
import { fetchPredictions } from "./api/predictions";
import { DatePicker } from "./components/DatePicker";
import { GameHeader } from "./components/GameHeader";
import { TeamCard } from "./components/TeamCard";
import type { GamePrediction } from "./types";

function todayStr(): string {
  return new Date().toISOString().slice(0, 10);
}

function App() {
  const [selectedDate, setSelectedDate] = useState(todayStr());
  const [games, setGames] = useState<GamePrediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedDate) return;

    setLoading(true);
    setError(null);

    fetchPredictions(selectedDate)
      .then((data) => {
        setGames(data.games);
      })
      .catch((err) => {
        setError(err.message);
        setGames([]);
      })
      .finally(() => setLoading(false));
  }, [selectedDate]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>MLB HR Predictor</h1>
        <p className="subtitle">
          Statcast-powered home run probability predictions
        </p>
        <DatePicker
          value={selectedDate}
          onChange={setSelectedDate}
          loading={loading}
        />
      </header>

      <main className="app-main">
        {error && <div className="error-banner">{error}</div>}

        {!loading && games.length === 0 && !error && (
          <div className="empty-state">
            No games found for {selectedDate}. Try selecting a different date.
          </div>
        )}

        {games.map((game) => (
          <section key={game.game_id} className="game-section">
            <GameHeader game={game} />
            <div className="teams-grid">
              {Object.entries(game.batters_by_team || {}).map(
                ([teamAbbr, batters]) => (
                  <TeamCard
                    key={teamAbbr}
                    teamAbbr={teamAbbr}
                    batters={batters}
                  />
                )
              )}
            </div>
          </section>
        ))}
      </main>

      <footer className="app-footer">
        <p>
          Data: Baseball Savant (Statcast) + MLB Stats API + OpenWeatherMap
        </p>
        <p>Model: PCA decorrelation + logistic combination + isotonic calibration</p>
      </footer>
    </div>
  );
}

export default App;
