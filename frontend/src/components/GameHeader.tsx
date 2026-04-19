import type { GamePrediction } from "../types";

interface GameHeaderProps {
  game: GamePrediction;
}

export function GameHeader({ game }: GameHeaderProps) {
  const weather = game.weather;

  return (
    <div className="game-header">
      <div className="matchup">
        <span className="team away">{game.away_abbr}</span>
        <span className="at">@</span>
        <span className="team home">{game.home_abbr}</span>
      </div>
      <div className="game-details">
        <span className="venue">{game.venue}</span>
        <div className="pitchers">
          <span>{game.away_pitcher} vs {game.home_pitcher}</span>
        </div>
      </div>
      <div className="game-conditions">
        <span className="weather" title={weather.description}>
          {Math.round(weather.temp_f)}°F
        </span>
        <span className="wind">
          {Math.round(weather.wind_speed_mph)} mph wind
        </span>
        <span className="humidity">{weather.humidity_pct}% humidity</span>
        <span className={`env-score ${game.env_score >= 1 ? "favorable" : "unfavorable"}`}>
          Env: {game.env_score.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
