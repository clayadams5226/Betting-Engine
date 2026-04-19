export interface BatterPrediction {
  batter_name: string;
  batter_id: number;
  team: string;
  position: string;
  batting_order: string;
  opposing_pitcher: string;
  hr_probability: number;
  batter_score: number;
  pitcher_score: number;
  env_score: number;
  confidence: "high" | "medium" | "low";
}

export interface GameWeather {
  temp_f: number;
  wind_speed_mph: number;
  wind_deg: number;
  humidity_pct: number;
  description: string;
}

export interface GamePrediction {
  game_id: number;
  home_team: string;
  home_abbr: string;
  away_team: string;
  away_abbr: string;
  venue: string;
  home_pitcher: string;
  away_pitcher: string;
  weather: GameWeather;
  env_score: number;
  batters: BatterPrediction[];
  batters_by_team: Record<string, BatterPrediction[]>;
}

export interface PredictionResponse {
  date: string;
  game_count: number;
  games: GamePrediction[];
}

export interface TopPick extends BatterPrediction {
  venue: string;
  game_id: number;
  home_team: string;
  away_team: string;
}

export interface TopPicksResponse {
  date: string;
  total_batters: number;
  top_picks: TopPick[];
}
