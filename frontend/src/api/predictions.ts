import type { PredictionResponse, TopPicksResponse } from "../types";

const API_BASE = "http://localhost:8000/api";

export async function fetchPredictions(
  date: string
): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/predictions/${date}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch predictions: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchTopPicks(
  date: string,
  limit: number = 20
): Promise<TopPicksResponse> {
  const res = await fetch(`${API_BASE}/top-picks/${date}?limit=${limit}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch top picks: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchGames(date: string) {
  const res = await fetch(`${API_BASE}/games/${date}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch games: ${res.statusText}`);
  }
  return res.json();
}
