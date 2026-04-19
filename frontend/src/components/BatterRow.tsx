import type { BatterPrediction } from "../types";

interface BatterRowProps {
  batter: BatterPrediction;
  rank?: number;
}

export function BatterRow({ batter, rank }: BatterRowProps) {
  const probPct = (batter.hr_probability * 100).toFixed(1);
  const confidenceClass = `confidence-${batter.confidence}`;

  return (
    <div className={`batter-row ${confidenceClass}`}>
      {rank !== undefined && <span className="rank">#{rank}</span>}
      <span className="batter-name">{batter.batter_name}</span>
      <span className="position">{batter.position}</span>
      <span className="team-badge">{batter.team}</span>
      <span className="vs-pitcher">vs {batter.opposing_pitcher}</span>
      <span className="probability">{probPct}%</span>
      <div className="prob-bar">
        <div
          className="prob-fill"
          style={{ width: `${Math.min(batter.hr_probability * 500, 100)}%` }}
        />
      </div>
      <div className="scores">
        <span title="Batter power score">B: {batter.batter_score.toFixed(2)}</span>
        <span title="Pitcher vulnerability score">P: {batter.pitcher_score.toFixed(2)}</span>
      </div>
      <span className={`confidence-badge ${confidenceClass}`}>
        {batter.confidence}
      </span>
    </div>
  );
}
