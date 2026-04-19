import type { BatterPrediction } from "../types";
import { BatterRow } from "./BatterRow";

interface TeamCardProps {
  teamAbbr: string;
  batters: BatterPrediction[];
}

export function TeamCard({ teamAbbr, batters }: TeamCardProps) {
  if (batters.length === 0) return null;

  const topProb = batters[0].hr_probability;
  const teamTotal = batters.reduce((sum, b) => sum + b.hr_probability, 0);

  return (
    <div className="team-card">
      <div className="team-card-header">
        <h3>{teamAbbr}</h3>
        <div className="team-stats">
          <span className="team-top" title="Top batter HR probability">
            Top: {(topProb * 100).toFixed(1)}%
          </span>
          <span className="team-expected" title="Expected team HRs">
            Exp HRs: {teamTotal.toFixed(2)}
          </span>
        </div>
      </div>
      <div className="team-batters">
        {batters.map((batter, i) => (
          <BatterRow key={batter.batter_id} batter={batter} rank={i + 1} />
        ))}
      </div>
    </div>
  );
}
