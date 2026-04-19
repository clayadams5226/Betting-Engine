interface DatePickerProps {
  value: string;
  onChange: (date: string) => void;
  loading: boolean;
}

export function DatePicker({ value, onChange, loading }: DatePickerProps) {
  return (
    <div className="date-picker">
      <label htmlFor="game-date">Game Date</label>
      <input
        id="game-date"
        type="date"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={loading}
      />
      {loading && <span className="loading-indicator">Loading...</span>}
    </div>
  );
}
