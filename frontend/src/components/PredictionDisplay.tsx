interface PredictionDisplayProps {
  prediction: number;
}

export default function PredictionDisplay({ prediction }: PredictionDisplayProps) {
  const pct = (prediction * 100).toFixed(2);
  const isPositive = prediction >= 0;
  const color = isPositive ? "text-green-600" : "text-red-600";
  const icon = isPositive ? "↑" : "↓";

  return (
    <div className="bg-white p-6 rounded-lg shadow-md text-center">
      <h3 className="text-lg font-semibold mb-2">Prediction</h3>
      <div className={`text-3xl font-bold ${color} flex items-center justify-center`}>
        <span className="mr-2">{icon}</span>
        <span>{pct}%</span>
      </div>
      <p className="text-gray-600 mt-2">
        Predicted return for next trading period
      </p>
    </div>
  );
}
