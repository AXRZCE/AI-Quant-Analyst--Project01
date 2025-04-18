import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";

interface SentimentGaugeProps {
  sentiment: {
    positive: number;
    neutral: number;
    negative: number;
  };
}

export default function SentimentGauge({ sentiment }: SentimentGaugeProps) {
  const data = [
    { name: "Positive", value: sentiment.positive, color: "#10b981" },
    { name: "Neutral", value: sentiment.neutral, color: "#6b7280" },
    { name: "Negative", value: sentiment.negative, color: "#ef4444" },
  ];

  return (
    <div className="bg-white p-4 rounded-lg shadow-md">
      <h3 className="text-lg font-semibold mb-4">News Sentiment</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip 
              formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
            />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
