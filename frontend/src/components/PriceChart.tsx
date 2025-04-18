import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

interface PriceChartProps {
  timestamps: string[];
  prices: number[];
  ma5: number[];
  rsi14: number[];
}

export default function PriceChart({ timestamps, prices, ma5, rsi14 }: PriceChartProps) {
  // Format the data for Recharts
  const data = timestamps.map((timestamp, i) => {
    const date = new Date(timestamp);
    return {
      time: `${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`,
      price: prices[i],
      ma5: ma5[i],
      rsi14: rsi14 ? rsi14[i] : undefined,
    };
  });

  return (
    <div className="bg-white p-4 rounded-lg shadow-md">
      <h3 className="text-lg font-semibold mb-4">Price History</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="time" 
            tick={{ fontSize: 12 }}
            interval="preserveStartEnd"
          />
          <YAxis 
            yAxisId="left"
            domain={['auto', 'auto']} 
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            yAxisId="right"
            orientation="right"
            domain={[0, 100]}
            tick={{ fontSize: 12 }}
            hide={!rsi14}
          />
          <Tooltip />
          <Legend />
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="price" 
            stroke="#3b82f6" 
            dot={false} 
            name="Price"
          />
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="ma5" 
            stroke="#10b981" 
            dot={false} 
            name="MA(5)"
          />
          {rsi14 && (
            <Line 
              yAxisId="right"
              type="monotone" 
              dataKey="rsi14" 
              stroke="#f59e0b" 
              dot={false} 
              name="RSI(14)"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
