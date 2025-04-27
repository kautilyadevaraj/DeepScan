"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartConfig,
} from "./ui/chart";

// Updated data to display losses, phishing attacks, and deepfake incidents over years
const impactData = [
  {
    year: "2019",
    lossesInBillions: 3.5,
    phishingAttacks: 1.5,
    deepfakeIncidents: 0.5,
  },
  {
    year: "2020",
    lossesInBillions: 4.2,
    phishingAttacks: 1.8,
    deepfakeIncidents: 0.9,
  },
  {
    year: "2021",
    lossesInBillions: 6.9,
    phishingAttacks: 2.4,
    deepfakeIncidents: 1.6,
  },
  {
    year: "2022",
    lossesInBillions: 8.8,
    phishingAttacks: 3.1,
    deepfakeIncidents: 2.7,
  },
  {
    year: "2023",
    lossesInBillions: 10.3,
    phishingAttacks: 3.8,
    deepfakeIncidents: 4.2,
  },
  {
    year: "2024",
    lossesInBillions: 12.7,
    phishingAttacks: 4.5,
    deepfakeIncidents: 5.8,
  },
];

// Configuration for chart series
const chartConfig = {
  lossesInBillions: {
    label: "Losses (Billions USD)",
    color: "hsl(var(--chart-1))",
  },
  phishingAttacks: {
    label: "Phishing Attacks (Millions)",
    color: "hsl(var(--chart-2))",
  },
  deepfakeIncidents: {
    label: "Deepfake Incidents (Millions)",
    color: "hsl(var(--chart-3))",
  },
} satisfies ChartConfig;

export function Chart2() {
  return (
    <Card className="border shadow-lg">
      <CardHeader>
        <CardTitle className="text-xl font-bold">
          Cybercrime Impact Over Time
        </CardTitle>
        <CardDescription>
          Annual financial losses, phishing attack volumes, and deepfake
          incidents
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="w-full">
          <ChartContainer config={chartConfig}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={impactData}
                margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="colorLosses" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient
                    id="colorPhishing"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient
                    id="colorDeepfake"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip content={<ChartTooltipContent />} />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="lossesInBillions"
                  name="Losses (Billions USD)"
                  stroke="#3b82f6"
                  fill="url(#colorLosses)"
                  activeDot={{ r: 6 }}
                  animationDuration={1500}
                />
                <Area
                  type="monotone"
                  dataKey="phishingAttacks"
                  name="Phishing Attacks (Millions)"
                  stroke="#10b981"
                  fill="url(#colorPhishing)"
                  activeDot={{ r: 6 }}
                  animationDuration={1500}
                />
                <Area
                  type="monotone"
                  dataKey="deepfakeIncidents"
                  name="Deepfake Incidents (Millions)"
                  stroke="#f59e0b"
                  fill="url(#colorDeepfake)"
                  activeDot={{ r: 6 }}
                  animationDuration={1500}
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>
      </CardContent>
    </Card>
  );
}
