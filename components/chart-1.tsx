"use client";

import * as React from "react";
import { TrendingUp } from "lucide-react";
import { Label, Pie, PieChart } from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart";

// Sector vulnerability data
const sectorVulnerability = [
  { name: "Government", value: 35 },
  { name: "Finance", value: 25 },
  { name: "Healthcare", value: 15 },
  { name: "Energy", value: 12 },
  { name: "Education", value: 8 },
  { name: "Other", value: 5 },
];

// Define explicit colors for each sector
const SECTOR_COLORS = [
  "#10b981", // Government - emerald
  "#3b82f6", // Finance - blue
  "#8b5cf6", // Healthcare - violet
  "#f59e0b", // Energy - amber
  "#ec4899", // Education - pink
  "#6b7280", // Other - gray
];

// Map data to include fill colors from SECTOR_COLORS
const chartData = sectorVulnerability.map((sector, index) => ({
  ...sector,
  fill: SECTOR_COLORS[index],
}));

const chartConfig = {
  value: { label: "Vulnerability (%)" },
  Government: { label: "Government", color: SECTOR_COLORS[0] },
  Finance: { label: "Finance", color: SECTOR_COLORS[1] },
  Healthcare: { label: "Healthcare", color: SECTOR_COLORS[2] },
  Energy: { label: "Energy", color: SECTOR_COLORS[3] },
  Education: { label: "Education", color: SECTOR_COLORS[4] },
  Other: { label: "Other", color: SECTOR_COLORS[5] },
} satisfies ChartConfig;

export function Chart1() {
  const totalVulnerability = React.useMemo(() => {
    return chartData.reduce((sum, item) => sum + item.value, 0);
  }, []);

  return (
    <Card className="flex flex-col">
      <CardHeader className="items-center pb-0">
        <CardTitle className="text-xl font-bold">
          Sector Vulnerability Distribution
        </CardTitle>
        <CardDescription>Percentage share by sector</CardDescription>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[350px] [&_.recharts-pie-label-text]:fill-foreground"
        >
          <PieChart>
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Pie
              data={chartData}
              dataKey="value"
              nameKey="name"
              innerRadius={80}
              strokeWidth={2}
            >
              {/* Center text label */}
              <Label
                content={({ viewBox }) => {
                  if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                    return (
                      <text
                        x={viewBox.cx}
                        y={viewBox.cy}
                        textAnchor="middle"
                        dominantBaseline="middle"
                      >
                        <tspan
                          x={viewBox.cx}
                          y={viewBox.cy}
                          className="fill-foreground text-3xl font-bold"
                        >
                          {totalVulnerability}%
                        </tspan>
                        <tspan
                          x={viewBox.cx}
                          y={(viewBox.cy || 0) + 24}
                          className="fill-muted-foreground"
                        >
                          Total
                        </tspan>
                      </text>
                    );
                  }
                }}
              />
            </Pie>
            <ChartLegend
              content={<ChartLegendContent nameKey="name" />}
              className="-translate-y-2 flex-wrap gap-2 [&>*]:basis-1/4 [&>*]:justify-center"
            />
          </PieChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col gap-2 text-sm">
        <div className="flex items-center gap-2 font-medium leading-none">
          Trending up by 5.2% this month <TrendingUp className="h-4 w-4" />
        </div>
        <div className="leading-none text-muted-foreground">
          Showing sector vulnerability breakdown
        </div>
      </CardFooter>
    </Card>
  );
}
