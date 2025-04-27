// "use client";

// import { useTheme } from "next-themes";
// import { useEffect, useState } from "react";
// import { CartesianGrid, Cell, Legend, Line, Pie, XAxis, YAxis } from "recharts";
// import {
//   ChartContainer,
//   ChartTooltip,
//   ChartTooltipContent,
//   type ChartConfig,
// } from "@/components/ui/chart";

// // Line chart config
// const lineChartConfig = {
//   losses: {
//     label: "Financial Losses",
//     color: "hsl(var(--chart-1))",
//   },
// } satisfies ChartConfig;

// // Pie chart config
// const pieChartConfig = {
//   politics: {
//     label: "Politics",
//     color: "hsl(var(--chart-1))",
//   },
//   entertainment: {
//     label: "Entertainment",
//     color: "hsl(var(--chart-2))",
//   },
//   finance: {
//     label: "Finance",
//     color: "hsl(var(--chart-3))",
//   },
//   social: {
//     label: "Social Media",
//     color: "hsl(var(--chart-4))",
//   },
//   education: {
//     label: "Education",
//     color: "hsl(var(--chart-5))",
//   },
// } satisfies ChartConfig;

// // Bar chart config
// const barChartConfig = {
//   detected: {
//     label: "Detected Deepfakes",
//     color: "hsl(var(--chart-1))",
//   },
// } satisfies ChartConfig;

// export function LineChart() {
//   const { theme } = useTheme();
//   const [chartData, setChartData] = useState([]);

//   useEffect(() => {
//     // Mock data
//     const data = [
//       { year: "2018", losses: 12 },
//       { year: "2019", losses: 25 },
//       { year: "2020", losses: 78 },
//       { year: "2021", losses: 130 },
//       { year: "2022", losses: 245 },
//       { year: "2023", losses: 320 },
//     ];
//     setChartData(data);
//   }, []);

//   return (
//     <ChartContainer config={lineChartConfig}>
//       <LineChart
//         data={chartData}
//         margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
//       >
//         <CartesianGrid strokeDasharray="3 3" />
//         <XAxis dataKey="year" />
//         <YAxis />
//         <ChartTooltip content={<ChartTooltipContent />} />
//         <Legend />
//         <Line
//           type="monotone"
//           dataKey="losses"
//           name="Financial Losses ($ Millions)"
//           stroke="hsl(var(--chart-1))"
//           strokeWidth={2}
//           dot={{ r: 4 }}
//           activeDot={{ r: 6 }}
//           animationDuration={1500}
//         />
//       </LineChart>
//     </ChartContainer>
//   );
// }

// export function PieChart() {
//   const [chartData, setChartData] = useState([]);

//   useEffect(() => {
//     // Mock data
//     const data = [
//       { name: "Politics", value: 35, id: "politics" },
//       { name: "Entertainment", value: 25, id: "entertainment" },
//       { name: "Finance", value: 20, id: "finance" },
//       { name: "Social Media", value: 15, id: "social" },
//       { name: "Education", value: 5, id: "education" },
//     ];
//     setChartData(data);
//   }, []);

//   return (
//     <ChartContainer config={pieChartConfig}>
//       <PieChart margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
//         <Pie
//           data={chartData}
//           cx="50%"
//           cy="50%"
//           labelLine={false}
//           outerRadius={80}
//           fill="#8884d8"
//           dataKey="value"
//           nameKey="name"
//           label={({ name, percent }) =>
//             `${name}: ${(percent * 100).toFixed(0)}%`
//           }
//           animationDuration={1500}
//         >
//           {chartData.map((entry) => (
//             <Cell
//               key={`cell-${entry.id}`}
//               fill={pieChartConfig[entry.id]?.color || "#8884d8"}
//             />
//           ))}
//         </Pie>
//         <ChartTooltip content={<ChartTooltipContent />} />
//         <Legend />
//       </PieChart>
//     </ChartContainer>
//   );
// }
