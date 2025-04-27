"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { ImageIcon, Video, AudioLines, ArrowRight, Globe } from "lucide-react";
import {
  Card,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";

export default function ToolsOverview() {
  const tools = [
    {
      title: "Image Detection",
      description:
        "Identify manipulated or AI-generated images with our advanced detection algorithms.",
      icon: ImageIcon,
      href: "/deepfake/image",
      color: "from-purple-500 to-indigo-500",
      bgColor: "bg-purple-50 dark:bg-purple-950/20",
    },
    {
      title: "Video Detection",
      description:
        "Analyze videos frame-by-frame to detect facial manipulations and inconsistencies.",
      icon: Video,
      href: "/deepfake/video",
      color: "from-blue-500 to-cyan-500",
      bgColor: "bg-blue-50 dark:bg-blue-950/20",
    },
    {
      title: "Audio Detection",
      description:
        "Detect synthetic voices and audio manipulations with frequency analysis.",
      icon: AudioLines,
      href: "/deepfake/audio",
      color: "from-teal-500 to-emerald-500",
      bgColor: "bg-teal-50 dark:bg-teal-950/20",
    },
    {
      title: "Phishing Detection",
      description:
        "Identify fraudulent websites and phishing attempts with our security analysis tools.",
      icon: Globe,
      href: "/deepfake/phishing",
      color: "from-orange-500 to-red-500",
      bgColor: "bg-orange-50 dark:bg-orange-950/20",
    },
  ];

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  };

  return (
    <div className="flex min-h-screen justify-center flex-col">
      <SiteHeader />
      <main className="flex justify-center">
        <section className="container grid items-center gap-6 pb-8 pt-6 md:py-10">
          <div className="relative">
            {/* Background decorative elements */}
            <div className="absolute -top-10 -left-10 w-64 h-64 bg-purple-200/20 dark:bg-purple-900/10 rounded-full blur-3xl -z-10"></div>
            <div className="absolute -bottom-10 -right-10 w-64 h-64 bg-blue-200/20 dark:bg-blue-900/10 rounded-full blur-3xl -z-10"></div>

            <div className="flex max-w-[980px] flex-col items-start gap-2">
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Badge className="mb-2 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600">
                  Tools Suite
                </Badge>
                <h1 className="text-3xl font-extrabold leading-tight tracking-tighter md:text-4xl bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-blue-600">
                  Deepfake Detection Tools
                </h1>
                <p className="max-w-[700px] text-lg text-muted-foreground mt-2">
                  Choose from our suite of specialized tools designed to detect
                  manipulated content across different media formats.
                </p>
              </motion.div>
            </div>
          </div>

          <div className="flex justify-center mb-6 mt-4">
            <Tabs defaultValue="overview" className="w-full max-w-5xl">
              <TabsList className="grid w-full grid-cols-5 p-1">
                <TabsTrigger
                  value="overview"
                  className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500/90 data-[state=active]:to-blue-500/90 data-[state=active]:text-white"
                >
                  Overview
                </TabsTrigger>
                <TabsTrigger
                  value="image"
                  className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500 data-[state=active]:to-indigo-500 data-[state=active]:text-white"
                >
                  Image
                </TabsTrigger>
                <TabsTrigger
                  value="video"
                  className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500 data-[state=active]:to-cyan-500 data-[state=active]:text-white"
                >
                  Video
                </TabsTrigger>
                <TabsTrigger
                  value="audio"
                  className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-teal-500 data-[state=active]:to-emerald-500 data-[state=active]:text-white"
                >
                  Audio
                </TabsTrigger>
                <TabsTrigger
                  value="phishing"
                  className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-orange-500 data-[state=active]:to-red-500 data-[state=active]:text-white"
                >
                  Phishing
                </TabsTrigger>
              </TabsList>
              <TabsContent value="overview" className="mt-6">
                <motion.div
                  className="grid gap-6 md:grid-cols-2 lg:grid-cols-2"
                  variants={container}
                  initial="hidden"
                  animate="show"
                >
                  {tools.map((tool) => (
                    <motion.div key={tool.title} variants={item}>
                      <Card
                        className={`h-full overflow-hidden transition-all hover:shadow-lg hover:shadow-purple-200/50 dark:hover:shadow-purple-900/20 border border-muted/60 ${tool.bgColor}`}
                      >
                        <CardHeader>
                          <div
                            className={`w-14 h-14 rounded-lg flex items-center justify-center bg-gradient-to-br shadow-md mb-2 p-3 text-white ${tool.color}`}
                          >
                            <tool.icon className="h-8 w-8" />
                          </div>
                          <CardTitle className="mt-2 text-xl">
                            {tool.title}
                          </CardTitle>
                          <CardDescription className="text-sm">
                            {tool.description}
                          </CardDescription>
                        </CardHeader>
                        <CardFooter className="pt-4 pb-6">
                          <Button
                            asChild
                            className={`w-full bg-gradient-to-r ${tool.color} hover:opacity-90 transition-all duration-300 group`}
                          >
                            <Link
                              href={tool.href}
                              className="flex items-center justify-center"
                            >
                              Launch Tool
                              <motion.div
                                initial={{ x: 0 }}
                                whileHover={{ x: 4 }}
                                transition={{ duration: 0.2 }}
                                className="ml-2"
                              >
                                <ArrowRight className="h-4 w-4" />
                              </motion.div>
                            </Link>
                          </Button>
                        </CardFooter>
                      </Card>
                    </motion.div>
                  ))}
                </motion.div>
              </TabsContent>
              <TabsContent value="image" className="mt-6">
                <Card className="border bg-purple-50 dark:bg-purple-950/20 border-purple-200/50 dark:border-purple-900/20 overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 to-indigo-500"></div>
                  <CardHeader>
                    <div className="w-14 h-14 rounded-lg flex items-center justify-center bg-gradient-to-br from-purple-500 to-indigo-500 shadow-md mb-2 p-3 text-white">
                      <ImageIcon className="h-8 w-8" />
                    </div>
                    <CardTitle>Image Detection</CardTitle>
                    <CardDescription>
                      Identify manipulated or AI-generated images with our
                      advanced detection algorithms.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4 pt-6">
                    <p className="font-medium">
                      Our image detection tool analyzes:
                    </p>
                    <ul className="grid gap-2">
                      {[
                        "Facial inconsistencies and artifacts",
                        "Metadata and compression anomalies",
                        "Lighting and shadow inconsistencies",
                        "Unnatural blending and boundaries",
                      ].map((item, i) => (
                        <motion.li
                          key={i}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.1, duration: 0.3 }}
                          className="flex items-center gap-2"
                        >
                          <div className="h-1.5 w-1.5 rounded-full bg-gradient-to-r from-purple-500 to-indigo-500"></div>
                          <span>{item}</span>
                        </motion.li>
                      ))}
                    </ul>
                  </CardContent>
                  <CardFooter className="pt-2 pb-6">
                    <Button
                      asChild
                      className="w-full bg-gradient-to-r from-purple-500 to-indigo-500 hover:opacity-90 transition-all duration-300 group"
                    >
                      <Link
                        href="/deepfake/image"
                        className="flex items-center justify-center"
                      >
                        Launch Image Detection
                        <motion.div
                          initial={{ x: 0 }}
                          whileHover={{ x: 4 }}
                          transition={{ duration: 0.2 }}
                          className="ml-2"
                        >
                          <ArrowRight className="h-4 w-4" />
                        </motion.div>
                      </Link>
                    </Button>
                  </CardFooter>
                </Card>
              </TabsContent>
              <TabsContent value="video" className="mt-6">
                <Card className="border bg-blue-50 dark:bg-blue-950/20 border-blue-200/50 dark:border-blue-900/20 overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-cyan-500"></div>
                  <CardHeader>
                    <div className="w-14 h-14 rounded-lg flex items-center justify-center bg-gradient-to-br from-blue-500 to-cyan-500 shadow-md mb-2 p-3 text-white">
                      <Video className="h-8 w-8" />
                    </div>
                    <CardTitle>Video Detection</CardTitle>
                    <CardDescription>
                      Analyze videos frame-by-frame to detect facial
                      manipulations and inconsistencies.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4 pt-6">
                    <p className="font-medium">
                      Our video detection tool analyzes:
                    </p>
                    <ul className="grid gap-2">
                      {[
                        "Temporal inconsistencies between frames",
                        "Facial movement and expression anomalies",
                        "Audio-visual synchronization issues",
                        "Blending artifacts and boundary issues",
                      ].map((item, i) => (
                        <motion.li
                          key={i}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.1, duration: 0.3 }}
                          className="flex items-center gap-2"
                        >
                          <div className="h-1.5 w-1.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"></div>
                          <span>{item}</span>
                        </motion.li>
                      ))}
                    </ul>
                  </CardContent>
                  <CardFooter className="pt-2 pb-6">
                    <Button
                      asChild
                      className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:opacity-90 transition-all duration-300 group"
                    >
                      <Link
                        href="/deepfake/video"
                        className="flex items-center justify-center"
                      >
                        Launch Video Detection
                        <motion.div
                          initial={{ x: 0 }}
                          whileHover={{ x: 4 }}
                          transition={{ duration: 0.2 }}
                          className="ml-2"
                        >
                          <ArrowRight className="h-4 w-4" />
                        </motion.div>
                      </Link>
                    </Button>
                  </CardFooter>
                </Card>
              </TabsContent>
              <TabsContent value="audio" className="mt-6">
                <Card className="border bg-teal-50 dark:bg-teal-950/20 border-teal-200/50 dark:border-teal-900/20 overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-teal-500 to-emerald-500"></div>
                  <CardHeader>
                    <div className="w-14 h-14 rounded-lg flex items-center justify-center bg-gradient-to-br from-teal-500 to-emerald-500 shadow-md mb-2 p-3 text-white">
                      <AudioLines className="h-8 w-8" />
                    </div>
                    <CardTitle>Audio Detection</CardTitle>
                    <CardDescription>
                      Detect synthetic voices and audio manipulations with
                      frequency analysis.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4 pt-6">
                    <p className="font-medium">
                      Our audio detection tool analyzes:
                    </p>
                    <ul className="grid gap-2">
                      {[
                        "Voice frequency patterns and anomalies",
                        "Breathing patterns and natural pauses",
                        "Background noise consistency",
                        "Unnatural transitions and splicing artifacts",
                      ].map((item, i) => (
                        <motion.li
                          key={i}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.1, duration: 0.3 }}
                          className="flex items-center gap-2"
                        >
                          <div className="h-1.5 w-1.5 rounded-full bg-gradient-to-r from-teal-500 to-emerald-500"></div>
                          <span>{item}</span>
                        </motion.li>
                      ))}
                    </ul>
                  </CardContent>
                  <CardFooter className="pt-2 pb-6">
                    <Button
                      asChild
                      className="w-full bg-gradient-to-r from-teal-500 to-emerald-500 hover:opacity-90 transition-all duration-300 group"
                    >
                      <Link
                        href="/deepfake/audio"
                        className="flex items-center justify-center"
                      >
                        Launch Audio Detection
                        <motion.div
                          initial={{ x: 0 }}
                          whileHover={{ x: 4 }}
                          transition={{ duration: 0.2 }}
                          className="ml-2"
                        >
                          <ArrowRight className="h-4 w-4" />
                        </motion.div>
                      </Link>
                    </Button>
                  </CardFooter>
                </Card>
              </TabsContent>
              <TabsContent value="phishing" className="mt-6">
                <Card className="border bg-orange-50 dark:bg-orange-950/20 border-orange-200/50 dark:border-orange-900/20 overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-orange-500 to-red-500"></div>
                  <CardHeader>
                    <div className="w-14 h-14 rounded-lg flex items-center justify-center bg-gradient-to-br from-orange-500 to-red-500 shadow-md mb-2 p-3 text-white">
                      <Globe className="h-8 w-8" />
                    </div>
                    <CardTitle>Phishing Detection</CardTitle>
                    <CardDescription>
                      Identify fraudulent websites and phishing attempts with
                      our security analysis tools.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4 pt-6">
                    <p className="font-medium">
                      Our phishing detection tool analyzes:
                    </p>
                    <ul className="grid gap-2">
                      {[
                        "Domain age and registration details",
                        "SSL certificate validity and authenticity",
                        "Suspicious redirect chains and URL patterns",
                        "Form elements requesting sensitive information",
                        "Visual similarity to legitimate websites",
                      ].map((item, i) => (
                        <motion.li
                          key={i}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.1, duration: 0.3 }}
                          className="flex items-center gap-2"
                        >
                          <div className="h-1.5 w-1.5 rounded-full bg-gradient-to-r from-orange-500 to-red-500"></div>
                          <span>{item}</span>
                        </motion.li>
                      ))}
                    </ul>
                  </CardContent>
                  <CardFooter className="pt-2 pb-6">
                    <Button
                      asChild
                      className="w-full bg-gradient-to-r from-orange-500 to-red-500 hover:opacity-90 transition-all duration-300 group"
                    >
                      <Link
                        href="/deepfake/phishing"
                        className="flex items-center justify-center"
                      >
                        Launch Phishing Detection
                        <motion.div
                          initial={{ x: 0 }}
                          whileHover={{ x: 4 }}
                          transition={{ duration: 0.2 }}
                          className="ml-2"
                        >
                          <ArrowRight className="h-4 w-4" />
                        </motion.div>
                      </Link>
                    </Button>
                  </CardFooter>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </section>
      </main>
      <SiteFooter />
    </div>
  );
}
