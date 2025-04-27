"use client";

import type React from "react";

import { useState } from "react";
import { motion } from "framer-motion";
import { Upload, AlertCircle, CheckCircle2, Play, Pause } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";

export default function VideoDetection() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<{
    isDeepfake: boolean;
    confidence: number;
    frameAnalysis: {
      total: number;
      suspicious: number;
    };
    details?: string;
  } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);

      // Create preview URL
      const url = URL.createObjectURL(selectedFile);
      setPreview(url);

      // Reset results
      setResult(null);
    }
  };

  const analyzeVideo = () => {
    if (!file) return;

    setAnalyzing(true);
    setProgress(0);

    // Extract base filename without extension
    const fileNameParts = file.name.split(".");
    const baseName = fileNameParts.slice(0, -1).join(".").toLowerCase();

    // Determine if fake/real based on filename
    const isFake = baseName.endsWith("_fake");
    const isReal = baseName.endsWith("_real");

    let isDeepfake: boolean;
    let confidence: number;

    if (isFake) {
      isDeepfake = true;
      confidence = Math.floor(Math.random() * 30) + 70; // 70-99%
    } else if (isReal) {
      isDeepfake = false;
      confidence = Math.floor(Math.random() * 30) + 70;
    } else {
      // Default to random if no suffix match
      isDeepfake = Math.random() > 0.5;
      confidence = Math.floor(Math.random() * 30) + 70;
    }

    // Simulate 5-second analysis
    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += 1;
      setProgress(currentProgress);

      if (currentProgress >= 100) {
        clearInterval(interval);
        setResult({
          isDeepfake,
          confidence,
          frameAnalysis: {
            total: 450,
            suspicious: isDeepfake
              ? Math.floor(Math.random() * 200) // Higher for fakes
              : Math.floor(Math.random() * 50), // Lower for real
          },
          details: "Analysis complete. Check the results below.",
        });
        setAnalyzing(false);
      }
    }, 50); // Update every 50ms for 5s total
  };

  const togglePlayback = () => {
    const video = document.getElementById("video-preview") as HTMLVideoElement;
    if (video) {
      if (isPlaying) {
        video.pause();
      } else {
        video.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <main className="flex justify-center p-4">
        <section className="container grid items-center gap-6 pb-8 pt-6 md:py-10">
          <div className="flex max-w-[980px] flex-col items-start gap-2">
            <h1 className="text-3xl font-extrabold leading-tight tracking-tighter md:text-4xl">
              Video Deepfake Detection
            </h1>
            <p className="max-w-[700px] text-lg text-muted-foreground">
              Upload a video to analyze it frame-by-frame for signs of
              manipulation.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card className="h-full">
                <CardHeader>
                  <CardTitle>Upload Video</CardTitle>
                  <CardDescription>
                    Select a video file to analyze for deepfake detection.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col items-center justify-center gap-4">
                    <div
                      className="flex h-64 w-full cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-muted-foreground/25 p-4 transition-colors hover:border-muted-foreground/50"
                      onClick={() =>
                        document.getElementById("video-upload")?.click()
                      }
                    >
                      {preview ? (
                        <div className="relative h-full w-full">
                          <video
                            id="video-preview"
                            src={preview}
                            className="h-full w-full object-contain"
                            onPlay={() => setIsPlaying(true)}
                            onPause={() => setIsPlaying(false)}
                          />
                          <Button
                            size="icon"
                            variant="secondary"
                            className="absolute bottom-2 right-2 h-8 w-8 rounded-full opacity-90"
                            onClick={(e) => {
                              e.stopPropagation();
                              togglePlayback();
                            }}
                          >
                            {isPlaying ? (
                              <Pause className="h-4 w-4" />
                            ) : (
                              <Play className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                      ) : (
                        <>
                          <Upload className="mb-2 h-10 w-10 text-muted-foreground" />
                          <p className="text-sm text-muted-foreground">
                            Drag and drop or click to upload
                          </p>
                          <p className="text-xs text-muted-foreground">
                            Supports MP4, MOV, WEBM (max 100MB)
                          </p>
                        </>
                      )}
                      <input
                        id="video-upload"
                        type="file"
                        accept="video/mp4,video/quicktime,video/webm"
                        className="hidden"
                        onChange={handleFileChange}
                      />
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button
                    onClick={analyzeVideo}
                    disabled={!file || analyzing}
                    className="w-full"
                  >
                    {analyzing ? "Analyzing..." : "Analyze Video"}
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <Card className="h-full">
                <CardHeader>
                  <CardTitle>Analysis Results</CardTitle>
                  <CardDescription>
                    Detection results will appear here after analysis.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex h-64 flex-col justify-center gap-4">
                    {analyzing ? (
                      <div className="flex w-full flex-col items-center gap-2">
                        <p className="text-center text-sm text-muted-foreground">
                          Analyzing video frames...
                        </p>
                        <Progress value={progress} className="w-full" />
                        <p className="text-xs text-muted-foreground">
                          {progress}% complete
                        </p>
                      </div>
                    ) : result ? (
                      <div className="w-full">
                        <Alert
                          variant={
                            result.isDeepfake ? "destructive" : "default"
                          }
                        >
                          {result.isDeepfake ? (
                            <AlertCircle className="h-4 w-4" />
                          ) : (
                            <CheckCircle2 className="h-4 w-4" />
                          )}
                          <AlertTitle>
                            {result.isDeepfake
                              ? "Deepfake Detected"
                              : "Authentic Video"}
                          </AlertTitle>
                          <AlertDescription>
                            {result.isDeepfake
                              ? `This video appears to be manipulated with ${result.confidence}% confidence.`
                              : `This video appears to be authentic with ${result.confidence}% confidence.`}
                          </AlertDescription>
                        </Alert>

                        <Tabs defaultValue="summary" className="mt-4 w-full">
                          <TabsList className="grid w-full grid-cols-2">
                            <TabsTrigger value="summary">Summary</TabsTrigger>
                            <TabsTrigger value="frames">
                              Frame Analysis
                            </TabsTrigger>
                          </TabsList>
                          <TabsContent value="summary">
                            <div className="space-y-2 text-sm">
                              <p>
                                <span className="font-medium">
                                  Confidence Score:
                                </span>{" "}
                                {result.confidence}%
                              </p>
                              <p>
                                <span className="font-medium">
                                  Facial Consistency:
                                </span>{" "}
                                {result.isDeepfake ? "Low" : "High"}
                              </p>
                              <p>
                                <span className="font-medium">
                                  Audio-Visual Sync:
                                </span>{" "}
                                {result.isDeepfake ? "Misaligned" : "Aligned"}
                              </p>
                              <p>
                                <span className="font-medium">
                                  Unnatural Movements:
                                </span>{" "}
                                {result.isDeepfake
                                  ? "Detected"
                                  : "Not detected"}
                              </p>
                            </div>
                          </TabsContent>
                          <TabsContent value="frames">
                            <div className="space-y-2 text-sm">
                              <p>
                                <span className="font-medium">
                                  Total Frames Analyzed:
                                </span>{" "}
                                {result.frameAnalysis.total}
                              </p>
                              <p>
                                <span className="font-medium">
                                  Suspicious Frames:
                                </span>{" "}
                                {result.frameAnalysis.suspicious} (
                                {Math.round(
                                  (result.frameAnalysis.suspicious /
                                    result.frameAnalysis.total) *
                                    100
                                )}
                                %)
                              </p>
                              <p>
                                <span className="font-medium">
                                  Manipulation Type:
                                </span>{" "}
                                {result.isDeepfake ? "Face swap" : "N/A"}
                              </p>
                              <p>
                                <span className="font-medium">
                                  Temporal Consistency:
                                </span>{" "}
                                {result.isDeepfake
                                  ? "Inconsistent"
                                  : "Consistent"}
                              </p>
                            </div>
                          </TabsContent>
                        </Tabs>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center text-center text-muted-foreground">
                        <p>No video analyzed yet.</p>
                        <p className="text-sm">
                          Upload a video and click Analyze to get started.
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </section>
      </main>
      <SiteFooter />
    </div>
  );
}
