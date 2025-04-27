"use client";

import type React from "react";

import { useState, useRef } from "react";
import { motion } from "framer-motion";
import {
  Upload,
  AlertCircle,
  CheckCircle2,
  Play,
  Pause,
  Volume2,
  VolumeX,
} from "lucide-react";
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
import { Slider } from "@/components/ui/slider";
import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";

interface PredictionResponse {
  final_prediction: string;
  final_confidence: number;
  model_prediction: string;
  model_confidence: number;
  heuristic_prediction: string;
  heuristic_confidence: number;
  heuristic_scores: {
    [key: string]: number;
  };
}

export default function AudioDetection() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      const url = URL.createObjectURL(selectedFile);
      setPreview(url);
      setResult(null);
      setError(null);
    }
  };

  const analyzeAudio = async () => {
    if (!file) return;

    setAnalyzing(true);
    setError(null);

    try {
      const reader = new FileReader();
      reader.readAsDataURL(file);

      const base64Data = await new Promise<string>((resolve, reject) => {
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = (error) => reject(error);
      });

      const audioBase64 = base64Data.split(",")[1];

      const response = await fetch("/api/audio", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ audio: audioBase64 }),
      });

      if (!response.ok) {
        throw new Error("Analysis failed");
      }

      const data: PredictionResponse = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to analyze audio. Please try again.");
    } finally {
      setAnalyzing(false);
    }
  };

  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = () => {
    if (audioRef.current) {
      audioRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleVolumeChange = (value: number[]) => {
    const newVolume = value[0];
    setVolume(newVolume);
    if (audioRef.current) {
      audioRef.current.volume = newVolume;
    }
  };

  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <main className="flex justify-center">
        <section className="container grid items-center gap-6 pb-8 pt-6 md:py-10">
          <div className="flex max-w-[980px] flex-col items-start gap-2">
            <h1 className="text-3xl font-extrabold leading-tight tracking-tighter md:text-4xl">
              Audio Deepfake Detection
            </h1>
            <p className="max-w-[700px] text-lg text-muted-foreground">
              Upload an audio file to analyze it for signs of voice manipulation
              or synthesis.
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
                  <CardTitle>Upload Audio</CardTitle>
                  <CardDescription>
                    Select an audio file to analyze for deepfake detection.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col items-center justify-center gap-4">
                    <div
                      className="flex h-64 w-full cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-muted-foreground/25 p-4 transition-colors hover:border-muted-foreground/50"
                      onClick={() =>
                        document.getElementById("audio-upload")?.click()
                      }
                    >
                      {preview ? (
                        <div className="flex h-full w-full flex-col items-center justify-center gap-4">
                          <div className="flex h-32 w-32 items-center justify-center rounded-full bg-primary/10">
                            <Button
                              size="icon"
                              variant="ghost"
                              className="h-16 w-16 rounded-full"
                              onClick={(e) => {
                                e.stopPropagation();
                                togglePlayback();
                              }}
                            >
                              {isPlaying ? (
                                <Pause className="h-8 w-8" />
                              ) : (
                                <Play className="h-8 w-8" />
                              )}
                            </Button>
                          </div>
                          <audio
                            ref={audioRef}
                            src={preview}
                            className="hidden"
                            onPlay={() => setIsPlaying(true)}
                            onPause={() => setIsPlaying(false)}
                            onEnded={() => setIsPlaying(false)}
                          />
                          <div className="flex w-full items-center gap-2">
                            <Button
                              size="icon"
                              variant="ghost"
                              className="h-8 w-8"
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleMute();
                              }}
                            >
                              {isMuted ? (
                                <VolumeX className="h-4 w-4" />
                              ) : (
                                <Volume2 className="h-4 w-4" />
                              )}
                            </Button>
                            <Slider
                              value={[volume]}
                              min={0}
                              max={1}
                              step={0.01}
                              onValueChange={handleVolumeChange}
                              onClick={(e) => e.stopPropagation()}
                              className="w-full"
                            />
                          </div>
                        </div>
                      ) : (
                        <>
                          <Upload className="mb-2 h-10 w-10 text-muted-foreground" />
                          <p className="text-sm text-muted-foreground">
                            Drag and drop or click to upload
                          </p>
                          <p className="text-xs text-muted-foreground">
                            Supports MP3, WAV, OGG (max 50MB)
                          </p>
                        </>
                      )}
                      <input
                        id="audio-upload"
                        type="file"
                        accept="audio/mp3,audio/wav,audio/ogg"
                        className="hidden"
                        onChange={handleFileChange}
                      />
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button
                    onClick={analyzeAudio}
                    disabled={!file || analyzing}
                    className="w-full"
                  >
                    {analyzing ? "Analyzing..." : "Analyze Audio"}
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
                    {error ? (
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>{error}</AlertDescription>
                      </Alert>
                    ) : analyzing ? (
                      <div className="flex w-full flex-col items-center gap-2">
                        <p className="text-center text-sm text-muted-foreground">
                          Analyzing audio patterns...
                        </p>
                        <Progress value={0} className="w-full" />
                      </div>
                    ) : result ? (
                      <div className="w-full">
                        <Alert
                          variant={
                            result.final_prediction === "fake"
                              ? "destructive"
                              : "default"
                          }
                        >
                          {result.final_prediction === "fake" ? (
                            <AlertCircle className="h-4 w-4" />
                          ) : (
                            <CheckCircle2 className="h-4 w-4" />
                          )}
                          <AlertTitle>
                            {result.final_prediction === "fake"
                              ? "Synthetic Voice Detected"
                              : "Authentic Audio"}
                          </AlertTitle>
                          <AlertDescription>
                            {result.final_prediction === "fake"
                              ? `This audio appears to be synthetically generated with ${(
                                  result.final_confidence * 100
                                ).toFixed(1)}% confidence.`
                              : `This audio appears to be authentic with ${(
                                  result.final_confidence * 100
                                ).toFixed(1)}% confidence.`}
                          </AlertDescription>
                        </Alert>

                        <div className="mt-4 space-y-3">
                          <h4 className="font-medium">Analysis Details</h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="font-medium">
                                Model Prediction:
                              </span>
                              <span>
                                {result.model_prediction} (
                                {(result.model_confidence * 100).toFixed(1)}%)
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="font-medium">
                                Heuristic Prediction:
                              </span>
                              <span>
                                {result.heuristic_prediction} (
                                {(result.heuristic_confidence * 100).toFixed(1)}
                                %)
                              </span>
                            </div>
                            <div className="mt-2">
                              <h5 className="font-medium mb-2">
                                Scores:
                              </h5>
                              <div className="grid grid-cols-2 gap-y-2 gap-x-4">
                                {Object.entries(result.heuristic_scores).map(
                                  ([key, value]) => (
                                    <div
                                      key={key}
                                      className="flex justify-between items-center"
                                    >
                                      <span className="capitalize">
                                        {key.replace(/_/g, " ")}:
                                      </span>
                                      <span className="font-mono">
                                        {(value * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                  )
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center text-center text-muted-foreground">
                        <p>No audio analyzed yet.</p>
                        <p className="text-sm">
                          Upload an audio file and click Analyze to get started.
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
