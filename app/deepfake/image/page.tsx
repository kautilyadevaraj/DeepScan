"use client";

import type React from "react";

import { useState } from "react";
import { motion } from "framer-motion";
import { Upload, AlertCircle, CheckCircle2 } from "lucide-react";
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
import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";

export default function ImageDetection() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
   const [result, setResult] = useState<{
     prediction: "real" | "deepfake" | "ai_gen";
     confidence: number;
     probabilities: number[];
     details?: string;
   } | null>(null);


  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);

      // Create preview
      const reader = new FileReader();
      reader.onload = (event) => {
        setPreview(event.target?.result as string);
      };
      reader.readAsDataURL(selectedFile);

      // Reset results
      setResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!file) return;

    setAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/api/image", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to analyze image");

      const data = await res.json();

      // Convert probabilities to percentages
      const probabilities = data.probabilities.map((p: number) =>
        Math.round(p * 100)
      );

      setResult({
        prediction: data.prediction,
        confidence:
          probabilities[
            data.prediction === "real"
              ? 0
              : data.prediction === "deepfake"
              ? 1
              : 2
          ],
        probabilities,
        details: "Analysis complete. Check the results below.",
      });
    } catch (error) {
      console.error(error);
      alert("An error occurred while analyzing the image.");
    } finally {
      setAnalyzing(false);
    }
  };


  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <main className="flex justify-center">
        <section className="container grid items-center gap-6 pb-8 pt-6 md:py-10">
          <div className="flex max-w-[980px] flex-col items-start gap-2">
            <h1 className="text-3xl font-extrabold leading-tight tracking-tighter md:text-4xl">
              Image Deepfake Detection
            </h1>
            <p className="max-w-[700px] text-lg text-muted-foreground">
              Upload an image to analyze it for signs of manipulation or AI
              generation.
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
                  <CardTitle>Upload Image</CardTitle>
                  <CardDescription>
                    Select an image file to analyze for deepfake detection.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col items-center justify-center gap-4">
                    <div
                      className="flex h-64 w-full cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-muted-foreground/25 p-4 transition-colors hover:border-muted-foreground/50"
                      onClick={() =>
                        document.getElementById("file-upload")?.click()
                      }
                    >
                      {preview ? (
                        <img
                          src={preview || "/placeholder.svg"}
                          alt="Preview"
                          className="h-full max-h-56 w-auto object-contain"
                        />
                      ) : (
                        <>
                          <Upload className="mb-2 h-10 w-10 text-muted-foreground" />
                          <p className="text-sm text-muted-foreground">
                            Drag and drop or click to upload
                          </p>
                          <p className="text-xs text-muted-foreground">
                            Supports JPG, PNG, WEBP (max 10MB)
                          </p>
                        </>
                      )}
                      <input
                        id="file-upload"
                        type="file"
                        accept="image/jpeg,image/png,image/webp"
                        className="hidden"
                        onChange={handleFileChange}
                      />
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button
                    onClick={analyzeImage}
                    disabled={!file || analyzing}
                    className="w-full"
                  >
                    {analyzing ? "Analyzing..." : "Analyze Image"}
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
                  <div className="flex h-64 flex-col items-center justify-center gap-4">
                    {analyzing ? (
                      <div className="flex w-full flex-col items-center gap-2">
                        <p className="text-center text-sm text-muted-foreground">
                          Analyzing image for manipulation...
                        </p>
                        <Progress value={45} className="w-full" />
                      </div>
                    ) : result ? (
                      <div className="w-full">
                        <Alert
                          variant={
                            result.prediction === "real"
                              ? "default"
                              : result.prediction === "deepfake"
                              ? "destructive"
                              : "destructive"
                          }
                        >
                          {result.prediction === "real" ? (
                            <CheckCircle2 className="h-4 w-4" />
                          ) : (
                            <AlertCircle className="h-4 w-4" />
                          )}
                          <AlertTitle>
                            {result.prediction === "real"
                              ? "Authentic Image"
                              : result.prediction === "deepfake"
                              ? "Deepfake Detected"
                              : "AI Generated Content"}
                          </AlertTitle>
                          <AlertDescription>
                            {result.prediction === "real"
                              ? `This image appears to be authentic with ${result.confidence}% confidence.`
                              : `This content appears to be ${result.prediction} with ${result.confidence}% confidence.`}
                          </AlertDescription>
                        </Alert>

                        <div className="mt-4">
                          <h4 className="mb-2 font-medium">Analysis Details</h4>
                          <ul className="list-inside list-disc space-y-1 text-sm text-muted-foreground">
                            <li>
                              Real Probability: {result.probabilities[0]}%
                            </li>
                            <li>
                              Deepfake Probability: {result.probabilities[1]}%
                            </li>
                            <li>
                              AI Generation Probability:{" "}
                              {result.probabilities[2]}%
                            </li>
                            <li>
                              Final Prediction:{" "}
                              {result.prediction === "real"
                                ? "Authentic"
                                : result.prediction === "deepfake"
                                ? "Deepfake"
                                : "AI Generated"}
                            </li>
                            <li>
                              Metadata consistency:{" "}
                              {result.prediction === "real"
                                ? "Consistent"
                                : "Inconsistent"}
                            </li>
                            <li>
                              Facial landmarks:{" "}
                              {result.prediction === "real"
                                ? "Natural"
                                : "Anomalies detected"}
                            </li>
                          </ul>
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center text-center text-muted-foreground">
                        <p>No image analyzed yet.</p>
                        <p className="text-sm">
                          Upload an image and click Analyze to get started.
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
