"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { AlertCircle, CheckCircle2, Globe } from "lucide-react";
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
import { Input } from "@/components/ui/input";

type AnalysisResult = {
  prediction: "legitimate" | "phishing" | "suspicious";
  confidence: number;
  probabilities: number[];
  details?: string;
  riskFactors?: string[];
  features?: Record<string, number>;
  url?: string;
};

export default function PhishingDetection() {
  const [url, setUrl] = useState<string>("");
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUrl(e.target.value);
    setResult(null);
    setError(null);
  };

  const analyzeWebsite = async () => {
    if (!url) return;

    setAnalyzing(true);
    setError(null);

    try {
      // Validate URL format
      if (!url.match(/^https?:\/\//i)) {
        throw new Error("URL must start with http:// or https://");
      }

      const response = await fetch("/api/phishing", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Analysis error:", err);
      setError(err instanceof Error ? err.message : "Failed to analyze URL");
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
              Phishing Website Detection
            </h1>
            <p className="max-w-[700px] text-lg text-muted-foreground">
              Enter a URL to analyze it for signs of phishing or fraudulent
              activity.
            </p>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="grid gap-6 md:grid-cols-2">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card className="h-full">
                <CardHeader>
                  <CardTitle>Enter Website URL</CardTitle>
                  <CardDescription>
                    Provide a website URL to analyze for phishing detection.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col items-center justify-center gap-4">
                    <div className="w-full space-y-4">
                      <div className="flex items-center space-x-2">
                        <Input
                          type="url"
                          placeholder="https://example.com"
                          value={url}
                          onChange={handleUrlChange}
                          className="flex-1"
                        />
                      </div>

                      <div className="flex h-40 w-full cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-muted-foreground/25 p-4">
                        <Globe className="mb-2 h-10 w-10 text-muted-foreground" />
                        <p className="text-sm text-muted-foreground">
                          {url
                            ? "Click Analyze to check this website"
                            : "Enter a URL above to analyze"}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          We'll check for phishing indicators and security risks
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button
                    onClick={analyzeWebsite}
                    disabled={!url || analyzing}
                    className="w-full"
                  >
                    {analyzing ? "Analyzing..." : "Analyze Website"}
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
                          Analyzing website for phishing indicators...
                        </p>
                        <Progress value={45} className="w-full" />
                      </div>
                    ) : result ? (
                      <div className="w-full space-y-4">
                        <Alert
                          variant={
                            result.prediction === "legitimate"
                              ? "default"
                              : "destructive"
                          }
                        >
                          {result.prediction === "legitimate" ? (
                            <CheckCircle2 className="h-4 w-4" />
                          ) : (
                            <AlertCircle className="h-4 w-4" />
                          )}
                          <AlertTitle className="capitalize">
                            {result.prediction} Website
                          </AlertTitle>
                          <AlertDescription>
                            Confidence: {result.confidence}% - {result.details}
                          </AlertDescription>
                        </Alert>

                        <div className="space-y-2">
                          <h4 className="font-medium">Technical Analysis</h4>
                          <div className="grid grid-cols-2 gap-2 text-sm">
                            {result.features &&
                              Object.entries(result.features).map(
                                ([key, value]) => (
                                  <div
                                    key={key}
                                    className="flex items-center justify-between"
                                  >
                                    <span className="text-muted-foreground">
                                      {key.replace(/_/g, " ")}:
                                    </span>
                                    <span>{value}</span>
                                  </div>
                                )
                              )}
                          </div>
                        </div>

                        {result.riskFactors &&
                          result.riskFactors.length > 0 && (
                            <div className="space-y-2">
                              <h4 className="font-medium">Risk Factors</h4>
                              <ul className="list-inside list-disc space-y-1 text-sm text-muted-foreground">
                                {result.riskFactors.map((factor, index) => (
                                  <li key={index}>{factor}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                      </div>
                    ) : (
                      <div className="flex flex-col items-center text-center text-muted-foreground">
                        <p>No website analyzed yet.</p>
                        <p className="text-sm">
                          Enter a URL and click Analyze to get started.
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
