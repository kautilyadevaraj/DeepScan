// app/api/audio/route.ts

import { NextResponse } from "next/server";

interface PredictionResponse {
  final_prediction: string;
  final_confidence: number;
  model_prediction: string;
  model_confidence: number;
  heuristic_prediction: string;
  heuristic_confidence: number;
  heuristic_scores: number[];
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const audioData = body.audio;

    if (!audioData) {
      return NextResponse.json({ error: "No audio provided" }, { status: 400 });
    }

    const payload = {
      image: audioData,
    };

    const response = await fetch("http://172.16.239.97:5000/predict_audio", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "Failed to connect to Flask backend" },
        { status: response.status }
      );
    }

    const modelResult = await response.json();

    const heuristicResult = {
      prediction: "fake",
      confidence: 0.8,
      scores: [0.1, 0.2, 0.3],
    };

    let finalPrediction = "";
    let finalConfidence = 0;

    if (modelResult.prediction === "real") {
      const combinedProb =
        modelResult.real_prob * 0.7 + (1 - heuristicResult.confidence) * 0.3;
      finalPrediction = combinedProb > 0.5 ? "real" : "fake";
      finalConfidence = Math.max(combinedProb, 1 - combinedProb);
    } else {
      const combinedProb =
        modelResult.fake_prob * 0.7 + heuristicResult.confidence * 0.3;
      finalPrediction = combinedProb > 0.5 ? "fake" : "real";
      finalConfidence = Math.max(combinedProb, 1 - combinedProb);
    }

    return NextResponse.json<PredictionResponse>({
      final_prediction: finalPrediction,
      final_confidence: finalConfidence,
      model_prediction: modelResult.prediction,
      model_confidence: Math.max(modelResult.real_prob, modelResult.fake_prob),
      heuristic_prediction: heuristicResult.prediction,
      heuristic_confidence: heuristicResult.confidence,
      heuristic_scores: heuristicResult.scores,
    });
  } catch (error) {
    console.error(error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
