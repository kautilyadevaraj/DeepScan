import { NextResponse } from "next/server";
import FormData from "form-data";

interface PredictionResponse {
  final_prediction: string;
  final_confidence: number;
  model_prediction: string;
  model_confidence: number;
  heuristic_prediction: string;
  heuristic_confidence: number;
  heuristic_scores: { [key: string]: number };
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const audioData = body.audio;

    if (!audioData) {
      return NextResponse.json({ error: "No audio provided" }, { status: 400 });
    }

    // Convert base64 audio data to Buffer
    const buffer = Buffer.from(audioData, "base64");

    // Create FormData and append the audio file
    const form = new FormData();
    form.append("audio", buffer, { filename: "audio.wav" });

    // Send request to Flask backend
    const response = await fetch("http://127.0.0.1:5000/predict_audio", {
      method: "POST",
      headers: form.getHeaders(),
      body: form.getBuffer(),
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "Failed to connect to Flask backend" },
        { status: response.status }
      );
    }

    // Parse the backend's complete response
    const result = await response.json();

    // Return the backend's result directly
    return NextResponse.json<PredictionResponse>({
      final_prediction: result.final_prediction,
      final_confidence: result.final_confidence,
      model_prediction: result.model_prediction,
      model_confidence: result.model_confidence,
      heuristic_prediction: result.heuristic_prediction,
      heuristic_confidence: result.heuristic_confidence,
      heuristic_scores: result.heuristic_scores,
    });
  } catch (error) {
    console.error(error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
