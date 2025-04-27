// pages/api/predict_audio.ts

import type { NextApiRequest, NextApiResponse } from "next";

// Define the structure of the audio prediction response
interface PredictionResponse {
  final_prediction: string;
  final_confidence: number;
  model_prediction: string;
  model_confidence: number;
  heuristic_prediction: string;
  heuristic_confidence: number;
  heuristic_scores: number[];
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method === "POST") {
    try {
      // Get the audio file from the request body
      const audioData = req.body.audio; // Assuming it's base64 encoded audio data

      if (!audioData) {
        return res.status(400).json({ error: "No audio provided" });
      }

      // Prepare the payload to send to the Flask backend
      const payload = {
        image: audioData, // Send the audio data as base64
      };

      // Fetch the prediction result from the Flask app
      const response = await fetch("http://172.16.239.97:5000/predict_audio", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        return res
          .status(response.status)
          .json({ error: "Failed to connect to Flask backend" });
      }

      const modelResult = await response.json();

      // Assuming you also have a heuristic result to combine (this could be from another API)
      const heuristicResult = {
        prediction: "fake", // Placeholder, replace with actual heuristic prediction logic
        confidence: 0.8, // Placeholder, replace with actual confidence score
        scores: [0.1, 0.2, 0.3], // Placeholder scores, replace with actual heuristic scores
      };

      // Combine results based on the provided logic
      let finalPrediction = "";
      let finalConfidence = 0;

      if (modelResult.prediction === "real") {
        const combinedProb =
          modelResult.real_prob * 0.7 + // model_weight (placeholder)
          (1 - heuristicResult.confidence) * 0.3; // heuristic_weight (placeholder)
        finalPrediction = combinedProb > 0.5 ? "real" : "fake";
        finalConfidence = Math.max(combinedProb, 1 - combinedProb);
      } else {
        const combinedProb =
          modelResult.fake_prob * 0.7 + // model_weight (placeholder)
          heuristicResult.confidence * 0.3; // heuristic_weight (placeholder)
        finalPrediction = combinedProb > 0.5 ? "fake" : "real";
        finalConfidence = Math.max(combinedProb, 1 - combinedProb);
      }

      // Return the final result
      return res.status(200).json({
        final_prediction: finalPrediction,
        final_confidence: finalConfidence,
        model_prediction: modelResult.prediction,
        model_confidence: Math.max(
          modelResult.real_prob,
          modelResult.fake_prob
        ),
        heuristic_prediction: heuristicResult.prediction,
        heuristic_confidence: heuristicResult.confidence,
        heuristic_scores: heuristicResult.scores,
      });
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: "Internal server error" });
    }
  } else {
    res.status(405).json({ error: "Method Not Allowed" });
  }
}
