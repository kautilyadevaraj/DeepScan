import { NextResponse } from "next/server";

export const runtime = "edge"; // Optional: Use Edge Runtime for faster responses

export async function POST(request: Request) {
  try {
    const { url } = await request.json();

    if (!url) {
      return NextResponse.json({ error: "URL is required" }, { status: 400 });
    }

    // Call your Flask backend
    const flaskResponse = await fetch(
      "http://localhost:5000/predict_phishing",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      }
    );

    if (!flaskResponse.ok) {
      throw new Error("Failed to analyze URL");
    }

    const data = await flaskResponse.json();

    // Transform the response to match your frontend expectations
    const transformedResponse = {
      url: data.url,
      prediction: data.prediction.includes("Legitimate")
        ? "legitimate"
        : "phishing",
      confidence: calculateConfidence(data.features), // Implement this function
      probabilities: calculateProbabilities(data.features), // Implement this function
      details: "Analysis complete based on URL features",
      riskFactors: getRiskFactors(data.features), // Implement this function
      features: data.features,
    };

    return NextResponse.json(transformedResponse);
  } catch (error) {
    console.error("Phishing analysis error:", error);
    return NextResponse.json(
      { error: "Failed to analyze URL" },
      { status: 500 }
    );
  }
}

// Helper functions
function calculateConfidence(features: Record<string, number>): number {
  // Implement your confidence calculation logic
  // This is a simplified example - adjust based on your model
  const positiveIndicators = [
    features.SSLfinal_State === 1,
    features.Shortining_Service === 0,
    features.having_At_Symbol === 0,
  ].filter(Boolean).length;

  return Math.min(100, 70 + positiveIndicators * 10);
}

function calculateProbabilities(features: Record<string, number>): number[] {
  // Implement your probability calculation logic
  const confidence = calculateConfidence(features);
  return [
    confidence, // legitimate
    100 - confidence, // phishing
    0, // suspicious (not used in current model)
  ];
}

function getRiskFactors(features: Record<string, number>): string[] {
  const risks = [];

  if (features.Shortining_Service === 1) {
    risks.push("Uses URL shortening service");
  }
  if (features.having_At_Symbol === 1) {
    risks.push("Contains @ symbol in URL");
  }
  if (features.Prefix_Suffix === 1) {
    risks.push("Uses hyphens in domain (potential spoofing)");
  }
  if (features.SSLfinal_State !== 1) {
    risks.push("Missing or invalid SSL certificate");
  }
  if (features.HTTPS_token === 1) {
    risks.push("Contains 'https' in domain name (potential spoofing)");
  }

  return risks.length > 0 ? risks : [];
}
