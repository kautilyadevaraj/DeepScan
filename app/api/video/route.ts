import { NextRequest, NextResponse } from "next/server";
import FormData from "form-data";
import fs from "fs";
import path from "path";

export const config = {
  api: {
    bodyParser: false, // Disable default body parsing
  },
};

export async function POST(req: NextRequest) {
  try {
    // Parse the incoming form data
    const formData = await req.formData();
    const audioFile = formData.get("audio") as File | null;

    if (!audioFile) {
      return NextResponse.json(
        { error: "No audio file provided" },
        { status: 400 }
      );
    }

    // Convert the File to a Buffer
    const audioBuffer = Buffer.from(await audioFile.arrayBuffer());

    // Create FormData to send to Flask backend
    const flaskFormData = new FormData();
    flaskFormData.append("audio", audioBuffer, {
      filename: audioFile.name,
      contentType: audioFile.type,
    });

    // Send to Flask backend
    const flaskResponse = await fetch("http://127.0.0.1:5000/predict_audio", {
      method: "POST",
      body: flaskFormData as any, // Type assertion needed
      headers: {
        ...flaskFormData.getHeaders(),
      },
    });

    if (!flaskResponse.ok) {
      const errorData = await flaskResponse.json();
      return NextResponse.json(
        { error: errorData.error || "Audio prediction failed" },
        { status: flaskResponse.status }
      );
    }

    const result = await flaskResponse.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error("Error in audio prediction:", error);
    return NextResponse.json(
      { error: "Internal server error during audio processing" },
      { status: 500 }
    );
  }
}
