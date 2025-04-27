import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // Read the file and convert to base64
    const arrayBuffer = await file.arrayBuffer();
    const base64String = Buffer.from(arrayBuffer).toString("base64");

    // Send to your backend API
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: base64String }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json({ error: errorText }, { status: 500 });
    }

    const result = await response.json();

    return NextResponse.json(result);
  } catch (error) {
    console.error("[ANALYZE_ERROR]", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
