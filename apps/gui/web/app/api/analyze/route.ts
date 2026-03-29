import { NextRequest, NextResponse } from "next/server";

// Stub — replace with real ML backend call (Modal, Replicate, etc.)
export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));

  // Mock response shape — update when real inference is wired up
  return NextResponse.json({
    status: "ok",
    clips: [
      { id: "clip_1", label: "rally", confidence: 0.91, start_frame: 0, end_frame: 150 },
      { id: "clip_2", label: "rally", confidence: 0.85, start_frame: 200, end_frame: 400 },
    ],
    input: body,
  });
}
