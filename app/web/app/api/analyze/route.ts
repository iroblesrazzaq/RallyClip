import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";
import { checkRateLimit, logUsage } from "@/lib/rate-limit";

// Stub — replace with real ML backend call (Modal, Replicate, etc.)
export async function POST(req: NextRequest) {
  const supabase = await createClient();
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser();

  if (!user || error) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { allowed, remaining } = await checkRateLimit(supabase, user.id);
  if (!allowed) {
    return NextResponse.json(
      { error: "Daily limit exceeded", limit: 20 },
      { status: 429, headers: { "X-RateLimit-Remaining": "0" } }
    );
  }

  await logUsage(supabase, user.id);

  const body = await req.json().catch(() => ({}));

  // Mock response shape — update when real inference is wired up
  return NextResponse.json({
    status: "ok",
    clips: [
      { id: "clip_1", label: "rally", confidence: 0.91, start_frame: 0, end_frame: 150 },
      { id: "clip_2", label: "rally", confidence: 0.85, start_frame: 200, end_frame: 400 },
    ],
    remaining,
    input: body,
  });
}
