import { SupabaseClient } from "@supabase/supabase-js";

const DAILY_LIMIT = 20;

export async function checkRateLimit(
  supabase: SupabaseClient,
  userId: string
) {
  const today = new Date().toISOString().split("T")[0];

  const { count, error } = await supabase
    .from("usage_log")
    .select("*", { count: "exact", head: true })
    .eq("user_id", userId)
    .gte("created_at", `${today}T00:00:00.000Z`);

  if (error) {
    console.error("Rate limit check failed:", error);
    return { allowed: true, remaining: DAILY_LIMIT };
  }

  const used = count ?? 0;
  return {
    allowed: used < DAILY_LIMIT,
    remaining: Math.max(0, DAILY_LIMIT - used),
  };
}

export async function logUsage(supabase: SupabaseClient, userId: string) {
  await supabase.from("usage_log").insert({ user_id: userId });
}
