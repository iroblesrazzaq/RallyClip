"use server";

import { createClient } from "@/lib/supabase/server";
import { redirect } from "next/navigation";

function encodeMessage(message: string) {
  return encodeURIComponent(message);
}

export async function createProfile(formData: FormData) {
  const fullName = formData.get("full_name") as string;
  if (!fullName?.trim()) {
    redirect("/onboarding?error=Please%20enter%20your%20name.");
  }

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const { error } = await supabase
    .from("profiles")
    .insert({ id: user.id, full_name: fullName.trim() });
  if (error) {
    redirect(`/onboarding?error=${encodeMessage(error.message)}`);
  }

  redirect("/?status=Profile%20completed.");
}

export async function updateProfile(formData: FormData) {
  const fullName = formData.get("full_name") as string;
  if (!fullName?.trim()) {
    redirect("/profile?error=Please%20enter%20your%20name.");
  }

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const { error } = await supabase
    .from("profiles")
    .upsert(
      { id: user.id, full_name: fullName.trim(), updated_at: new Date().toISOString() },
      { onConflict: "id" }
    );
  if (error) {
    redirect(`/profile?error=${encodeMessage(error.message)}`);
  }

  redirect("/profile?status=Profile%20updated.");
}
