"use server";

import { createClient } from "@/lib/supabase/server";
import { redirect } from "next/navigation";

export async function createProfile(formData: FormData) {
  const fullName = formData.get("full_name") as string;
  if (!fullName?.trim()) return;

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  await supabase
    .from("profiles")
    .insert({ id: user.id, full_name: fullName.trim() });

  redirect("/");
}

export async function updateProfile(formData: FormData) {
  const fullName = formData.get("full_name") as string;
  if (!fullName?.trim()) return;

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  await supabase
    .from("profiles")
    .update({ full_name: fullName.trim(), updated_at: new Date().toISOString() })
    .eq("id", user.id);

  redirect("/profile");
}
