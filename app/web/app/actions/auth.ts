"use server";

import { createClient } from "@/lib/supabase/server";
import { headers } from "next/headers";
import { redirect } from "next/navigation";

function encodeMessage(message: string) {
  return encodeURIComponent(message);
}

async function appUrl(path: string) {
  const hdrs = await headers();
  const origin =
    process.env.NEXT_PUBLIC_SITE_URL ||
    process.env.NEXT_PUBLIC_VERCEL_URL?.replace(/^/, "https://") ||
    hdrs.get("origin") ||
    "http://localhost:3000";
  return new URL(path, origin).toString();
}

export async function signOut() {
  const supabase = await createClient();
  await supabase.auth.signOut();
  redirect("/login");
}

export async function requestPasswordReset(formData: FormData) {
  const email = String(formData.get("email") || "").trim();
  if (!email) {
    redirect("/forgot-password?error=Please%20enter%20your%20email.");
  }

  const supabase = await createClient();
  const redirectTo = await appUrl("/auth/callback?next=/reset-password");
  const { error } = await supabase.auth.resetPasswordForEmail(email, { redirectTo });

  if (error) {
    redirect(`/forgot-password?error=${encodeMessage(error.message)}`);
  }

  redirect(
    "/forgot-password?status=Check%20your%20email%20for%20a%20password%20reset%20link."
  );
}

export async function updatePassword(formData: FormData) {
  const password = String(formData.get("password") || "");
  const confirmPassword = String(formData.get("confirm_password") || "");

  if (!password || password.length < 6) {
    redirect("/reset-password?error=Password%20must%20be%20at%20least%206%20characters.");
  }
  if (password !== confirmPassword) {
    redirect("/reset-password?error=Passwords%20do%20not%20match.");
  }

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) {
    redirect("/reset-password?error=Your%20reset%20session%20is%20invalid%20or%20expired.");
  }

  const { error } = await supabase.auth.updateUser({ password });
  if (error) {
    redirect(`/reset-password?error=${encodeMessage(error.message)}`);
  }

  redirect("/profile?status=Password%20updated%20successfully.");
}

export async function requestEmailChange(formData: FormData) {
  const email = String(formData.get("email") || "").trim();
  if (!email) {
    redirect("/profile?error=Please%20enter%20a%20new%20email%20address.");
  }

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) {
    redirect("/login");
  }

  const emailRedirectTo = await appUrl("/auth/callback?next=/profile?status=email-confirmed");
  const { error } = await supabase.auth.updateUser(
    { email },
    { emailRedirectTo }
  );

  if (error) {
    redirect(`/profile?error=${encodeMessage(error.message)}`);
  }

  redirect(
    "/profile?status=Verification%20emails%20sent.%20Confirm%20the%20change%20from%20your%20inbox."
  );
}
