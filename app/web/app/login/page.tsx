"use client";

import Link from "next/link";
import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-[var(--background)]">
          <div className="w-full max-w-sm rounded-lg border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <h1 className="mb-6 text-center text-2xl font-semibold text-[var(--foreground)]">
              Sign in to RallyClip
            </h1>
          </div>
        </div>
      }
    >
      <LoginForm />
    </Suspense>
  );
}

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const supabase = createClient();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSignUp, setIsSignUp] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const callbackError = searchParams.get("error");
  const callbackStatus = searchParams.get("status");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setMessage(null);
    setLoading(true);

    if (isSignUp) {
      const { error } = await supabase.auth.signUp({ email, password });
      if (error) {
        setError(error.message);
      } else {
        setMessage("Check your email for a confirmation link.");
      }
    } else {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      if (error) {
        setError(error.message);
      } else {
        router.push("/");
        router.refresh();
      }
    }

    setLoading(false);
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--background)]">
      <div className="w-full max-w-sm rounded-lg border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h1 className="mb-6 text-center text-2xl font-semibold text-[var(--foreground)]">
          {isSignUp ? "Create account" : "Sign in to RallyClip"}
        </h1>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-500 dark:border-zinc-700"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={6}
            className="rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-500 dark:border-zinc-700"
          />

          {(error || callbackError) && (
            <p className="text-sm text-red-500">{error || callbackError}</p>
          )}
          {(message || callbackStatus) && (
            <p className="text-sm text-green-600 dark:text-green-400">
              {message || callbackStatus}
            </p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            {loading
              ? "Loading..."
              : isSignUp
                ? "Sign up"
                : "Sign in"}
          </button>
        </form>

        <p className="mt-4 text-center text-sm text-zinc-500">
          {isSignUp ? "Already have an account?" : "Don't have an account?"}{" "}
          <button
            onClick={() => {
              setIsSignUp(!isSignUp);
              setError(null);
              setMessage(null);
            }}
            className="underline hover:text-[var(--foreground)]"
          >
            {isSignUp ? "Sign in" : "Sign up"}
          </button>
        </p>

        {!isSignUp ? (
          <p className="mt-2 text-center text-sm text-zinc-500">
            <Link
              href="/forgot-password"
              className="underline hover:text-[var(--foreground)]"
            >
              Forgot your password?
            </Link>
          </p>
        ) : null}
      </div>
    </div>
  );
}
