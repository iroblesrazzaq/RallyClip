import Link from "next/link";
import { createClient } from "@/lib/supabase/server";
import { updatePassword } from "@/app/actions/auth";

type SearchParamsValue = string | string[] | undefined;

function firstParam(value: SearchParamsValue) {
  return Array.isArray(value) ? value[0] : value;
}

export default async function ResetPasswordPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, SearchParamsValue>>;
}) {
  const params = await searchParams;
  const error = firstParam(params.error);
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  const hasRecoverySession = Boolean(user);

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--background)]">
      <div className="w-full max-w-sm rounded-lg border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h1 className="mb-2 text-center text-2xl font-semibold text-[var(--foreground)]">
          Choose a new password
        </h1>
        <p className="mb-6 text-center text-sm text-zinc-500">
          {hasRecoverySession
            ? "Your recovery session is active. Enter a new password below."
            : "This recovery link is invalid or expired. Request a new one."}
        </p>

        {error ? <p className="mb-4 text-sm text-red-500">{error}</p> : null}

        {hasRecoverySession ? (
          <form action={updatePassword} className="flex flex-col gap-4">
            <input
              type="password"
              name="password"
              placeholder="New password"
              required
              minLength={6}
              className="rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-500 dark:border-zinc-700"
            />
            <input
              type="password"
              name="confirm_password"
              placeholder="Confirm new password"
              required
              minLength={6}
              className="rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-500 dark:border-zinc-700"
            />

            <button
              type="submit"
              className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              Update password
            </button>
          </form>
        ) : (
          <Link
            href="/forgot-password"
            className="inline-flex w-full justify-center rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            Request a new reset link
          </Link>
        )}

        <p className="mt-4 text-center text-sm text-zinc-500">
          <Link href="/login" className="underline hover:text-[var(--foreground)]">
            Back to sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
