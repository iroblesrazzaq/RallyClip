import Link from "next/link";
import { requestPasswordReset } from "@/app/actions/auth";

type SearchParamsValue = string | string[] | undefined;

function firstParam(value: SearchParamsValue) {
  return Array.isArray(value) ? value[0] : value;
}

export default async function ForgotPasswordPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, SearchParamsValue>>;
}) {
  const params = await searchParams;
  const error = firstParam(params.error);
  const status = firstParam(params.status);

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--background)]">
      <div className="w-full max-w-sm rounded-lg border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h1 className="mb-2 text-center text-2xl font-semibold text-[var(--foreground)]">
          Reset your password
        </h1>
        <p className="mb-6 text-center text-sm text-zinc-500">
          Enter your email and we&apos;ll send you a recovery link.
        </p>

        <form action={requestPasswordReset} className="flex flex-col gap-4">
          <input
            type="email"
            name="email"
            placeholder="Email"
            required
            className="rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-500 dark:border-zinc-700"
          />

          {error ? <p className="text-sm text-red-500">{error}</p> : null}
          {status ? (
            <p className="text-sm text-green-600 dark:text-green-400">
              {status}
            </p>
          ) : null}

          <button
            type="submit"
            className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            Send reset link
          </button>
        </form>

        <p className="mt-4 text-center text-sm text-zinc-500">
          <Link href="/login" className="underline hover:text-[var(--foreground)]">
            Back to sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
