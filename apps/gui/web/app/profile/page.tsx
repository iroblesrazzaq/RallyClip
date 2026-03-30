import Link from "next/link";
import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import {
  requestEmailChange,
  signOut,
  updatePassword,
} from "@/app/actions/auth";
import { updateProfile } from "@/app/actions/profile";

type SearchParamsValue = string | string[] | undefined;

function firstParam(value: SearchParamsValue) {
  return Array.isArray(value) ? value[0] : value;
}

export default async function ProfilePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, SearchParamsValue>>;
}) {
  const params = await searchParams;
  const status = firstParam(params.status);
  const error = firstParam(params.error);
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const { data: profile } = await supabase
    .from("profiles")
    .select("full_name")
    .eq("id", user.id)
    .single();
  const fullName = profile?.full_name ?? "";

  return (
    <div className="flex flex-col flex-1 bg-zinc-50 font-sans dark:bg-black">
      <header className="flex items-center justify-between border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
        <Link
          href="/"
          className="text-lg font-semibold text-[var(--foreground)] hover:opacity-80"
        >
          RallyClip
        </Link>
        <form action={signOut}>
          <button
            type="submit"
            className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
          >
            Sign out
          </button>
        </form>
      </header>

      <main className="flex flex-1 justify-center px-6 py-12">
        <div className="w-full max-w-md">
          <h2 className="mb-8 text-2xl font-semibold text-[var(--foreground)]">
            Account settings
          </h2>

          {error ? (
            <div className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
              {error}
            </div>
          ) : null}
          {status ? (
            <div className="mb-6 rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-800 dark:border-green-900 dark:bg-green-950/40 dark:text-green-200">
              {status}
            </div>
          ) : null}

          <div className="mb-8 rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
            <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-zinc-400">
              Email
            </label>
            <p className="text-sm text-[var(--foreground)]">{user.email}</p>
          </div>

          <form
            action={updateProfile}
            className="mb-8 rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900"
          >
            <label
              htmlFor="full_name"
              className="mb-1 block text-xs font-medium uppercase tracking-wide text-zinc-400"
            >
              Name
            </label>
            <input
              id="full_name"
              type="text"
              name="full_name"
              defaultValue={fullName}
              required
              className="mb-4 w-full rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-500 dark:border-zinc-700"
            />
            <button
              type="submit"
              className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              Save name
            </button>
          </form>

          <form
            action={requestEmailChange}
            className="mb-8 rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900"
          >
            <label
              htmlFor="email"
              className="mb-1 block text-xs font-medium uppercase tracking-wide text-zinc-400"
            >
              Change email
            </label>
            <input
              id="email"
              type="email"
              name="email"
              defaultValue={user.email ?? ""}
              required
              className="mb-2 w-full rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-500 dark:border-zinc-700"
            />
            <p className="mb-4 text-xs text-zinc-500">
              We&apos;ll send verification instructions to confirm the new email
              address.
            </p>
            <button
              type="submit"
              className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              Change email
            </button>
          </form>

          <form
            action={updatePassword}
            className="rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900"
          >
            <h3 className="mb-1 text-sm font-medium text-[var(--foreground)]">
              Change password
            </h3>
            <p className="mb-4 text-xs text-zinc-500">
              Update your password directly while signed in.
            </p>
            <input
              type="password"
              name="password"
              placeholder="New password"
              required
              minLength={6}
              className="mb-3 w-full rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-500 dark:border-zinc-700"
            />
            <input
              type="password"
              name="confirm_password"
              placeholder="Confirm new password"
              required
              minLength={6}
              className="mb-4 w-full rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-500 dark:border-zinc-700"
            />
            <button
              type="submit"
              className="rounded-md border border-zinc-300 px-4 py-2 text-sm text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
            >
              Change password
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
