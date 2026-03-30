import Link from "next/link";
import { createClient } from "@/lib/supabase/server";
import { signOut } from "@/app/actions/auth";

type SearchParamsValue = string | string[] | undefined;

function firstParam(value: SearchParamsValue) {
  return Array.isArray(value) ? value[0] : value;
}

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<Record<string, SearchParamsValue>>;
}) {
  const params = await searchParams;
  const status = firstParam(params.status);
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return <LandingPage />;
  }

  const { data: profile } = await supabase
    .from("profiles")
    .select("full_name")
    .eq("id", user.id)
    .single();
  const fullName = profile?.full_name?.trim() || null;

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 font-sans text-[var(--foreground)] dark:bg-black">
      <header className="flex items-center justify-between border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
        <h1 className="text-lg font-semibold">RallyClip</h1>
        <div className="flex items-center gap-4">
          <Link
            href="/profile"
            className="text-sm text-zinc-600 hover:text-[var(--foreground)] dark:text-zinc-400"
          >
            {fullName || user.email}
          </Link>
          <form action={signOut}>
            <button
              type="submit"
              className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
            >
              Sign out
            </button>
          </form>
        </div>
      </header>

      <main className="flex flex-1 flex-col items-center justify-center gap-6 px-6 py-10">
        {status ? (
          <div className="w-full max-w-2xl rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-800 dark:border-green-900 dark:bg-green-950/40 dark:text-green-200">
            {status}
          </div>
        ) : null}

        {!fullName ? (
          <div className="w-full max-w-2xl rounded-lg border border-zinc-200 bg-white p-6 text-left shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <h2 className="text-lg font-semibold">Complete your profile</h2>
            <p className="mt-2 text-sm text-zinc-500">
              Add your name so your account is fully set up. You can do it now
              or anytime later from your profile page.
            </p>
            <div className="mt-4 flex gap-3">
              <Link
                href="/onboarding"
                className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
              >
                Finish setup
              </Link>
              <Link
                href="/profile"
                className="rounded-md border border-zinc-300 px-4 py-2 text-sm text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                Open profile
              </Link>
            </div>
          </div>
        ) : null}

        <div className="w-full max-w-2xl rounded-2xl border border-zinc-200 bg-white p-8 text-left shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-2xl font-semibold">
            Rally segmentation for real match footage
          </h2>
          <p className="mt-4 text-zinc-500">
            RallyClip extracts only the points from a full tennis match video,
            removes the dead time between rallies, and returns a condensed video
            plus optional CSV timestamps. Hosted analysis is coming — your
            account and profile are live now.
          </p>
        </div>
      </main>
    </div>
  );
}

function LandingPage() {
  return (
    <div className="min-h-screen bg-zinc-50 font-sans text-[var(--foreground)] dark:bg-black">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6">
        <div>
          <p className="text-sm text-zinc-500">Open-source rally segmentation</p>
          <h1 className="text-lg font-semibold">RallyClip</h1>
        </div>
        <div className="flex items-center gap-3">
          <Link
            href="/login"
            className="rounded-md border border-zinc-300 px-4 py-2 text-sm text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            Log in
          </Link>
          <Link
            href="/login?mode=signup"
            className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            Sign up
          </Link>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-col gap-16 px-6 pb-20 pt-8">
        <section className="grid gap-10 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">
          <div>
            <p className="text-sm font-medium uppercase tracking-[0.25em] text-zinc-500">
              Free and open source
            </p>
            <h2 className="mt-4 text-5xl font-semibold leading-tight text-balance">
              Extract only the points from a full tennis match video.
            </h2>
            <p className="mt-6 max-w-2xl text-lg text-zinc-600 dark:text-zinc-400">
              Removes the dead time between rallies. Returns a condensed video
              and optional CSV timestamps.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <Link
                href="/login?mode=signup"
                className="rounded-md bg-zinc-900 px-5 py-3 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
              >
                Create account
              </Link>
              <Link
                href="/login"
                className="rounded-md border border-zinc-300 px-5 py-3 text-sm text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                Log in
              </Link>
            </div>
          </div>

          <div className="rounded-3xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="grid gap-4 sm:grid-cols-2">
              <StatCard label="What goes in" value="Full match video" />
              <StatCard
                label="What comes out"
                value="Point-only cut + optional CSV"
              />
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-2xl">
          <ul className="space-y-3 text-zinc-600 dark:text-zinc-400">
            <li>Free match segmentation without a paid subscription.</li>
            <li>Runs locally on your own machine. No upload required.</li>
            <li>Open source — CLI, model, and training pipeline.</li>
          </ul>
        </section>
      </main>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-zinc-200 p-4 dark:border-zinc-800">
      <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">{label}</p>
      <p className="mt-3 text-lg font-medium">{value}</p>
    </div>
  );
}

