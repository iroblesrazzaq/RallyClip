import { redirect } from "next/navigation";
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
  if (!user) redirect("/login");

  const { data: profile } = await supabase
    .from("profiles")
    .select("full_name")
    .eq("id", user.id)
    .single();
  const fullName = profile?.full_name?.trim() || null;

  return (
    <div className="flex flex-col flex-1 bg-zinc-50 font-sans dark:bg-black">
      <header className="flex items-center justify-between border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
        <h1 className="text-lg font-semibold text-[var(--foreground)]">
          RallyClip
        </h1>
        <div className="flex items-center gap-4">
          <a
            href="/profile"
            className="text-sm text-zinc-600 hover:text-[var(--foreground)] dark:text-zinc-400"
          >
            {fullName || user.email}
          </a>
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
            <h2 className="text-lg font-semibold text-[var(--foreground)]">
              Complete your profile
            </h2>
            <p className="mt-2 text-sm text-zinc-500">
              Add your name so your account is fully set up. You can do it now
              or anytime later from your profile page.
            </p>
            <div className="mt-4 flex gap-3">
              <a
                href="/onboarding"
                className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
              >
                Finish setup
              </a>
              <a
                href="/profile"
                className="rounded-md border border-zinc-300 px-4 py-2 text-sm text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                Open profile
              </a>
            </div>
          </div>
        ) : null}

        <div className="max-w-lg text-center">
          <h2 className="text-2xl font-semibold text-[var(--foreground)]">
            Rally Clip Detection
          </h2>
          <p className="mt-3 text-zinc-500">
            Upload a badminton video to automatically detect and extract rally
            clips. Inference backend coming soon.
          </p>
        </div>
      </main>
    </div>
  );
}
