import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { createProfile } from "@/app/actions/profile";

type SearchParamsValue = string | string[] | undefined;

function firstParam(value: SearchParamsValue) {
  return Array.isArray(value) ? value[0] : value;
}

export default async function OnboardingPage({
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
  if (!user) redirect("/login");

  // If they already have a profile, skip onboarding
  const { data: profile } = await supabase
    .from("profiles")
    .select("id")
    .eq("id", user.id)
    .single();
  if (profile) redirect("/profile");

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--background)]">
      <div className="w-full max-w-sm rounded-lg border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h1 className="mb-2 text-center text-2xl font-semibold text-[var(--foreground)]">
          Welcome to RallyClip
        </h1>
        <p className="mb-6 text-center text-sm text-zinc-500">
          Add the name you&apos;d like to use in RallyClip.
        </p>

        {error ? <p className="mb-4 text-sm text-red-500">{error}</p> : null}

        <form action={createProfile} className="flex flex-col gap-4">
          <input
            type="text"
            name="full_name"
            placeholder="Your name"
            required
            autoFocus
            className="rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-500 dark:border-zinc-700"
          />
          <button
            type="submit"
            className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            Continue
          </button>
        </form>
      </div>
    </div>
  );
}
