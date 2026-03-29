import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { signOut } from "@/app/actions/auth";
import { updateProfile } from "@/app/actions/profile";

export default async function ProfilePage() {
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
  if (!profile) redirect("/onboarding");

  return (
    <div className="flex flex-col flex-1 bg-zinc-50 font-sans dark:bg-black">
      <header className="flex items-center justify-between border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
        <a
          href="/"
          className="text-lg font-semibold text-[var(--foreground)] hover:opacity-80"
        >
          RallyClip
        </a>
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
            Profile
          </h2>

          <div className="mb-8 rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
            <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-zinc-400">
              Email
            </label>
            <p className="text-sm text-[var(--foreground)]">{user.email}</p>
          </div>

          <form
            action={updateProfile}
            className="rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900"
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
              defaultValue={profile.full_name}
              required
              className="mb-4 w-full rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-500 dark:border-zinc-700"
            />
            <button
              type="submit"
              className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              Save
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
