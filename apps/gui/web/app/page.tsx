import { signOut } from "@/app/actions/auth";

export default function Home() {
  return (
    <div className="flex flex-col flex-1 bg-zinc-50 font-sans dark:bg-black">
      <header className="flex items-center justify-between border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
        <h1 className="text-lg font-semibold text-[var(--foreground)]">
          RallyClip
        </h1>
        <form action={signOut}>
          <button
            type="submit"
            className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
          >
            Sign out
          </button>
        </form>
      </header>

      <main className="flex flex-1 items-center justify-center px-6">
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
