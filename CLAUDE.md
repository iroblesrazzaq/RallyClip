# RallyClip — agent guide

## PR review loop (Greptile)

This repo uses **Greptile** for automated PR review. After opening or updating a PR,
run this loop until the PR is clean. **Do not merge — the user merges.**

1. **Pull the review** — summary (with `Confidence Score: N/5`) + inline comments:
   - Summary: `gh api repos/<owner>/<repo>/issues/<pr>/comments`
   - Inline: `gh api repos/<owner>/<repo>/pulls/<pr>/comments`
2. **Critique each claim** — legitimate vs not. Do **not** blindly apply suggestions; verify
   against the actual code. Greptile's literal patch can be wrong even when the underlying
   concern is valid (e.g. it may reuse a variable that's still needed elsewhere). State a
   verdict + reasoning for each comment.
3. **Fix the legitimate claims** with the smallest correct change. Skip invalid ones, noting why.
4. **Re-trigger** — commit + push, then `gh pr comment <pr> --body "@greptileai"` on the new commit.
5. **Verify 5/5** — wait ~1–3 min for the re-review and confirm **Confidence Score: 5/5**.
6. Repeat 1–5 until 5/5 (or it was already 5/5 with no edits needed).
7. **Hand back** to the user with the 5/5 result for them to merge.
