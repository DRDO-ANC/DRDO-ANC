# Contributing

## Rules
- `main` is protected. Nobody pushes directly. Everything goes through a Pull Request reviewed and merged by the lead (P1).
- One branch per task: `<area>/<short-name>`, e.g. `dsp/nlms-filter`, `data/snr-mixer`, `rt/loopback-test`, `backend/results-api`.
- Small PRs (one task, ideally < 400 lines). Link the issue: `Closes #12`.
- Commit messages: `feat: implement NLMS adaptive filter`, `fix:`, `docs:`, `test:`, `chore:`. No `update` / `final2`.
- Never commit datasets, audio, checkpoints, or secrets. Share data via a drive link noted in `docs/datasets.md`.
- Every number in a doc or PR must come from a logged experiment (config + result file in `experiments/`).

## Workflow
```bash
git checkout main && git pull
git checkout -b dsp/nlms-filter
# work, commit
git push -u origin dsp/nlms-filter
# open PR on GitHub -> fill the template -> request review from the lead
```
If `main` moved: `git fetch && git rebase origin/main`, then `git push --force-with-lease`.

## Review checklist (lead)
- Matches `docs/INTERFACES.md` (sample rate, shapes, config keys)?
- Has a test or a runnable script, and says how it was run?
- CI green? No large files? Folder ownership respected?
- Squash-merge with a clean message.
