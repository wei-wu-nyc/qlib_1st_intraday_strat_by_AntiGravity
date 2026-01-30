---
description: Wrap up the current project stage: cleanup, document, commit, and PR.
---

1. **Analyze Progress**: 
   - Check `git status` and `git diff --stat`.
   - Read `task.md` and `experiments.md` (if they exist).
   - Generate a chronological summary of all key changes, bug fixes, and experiment results since the last commit.

2. **Update Documentation**:
   - Update `task.md` to mark completed phases as `[x]`.
   - Append new findings to `experiments.md` or `walkthrough.md`.

3. **Sync Artifacts**:
   - Copy all markdown artifacts from the brain directory to the repository root to ensure they are tracked.
   - Command: `cp <appDataDir>/brain/<conversation-id>/*.md .` (You need to find the correct path first).

4. **System Cleanup**:
   - Remove temporary system files and debris.
   - // turbo-all
   - `find . -name "qlib_*.txt" -delete`
   - `find . -name "__pycache__" -type d -exec rm -rf {} +`
   - `find . -name ".DS_Store" -delete`
   - `find . -name "._*" -delete`

5. **Git Operations**:
   - Propose a descriptive branch name based on the summary (e.g., `feat/quarterly-validation`).
   - Create the branch: `git checkout -b <branch_name>`
   - Stage all files: `git add .`
   - **Commit**: Create a detailed commit message using the summary from Step 1.
   - **Pull Request**: Create a PR with the summary as the body.
   - Command: `gh pr create --title "<Title>" --body "<Detailed Body>" --draft --base main`
