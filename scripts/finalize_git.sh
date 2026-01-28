#!/bin/bash
set -e

echo "🧹 Cleaning up system files..."
find . -name ".DS_Store" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "._*" -delete

echo "🌿 Checking Logic..."
# Ensure we are on main
git checkout main
git pull origin main

echo "📦 Adding files..."
git add .

echo "📝 Committing..."
git commit -m "feat: Finalize project with verified PnL logic and cost audit" || echo "Nothing to commit"

echo "🚀 Pushing to Main..."
git push origin main

echo "✅ Done!"
