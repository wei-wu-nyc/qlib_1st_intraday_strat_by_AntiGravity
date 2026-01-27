#!/bin/bash
set -e

echo "🧹 Cleaning up system files..."
find . -name ".DS_Store" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "._*" -delete

echo "🌿 Checking/Creating Branch..."
# Try to switch to existing or create new
git checkout feature/ensemble-moe-complete 2>/dev/null || git checkout -b feature/ensemble-moe-complete

echo "📦 Adding files..."
git add .

echo "📝 Committing..."
git commit -m "feat: Finalize Strategy (Ensemble MoE 36-bar) and add comprehensive dashboards" || echo "Nothing to commit"

echo "🚀 Pushing..."
git push origin feature/ensemble-moe-complete

echo "✅ Done!"
