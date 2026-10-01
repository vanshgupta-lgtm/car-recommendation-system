#!/bin/bash
# Move to repository directory
cd "$(dirname "$0")"

echo "=========================================================="
echo "🚀 Push Car Recommendation System to GitHub..."
echo "=========================================================="

echo "Enter your GitHub Repository URL (e.g., https://github.com/vanshgupta-lgtm/car-recommendation-system.git):"
read repo_url

if [ -n "$repo_url" ]; then
    git remote remove origin 2>/dev/null
    git remote add origin "$repo_url"
    git branch -M main
    git push -u origin main
    echo "✅ Code pushed successfully to GitHub!"
    echo "Now visit: https://share.streamlit.io to deploy in 1-click!"
else
    echo "❌ No URL provided. Aborting."
fi
