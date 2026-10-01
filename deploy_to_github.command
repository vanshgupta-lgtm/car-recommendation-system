#!/bin/bash
cd "$(dirname "$0")"

echo "=========================================================="
echo "🚀 Push Car Recommendation System to GitHub"
echo "=========================================================="
echo ""

REPO="https://github.com/vanshgupta-lgtm/car-recommendation-system.git"
git remote remove origin 2>/dev/null
git remote add origin "$REPO"
git branch -M main

echo "GitHub par upload karne ke 2 aasan tareeqe hain:"
echo "1) Browser se Login (Automatic - Press 1 & Enter)"
echo "2) GitHub Personal Token se (Press 2 & Enter)"
echo ""
read -p "Select option (1 ya 2, default is 1): " choice

if [ "$choice" == "2" ]; then
    echo ""
    echo "Token lene ke liye is link par jayein:"
    echo "👉 https://github.com/settings/tokens/new"
    echo "(Note: 'repo' checkbox tick karke Generate Token click karein)"
    echo ""
    read -p "Paste your GitHub Token (ghp_...): " token
    token=$(echo "$token" | tr -d '[:space:]')
    if [ -n "$token" ]; then
        echo "Pushing code to GitHub..."
        git remote set-url origin "https://${token}@github.com/vanshgupta-lgtm/car-recommendation-system.git"
        if git push -u origin main; then
            echo ""
            echo "=========================================================="
            echo "🎉 SUCCESS! Code GitHub par successfully push ho gaya!"
            echo "=========================================================="
            echo "Ab seedha https://share.streamlit.io par jaakar Deploy karein!"
        else
            echo "❌ Push fail ho gaya. Kripya check karein ki token mein 'repo' permission thi."
        fi
        git remote set-url origin "$REPO"
    fi
else
    echo ""
    echo "Starting Browser Login..."
    gh auth login --web -p https
    gh auth setup-git
    echo ""
    echo "Pushing code to GitHub..."
    if git push -u origin main; then
        echo ""
        echo "=========================================================="
        echo "🎉 SUCCESS! Code GitHub par successfully push ho gaya!"
        echo "=========================================================="
        echo "Ab seedha https://share.streamlit.io par jaakar Deploy karein!"
    else
        echo "❌ Push fail ho gaya. Kripya GitHub par check karein ki repository 'car-recommendation-system' create ho chuki hai."
    fi
fi

echo ""
echo "Press Enter to exit..."
read
