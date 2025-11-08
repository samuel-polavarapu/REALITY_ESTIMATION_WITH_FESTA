#!/bin/bash
# Quick GitHub Push Script
# Replace YOUR_USERNAME with your actual GitHub username
echo "=================================================="
echo "FESTA 4×14 Strategy - GitHub Push Script"
echo "=================================================="
echo ""
# Prompt for GitHub username
read -p "Enter your GitHub username: " GITHUB_USER
# Prompt for repository name (with default)
read -p "Enter repository name (default: FESTA-4x14-Strategy): " REPO_NAME
REPO_NAME=${REPO_NAME:-FESTA-4x14-Strategy}
echo ""
echo "Setting up remote for: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
# Add remote
git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git || git remote set-url origin https://github.com/$GITHUB_USER/$REPO_NAME.git
# Show remote
echo "Remote configured:"
git remote -v
echo ""
echo "Pushing to GitHub..."
echo ""
# Push to main branch
git branch -M main
git push -u origin main
echo ""
echo "=================================================="
echo "✅ Push complete!"
echo "=================================================="
echo ""
echo "Your repository is now at:"
echo "https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
