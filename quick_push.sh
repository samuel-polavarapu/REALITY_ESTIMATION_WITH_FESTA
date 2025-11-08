#!/bin/bash

# Quick Push to GitHub Script
# This script pushes to GitHub using the provided token

echo "==================================="
echo "FESTA - Quick GitHub Push"
echo "==================================="
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "Error: Not a git repository!"
    exit 1
fi

# Get repository information
read -p "Enter your GitHub username: " GITHUB_USERNAME
read -p "Enter your repository name (default: LLAVA-V5-2): " REPO_NAME
REPO_NAME=${REPO_NAME:-LLAVA-V5-2}

# Your GitHub token
GITHUB_TOKEN="YOUR_GITHUB_TOKEN_HERE"

# Construct the repository URL
REPO_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

echo ""
echo "Repository: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo ""

# Remove existing origin if present
git remote remove origin 2>/dev/null

# Add remote with token
echo "Adding remote repository..."
git remote add origin "$REPO_URL"

if [ $? -ne 0 ]; then
    echo "Error: Failed to add remote!"
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Show what will be pushed
echo ""
echo "Commits to be pushed:"
git log --oneline -5
echo ""

# Confirm push
read -p "Push to GitHub now? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Push cancelled."
    exit 0
fi

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git push -u origin $CURRENT_BRANCH

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully pushed to GitHub!"
    echo ""
    echo "View your repository at:"
    echo "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
    echo ""
else
    echo ""
    echo "✗ Push failed!"
    echo ""
    echo "Troubleshooting:"
    echo "1. Make sure the repository exists on GitHub"
    echo "2. Verify your username is correct: $GITHUB_USERNAME"
    echo "3. Check that the token has the required permissions"
    echo "4. Ensure repository name is correct: $REPO_NAME"
    exit 1
fi

echo "==================================="
echo "Next Steps:"
echo "==================================="
echo "1. Go to: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo "2. Add a description to your repository"
echo "3. Add topics: vision-language-models, evaluation-framework, etc."
echo "4. Configure repository settings as needed"
echo ""

