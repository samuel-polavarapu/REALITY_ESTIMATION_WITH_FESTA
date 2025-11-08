#!/bin/bash

# FESTA GitHub Setup Script
# This script helps you connect your local repository to GitHub

echo "==================================="
echo "FESTA GitHub Repository Setup"
echo "==================================="
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Error: Git repository not initialized!"
    echo "Run 'git init' first."
    exit 1
fi

echo "Current repository status:"
git log --oneline -5 2>/dev/null || echo "No commits yet"
echo ""

# Ask for GitHub repository URL
echo "Please provide your GitHub repository information:"
echo ""
read -p "Enter GitHub repository URL (e.g., https://github.com/username/repo.git): " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "Error: Repository URL cannot be empty!"
    exit 1
fi

# Add remote origin
echo ""
echo "Adding remote origin..."
git remote remove origin 2>/dev/null  # Remove if exists
git remote add origin "$REPO_URL"

if [ $? -eq 0 ]; then
    echo "✓ Remote origin added successfully!"
else
    echo "✗ Failed to add remote origin"
    exit 1
fi

# Show remote
echo ""
echo "Remote repositories:"
git remote -v

# Ask if user wants to push now
echo ""
read -p "Do you want to push to GitHub now? (y/n): " PUSH_NOW

if [ "$PUSH_NOW" = "y" ] || [ "$PUSH_NOW" = "Y" ]; then
    echo ""
    echo "Pushing to GitHub..."
    echo "Note: You may be prompted for your GitHub credentials"
    echo ""

    # Push to main branch
    git push -u origin main

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Successfully pushed to GitHub!"
        echo "Your repository is now available at: $REPO_URL"
    else
        echo ""
        echo "✗ Push failed. Please check your credentials and try again."
        echo ""
        echo "To push manually later, run:"
        echo "  git push -u origin main"
    fi
else
    echo ""
    echo "Skipping push. To push later, run:"
    echo "  git push -u origin main"
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Go to GitHub and verify your repository"
echo "2. Add any collaborators if needed"
echo "3. Update repository settings (description, topics, etc.)"
echo ""

