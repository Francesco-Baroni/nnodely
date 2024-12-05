#!/bin/sh

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: ./release <branch_name> <commit_message>"
  exit 1
fi

BRANCH_NAME=$1
COMMIT_MESSAGE=$2

# Create a new branch and check it out
git checkout -b "release/$BRANCH_NAME"

# Commit the changes
git add .
git commit -m "$COMMIT_MESSAGE"

# Push the new branch to origin with the commit message
git push origin --set-upstream origin release/$BRANCH_NAME