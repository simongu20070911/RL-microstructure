# Version Control Guide

## Basic Git Commands

### Setup
```bash
git init  # Initialize new repository
git clone <url>  # Clone existing repository
```

### Daily Workflow
```bash
git status  # Check current status
git add <files>  # Stage changes
git commit -m "Descriptive message"  # Commit changes
git push  # Push to remote
git pull  # Pull latest changes
```

### Branching
```bash
git branch  # List branches
git checkout -b feature/name  # Create new feature branch
git merge branch-name  # Merge branches
```

### Viewing History
```bash
git log  # Show commit history
git log --oneline  # Show compact history
git log --graph  # Show history with branches
git log -p <file> # Show history for a specific file
git show <commit> # Show details of a specific commit
```

### Managing Changes
```bash
git diff  # Show unstaged changes
git diff --staged  # Show staged changes
git checkout -- <file>  # Discard changes in working directory
git reset HEAD <file>  # Unstage a file
```

### Rolling Back Changes
**Caution:** Use `reset --hard` with care, as it discards changes permanently.

```bash
# Revert a specific commit (creates a new commit undoing the changes)
git revert <commit_hash>

# Reset to a previous commit (moves HEAD, potentially discarding history)
# --soft: Keeps changes staged
# --mixed (default): Keeps changes in working directory, unstaged
# --hard: Discards all changes since the commit
git reset --soft <commit_hash>
git reset <commit_hash>
git reset --hard <commit_hash>

# Restore a specific file from a previous commit
git checkout <commit_hash> -- <file>
```

## Best Practices

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Add details in body if needed

### Branch Naming
- feature/name - New features
- bugfix/name - Bug fixes
- hotfix/name - Critical production fixes

## Workflow Guidelines

1. Always pull latest changes before starting work
2. Create a new branch for each feature/bugfix
3. Commit small, logical changesets
4. Push your branch frequently
5. Create pull requests for code review
6. Delete merged branches

## Ignored Files
The following are excluded in .gitignore:
- logs/
- experiments/
- data files
- IDE config files
