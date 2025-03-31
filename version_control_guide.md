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
