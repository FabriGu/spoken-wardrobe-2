# Migrating from Cursor to Claude Code on Mac: Complete Setup Guide

### Understanding CLAUDE.md: your project memory

**CLAUDE.md structure for React/TypeScript project:**

````markdown
# MyApp - React + TypeScript + Node.js

## Architecture

- **Frontend:** React 18 with Vite, TypeScript strict mode
- **Backend:** Node.js/Express with PostgreSQL
- **API:** REST endpoints at http://localhost:3001/api/
- **State:** Redux Toolkit with RTK Query

## Development Commands

```bash
npm run dev         # Start dev server (port 3000)
npm test            # Run Jest + React Testing Library
npm run build       # Production build
npm run lint        # ESLint + Prettier check
```
````

## Code Style Conventions

- Use **ES modules** (import/export), never CommonJS
- **Destructure imports:** `import { useState } from 'react'`
- **Always use TypeScript strict mode**
- **Prefer async/await** over .then() chains
- **Functional components only**

## Critical Warnings

- ⚠️ Never commit .env files or secrets
- ⚠️ Production uses PostgreSQL, dev uses SQLite
- ⚠️ Always sanitize user input

````

### Permission configuration examples

**JavaScript/TypeScript project:**

```json
{
  "defaultMode": "default",
  "permissions": {
    "allow": [
      "Read(*)",
      "Grep(*)",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Bash(npm run *)",
      "Bash(git status|diff|log *)"
    ],
    "deny": [
      "Read(.env*)",
      "Edit(.git/**)",
      "Edit(package-lock.json)",
      "Bash(rm *)"
    ]
  }
}
````

**Python project:**

```json
{
  "defaultMode": "default",
  "permissions": {
    "allow": [
      "Read(*)",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Bash(poetry run *)",
      "Bash(pytest*)"
    ],
    "deny": ["Read(.env*)", "Bash(pip install *)"]
  }
}
```

---

## 4. MCP Server Setup for Extended Capabilities

### Essential MCP servers

**Priority 1: Web search (Brave)**

```bash
# Get FREE API key from https://brave.com/search/api/
# Install Brave Search MCP
claude mcp add brave-search -s user -- npx -y @modelcontextprotocol/server-brave-search
```

**Priority 2: GitHub integration**

```bash
# Create GitHub Personal Access Token at github.com/settings/tokens
# Install GitHub MCP
claude mcp add github -s user -- docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server
```

**Priority 3: Filesystem access**

```bash
# Install Filesystem MCP (specify only safe directories)
claude mcp add filesystem -s user -- npx -y @modelcontextprotocol/server-filesystem ~/Documents ~/Projects
```

### Complete MCP configuration

**Location:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "YOUR_BRAVE_API_KEY"
      }
    },
    "github": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_TOKEN"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/YOUR_USERNAME/Documents",
        "/Users/YOUR_USERNAME/Projects"
      ]
    },
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
    }
  }
}
```

### Verify MCP servers work

```bash
# List configured servers
claude mcp list

# Expected output:
# * brave-search: connected
# * filesystem: connected
# * github: connected
# * puppeteer: connected

# Test in Claude session
claude
"Search the web for latest TypeScript best practices"
"List files in my Documents folder"
```

---

## 5. Creating Plan Files and Project Documentation

### Plan file structure

**Location:** `your-repo/plans/features/authentication.md`

````markdown
# Feature: User Authentication System

## Goal

Implement secure user authentication with email/password and OAuth providers.

## Requirements

- [ ] User registration with email verification
- [ ] Login with email/password
- [ ] OAuth integration (Google, GitHub)
- [ ] JWT token-based session management
- [ ] Password reset flow

## Technical Approach

### Database Schema

```sql
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255),
  email_verified BOOLEAN DEFAULT FALSE
);
```
````

### API Endpoints

- POST /api/auth/register
- POST /api/auth/login
- POST /api/auth/logout
- GET /api/auth/oauth/:provider

### Security Considerations

- ⚠️ Never store plain-text passwords
- ⚠️ Use HTTPS in production
- ⚠️ Rate limit authentication endpoints

## Testing Strategy

- Unit tests for password hashing
- Integration tests for auth flows
- Security tests for vulnerabilities

````

### Using plan files

```bash
claude

# Reference plan in conversation
"@plans/features/authentication.md Please implement this feature following the plan"

# Or create custom command
# .claude/commands/implement-auth.md:
"Read @plans/features/authentication.md and implement per specifications"

# Then use:
/implement-auth
````

---

## 6. Optimal Configuration Files

### Complete .claude/settings.json

**JavaScript/TypeScript with hooks:**

```json
{
  "defaultMode": "default",
  "permissions": {
    "allow": [
      "Read(*)",
      "Grep(*)",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Bash(npm run *)",
      "Bash(git status|diff|add|commit *)"
    ],
    "deny": [
      "Read(.env*)",
      "Edit(.git/**)",
      "Edit(package-lock.json)",
      "Bash(rm *)",
      "Bash(git push origin main)"
    ]
  },
  "hooks": [
    {
      "matcher": "Edit|Write",
      "hooks": [
        {
          "type": "command",
          "command": "prettier --write \"$CLAUDE_FILE_PATHS\" || echo '⚠️ Prettier failed'"
        }
      ]
    },
    {
      "matcher": "Edit(src/**/*.ts)",
      "hooks": [
        {
          "type": "command",
          "command": "npx tsc --noEmit --skipLibCheck \"$CLAUDE_FILE_PATHS\" || echo '⚠️ Type errors detected'"
        }
      ]
    }
  ]
}
```

### Custom commands

**Create reusable commands in .claude/commands/:**

**.claude/commands/test.md:**

```markdown
Run the complete test suite with coverage and report any failures.
```

**.claude/commands/refactor.md:**

```markdown
Review the current file for code quality issues:

1. Identify code smells
2. Suggest performance optimizations
3. Check for best practices
4. Ensure proper error handling

Provide recommendations but wait for approval before implementing.
```

**Usage:**

```bash
claude
/test      # Runs test command
/refactor  # Runs refactor analysis
```

---

## 7. Migration from Cursor

### Core differences summary

**Cursor:** GUI-first IDE where you drive and AI assists
**Claude Code:** CLI-first terminal where AI drives and you supervise

**What each does better:**

| Feature              | Better Tool | Why                    |
| -------------------- | ----------- | ---------------------- |
| Tab completion       | Cursor      | Claude Code has none   |
| Quick inline edits   | Cursor      | Cmd+K is faster        |
| Visual diff review   | Cursor      | Side-by-side in editor |
| Large file editing   | Claude Code | Handles 18,000+ lines  |
| Complex refactoring  | Claude Code | Better reasoning       |
| Multi-file changes   | Claude Code | More comprehensive     |
| Terminal integration | Claude Code | Native CLI experience  |
| Predictable cost     | Cursor      | Flat $20/month         |

### Recommended hybrid workflow

**Keep both tools:**

1. **Use Cursor for:**

   - Tab completions while typing
   - Quick Cmd+K inline fixes
   - Visual code review
   - Navigation and exploration

2. **Use Claude Code for:**

   - Complex multi-file refactors
   - Building entire features
   - Large codebase changes
   - Test generation and fixing

3. **Run Claude Code inside Cursor's terminal:**

   ```bash
   # Open Cursor
   code your-project

   # In Cursor's integrated terminal
   claude

   # Get Claude's intelligence + Cursor's visual interface
   ```

### Common migration issues

**Issue 1: Missing tab completion**
**Solution:** Keep Cursor installed, use it for quick completions

**Issue 2: No visual diffs**
**Solution:** Run Claude Code in Cursor's terminal, or use `git diff`

**Issue 3: Expensive token usage**
**Solution:** Use `/clear` frequently, switch to Sonnet for routine tasks

**Issue 4: Permission prompts**
**Solution:** Configure `.claude/settings.json` with appropriate permissions

**Issue 5: Shift+Enter doesn't work**
**Solution:** Run `/terminal-setup` or use `\` + Enter alternative

---

## 8. Workflow Best Practices

### Starting a Claude Code session

**Daily workflow:**

```bash
# 1. Navigate to repository root
cd your-project

# 2. Create feature branch (NEVER work on main)
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# 3. Launch Claude Code
claude

# 4. Set context
"Hi! I'm working on [feature name]. Here's what I need to accomplish: [clear description]"

# 5. Work with Claude
# Give clear instructions
# Review changes before accepting
# Test frequently

# 6. Clear context between tasks
/clear

# 7. Exit and review
/exit
git diff
git status

# 8. Commit
git add .
git commit -m "feat: add feature description"
git push origin feature/your-feature-name
```

### Context management strategies

**Clear context frequently:**

```bash
# Between different tasks
/clear

# When approaching context limits
/compact "preserve authentication logic"

# Resume previous session
claude --resume
```

**Use hierarchical CLAUDE.md files:**

```
~/.claude/CLAUDE.md           # Global preferences
~/projects/CLAUDE.md          # Organization standards
~/project/CLAUDE.md           # Project-specific (highest priority)
~/project/frontend/CLAUDE.md  # Component-specific
```

### Git integration best practices

**Always use feature branches:**

```bash
# ✅ Correct workflow
git checkout -b feature/new-feature
claude
# Work with Claude on feature branch
git add .
git commit -m "feat: implement feature"

# ❌ NEVER do this
git checkout main
claude  # Working directly on main is dangerous
```

**Commit frequently:**

```bash
# Create checkpoints during development
git add .
git commit -m "WIP: progress checkpoint"

# Easy rollback if Claude breaks something
git reset --hard HEAD~1
```

---

## 9. Safety and Backup

### Git branch strategies

**Required safety pattern:**

```bash
# ALWAYS create feature branches before using Claude Code
git checkout -b feature/your-feature

# Use git worktrees for parallel development
git worktree add ../project-feature-1 -b feature/auth
git worktree add ../project-feature-2 -b feature/payments

# Run separate Claude instances in each
cd ../project-feature-1 && claude
# (in new terminal) cd ../project-feature-2 && claude
```

### Reviewing changes before committing

**Three-step review process:**

```bash
# 1. Stop and review
Esc  # Stop Claude execution

# 2. Check what changed
git diff
git status

# 3. Test changes
npm test
npm run lint
npm run build  # Verify builds

# 4. Only commit if all passes
git add .
git commit -m "feat: description"
```

### Rollback procedures

**If something goes wrong:**

```bash
# Option 1: Rewind (v2.0.0+)
Esc Esc
/rewind  # Roll back to checkpoint

# Option 2: Git reset
git reset --hard HEAD  # Discard all changes
git clean -fd          # Remove untracked files

# Option 3: Discard branch
git checkout main
git branch -D broken-feature
git checkout -b feature-retry

# Option 4: Restore from backup
cd /path/to/parent
rm -rf your-project
cp -r your-project-backup your-project
```

### Permissions to NEVER grant

**Critical rules:**

```json
{
  "permissions": {
    "deny": [
      "Read(.env*)", // Never expose secrets
      "Read(secrets/**)", // Never read credentials
      "Edit(.git/**)", // Never modify git internals
      "Bash(rm -rf *)", // Never allow destructive commands
      "Bash(git push origin main)", // Never push to main
      "WebFetch(*)" // Restrict network access
    ]
  }
}
```

**Never use `--dangerously-skip-permissions` except:**

- On isolated feature branches
- In git worktrees
- For experimental code only
- NEVER on main branch
- NEVER with production systems

---

## 10. Verification and Testing

### System health check

```bash
# Primary diagnostic
claude doctor

# Expected output:
# ✓ Claude CLI version
# ✓ Node.js version
# ✓ Installation method
# ✓ Authentication status
# ✓ Ready for AI-assisted coding
```

### Test file access

```bash
claude

# Test basic operations
"list all files in this directory"
"read the README.md file"
"create a test file called test.txt"

# Verify
ls test.txt
rm test.txt
```

### Test web search (if MCP configured)

```bash
claude
"search the web for latest React 18 features"

# Should use Brave search and return results
```

### Test GitHub integration

```bash
claude
"show me my recent GitHub repositories"
"what are the open issues in this repository?"
```

### Troubleshooting common issues

**Command not found:**

```bash
# Check PATH
echo $PATH
which claude

# Add to PATH if needed
export PATH="$PATH:~/.npm-global/bin"
echo 'export PATH="$PATH:~/.npm-global/bin"' >> ~/.zshrc
```

**Permission errors:**

```bash
# Fix npm permissions
sudo chown -R $(whoami) ~/.npm
sudo chown -R $(whoami) $(npm config get prefix)

# Or migrate to local installer
claude migrate-installer
```

**Authentication failing:**

```bash
# Sign out and re-authenticate
/logout
claude
# Complete OAuth flow again
```

**MCP servers not connecting:**

```bash
# Enable debugging
claude --mcp-debug

# Check status
claude mcp list

# View logs
cat ~/Library/Logs/Claude/mcp.log
```

**Search not working:**

```bash
# Install ripgrep
brew install ripgrep

# Verify
rg --version
```

---

## Quick Reference: Essential Commands

### Installation

```bash
npm install -g @anthropic-ai/claude-code
claude --version
claude doctor
```

### Daily Workflow

```bash
cd your-repo
git checkout -b feature/name
claude
/init  # First time only
/clear  # Between tasks
/exit
```

### Inside Claude Code

```bash
/help          # Show commands
/status        # Check authentication
/model         # Switch models
/clear         # Clear context
/exit          # End session
```

### Safety Commands

```bash
Esc            # Stop Claude
git diff       # Review changes
git reset --hard  # Undo changes
```

### Configuration Files

```
~/.claude.json                    # Global settings
~/your-repo/CLAUDE.md             # Project context
~/your-repo/.claude/settings.json # Permissions
~/Library/Application Support/Claude/claude_desktop_config.json  # MCP servers
```

---

## Critical Safety Reminders

✅ **DO:**

- Always work on feature branches
- Backup before first use
- Review all diffs before committing
- Start with restrictive permissions
- Clear context frequently
- Test changes before committing
- Commit CLAUDE.md and .claude/settings.json

❌ **DON'T:**

- Work directly on main branch
- Use `--dangerously-skip-permissions` on main
- Grant access to .env files
- Allow unlimited Bash permissions
- Trust AI output without review
- Forget to backup
- Commit secrets or credentials

---

## Conclusion

**Claude Code is not a replacement for Cursor—it's a complementary tool.** The optimal setup for most developers in 2024-2025 is:

1. **Keep Cursor** for tab completions, quick edits, and visual review
2. **Add Claude Code** for complex refactoring and feature development
3. **Run Claude Code inside Cursor's terminal** for best of both worlds
4. **Use feature branches religiously** with AI agents
5. **Review everything** before committing

**Total investment:** $20/month (Cursor) + $20-100/month (Claude) = $40-120/month for professional developers is justified by time savings and code quality improvements.

**Budget alternative:** VS Code (free) + Claude Code Pro ($20/month) gives 80% of the value at lowest cost.

Start with safe, restrictive permissions and gradually expand as you build confidence. The most important safety measure is using feature branches—this single practice makes all Claude Code experimentation safe and reversible.

For the latest updates and community support:

- Official docs: https://docs.claude.com/en/docs/claude-code/
- GitHub: https://github.com/anthropics/claude-code
- Discord: Claude Developers Discord (check Anthropic documentation for invite)
