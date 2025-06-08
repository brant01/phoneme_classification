<!-- CLAUDE.md Template Version: 1.0 -->
# Claude.md - Project Configuration v1.0

## NAVIGATION GUIDE FOR CLAUDE

- **First time**: Read entire file
- **Session start**: Read [PROJECT DETAILS] + [SESSION START CHECKLIST]
- **Before coding**: Read [BEFORE WRITING CODE CHECKLIST]
- **Before commit**: Read [COMMIT CHECKLIST] + [GIT REMINDERS]
- **Session end**: Update [PROJECT LOG] + read [SESSION END CHECKLIST]
- **If unsure**: Check [CORE PRINCIPLES] or [WHEN TO ASK]

---

## [PROJECT DETAILS] - EDITABLE BY CLAUDE

<!-- Claude updates this section based on project needs -->

Project Name: [PROJECT_NAME]  
Last Updated: [DATE]  
Description: [Brief project description]

### Project Context

- **Primary Goal**: [Main objective]
- **Project Type**: [e.g., Web app, CLI tool, Library]
- **Data Types**: [Key data structures used]
- **Key Libraries**: [Main dependencies]
- **Performance Requirements**: [Speed, memory, etc.]
- **Domain Context**: [Business/technical domain]

### Core Functionality

[List main features and components]

### Design Decisions

[Document key architectural choices and rationale]

### Architecture

```
[Project structure diagram]
```

---

## [CORE PRINCIPLES] - PROTECTED

### Critical Mindset

1. **THINK FIRST** - 80% planning, 20% coding
2. **CHALLENGE EVERYTHING** - If you see a better way, say so
3. **VERIFY ASSUMPTIONS** - Question my suggestions
4. **BE CRITICAL** - Present honest tradeoffs
5. **TEACH CLEARLY** - I'm self-taught, explain thoroughly

### Communication Rules

- NO emojis, icons, or symbols anywhere in the project
- NO "co-authored by" or "with Claude" in commits or comments
- Professional, direct communication
- Challenge suboptimal suggestions immediately

### Technical Standards

- **Language**: [Primary language and version]
- **Dependencies**: [Dependency management approach]
- **Error Handling**: [Error handling strategy]
- **Testing**: [Testing approach]
- **Structure**: [Code organization principles]

### Security Requirements (NON-NEGOTIABLE)

1. Never store credentials in code
2. Use environment variables for secrets
3. Validate all user input
4. Handle file paths safely
5. Follow security best practices for the domain

---

## [WHEN TO ASK] - PROTECTED

### Always Ask Before

- Adding new dependencies
- Changing core architecture
- Modifying data structures
- Adding complex features
- Changing default behaviors

### How to Present Options

"I see X approaches:
- Option A: [approach] - Pros: [list] Cons: [list]
- Option B: [approach] - Pros: [list] Cons: [list]
Which aligns with your needs?"

---

## [SESSION START CHECKLIST]

- [ ] Read PROJECT DETAILS section
- [ ] Check recent changes in PROJECT LOG
- [ ] Review any pending TODOs
- [ ] Verify development environment

---

## [BEFORE WRITING CODE CHECKLIST]

- [ ] Is this the simplest solution? (KISS)
- [ ] Are we building only what's needed? (YAGNI)
- [ ] Does this follow project conventions?
- [ ] Error handling comprehensive?
- [ ] Have I presented alternatives?

---

## [COMMIT CHECKLIST]

- [ ] Code follows project style
- [ ] Functions/methods documented
- [ ] Error messages are helpful
- [ ] No hardcoded values
- [ ] Tests pass (if applicable)
- [ ] Updated PROJECT LOG
- [ ] Commit message clear (no co-author references)

---

## [SESSION END CHECKLIST]

- [ ] Update PROJECT LOG with decisions
- [ ] Note any unresolved questions
- [ ] Document next steps
- [ ] Add TODOs for incomplete work

---

## [GIT REMINDERS]

### Before EVERY Push

1. **Test core functions**: [project-specific test command]
2. **Run linter**: [project-specific lint command]
3. **Push**: `git push origin main`

### Commit Format

```
type: brief description

Detailed explanation if needed
```

NO "Co-authored-by" tags ever.

### Required .gitignore Entries

```
# Environment
.env
*.local

# Dependencies
[language-specific ignore patterns]

# OS files
.DS_Store
Thumbs.db
```

---

## [TODO/FIXME GUIDELINES]

When adding TODOs that should be tracked by pm tool:

```python
# TODO: Add error handling for API calls
# TODO Add validation for user input
# FIXME: This breaks when input is None
# TODO - implement caching mechanism
```

Requirements:
- Must be uppercase: `TODO` or `FIXME`
- Must be in .py files within src/ directory
- Common formats: `# TODO:`, `# TODO`, `# FIXME:`, inline comments
- Also works in docstrings: `"""TODO: implement this method"""`

---

## [PROJECT LOG] - EDITABLE BY CLAUDE

### Template Version: v1.0 (2025-06-03)

### Project Sessions

#### [DATE] - Session 1: Initial Setup

**Decisions**:
- [Key decisions made]

**Architecture Choices**:
- [Technical choices and rationale]

**Next Steps**:
- [ ] [Immediate tasks]

#### Active Development

<!-- Current work - clean up when complete -->
- Working on: [Current focus]
- Questions: [Open questions]

---

## [QUICK REFERENCE]

### Common Commands

```bash
# Development
[project-specific commands]

# Testing
[test commands]

# Deployment
[deployment commands]
```

### Development Philosophy

**KISS** > YAGNI > DRY  
Simple > Clever  
Explicit > Implicit  
Working > Perfect