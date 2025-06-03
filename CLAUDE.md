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

Project Name: Phoneme Classification VAE
Last Updated: 2025-06-03
Description: Variational Autoencoder for phoneme classification that learns voice-invariant representations from wav recordings

### Project Context

- **Primary Goal**: Train a VAE to classify phonemes while being invariant to speaker gender (male/female voices)
- **Project Type**: Experiment/Research
- **Data Types**: .wav files (16kHz), numpy arrays, model checkpoints (.pth), latent representations (.npy)
- **Key Libraries**: torch, torchaudio, soundfile, polars, scikit-learn, pywavelets
- **Performance Requirements**: Memory-efficient for audio processing, reproducible training with fixed seeds
- **Domain Context**: Speech processing/phonetics research - audio data contains phoneme recordings from male and female speakers

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

- **Python 3.12+** with modern syntax (built-in generics)
- **Package Management**: `uv` only
- **Formatting/Linting**: `ruff` (88-char lines)
- **Testing**: `pytest` with descriptive names
- **Config**: `hydra`/`omegaconf` ONLY for experiment projects
- **Structure**: Use modern best practices for project organization

### Security Requirements (NON-NEGOTIABLE)

1. ALL data is PHI until proven otherwise
2. NEVER commit data files
3. NEVER hardcode paths or credentials
4. ALWAYS use environment variables
5. Add comprehensive .gitignore before first commit

---

## [WHEN TO ASK] - PROTECTED

### Always Ask Before

- Adding dependencies
- Making architectural decisions
- Optimizing code (prove it's slow first)
- Creating abstractions
- Deviating from patterns

### How to Present Options

"I see X approaches:

- Option A: [approach] - Pros: [list] Cons: [list]
- Option B: [approach] - Pros: [list] Cons: [list] Which aligns with your needs?"

---

## [SESSION START CHECKLIST]

- [ ] Read PROJECT DETAILS section
- [ ] Check recent changes in PROJECT LOG
- [ ] Verify environment setup
- [ ] Review any pending questions/tasks

---

## [BEFORE WRITING CODE CHECKLIST]

- [ ] Is this the simplest solution? (KISS)
- [ ] Are we building only what's needed? (YAGNI)
- [ ] Will this introduce bias or errors?
- [ ] Any data security risks?
- [ ] Have I presented alternatives?

---

## [COMMIT CHECKLIST]

- [ ] Functions documented with types
- [ ] Tests written for new code
- [ ] No hardcoded paths/credentials
- [ ] Ran `ruff check . --fix`
- [ ] Ran `ruff format .`
- [ ] Updated PROJECT LOG
- [ ] Commit message clear (no co-author references)

---

## [SESSION END CHECKLIST]

- [ ] Update PROJECT LOG with decisions
- [ ] Note any unresolved questions
- [ ] Document next steps
- [ ] Clean up "Active Development Notes"

---

## [GIT REMINDERS]

### Before EVERY Push

1. **Sync claude.md to private repo**
2. **Run**: `uv run pytest`
3. **Run**: `uv run ruff check .`
4. **Push**: `git push origin main` (be explicit)

### Commit Format

```
type: brief description

Detailed explanation if needed
```

NO "Co-authored-by" tags ever.

### Required .gitignore Entries

```
# Data files
data/
*.csv
*.txt
*.wav
*.rhs
*.pkl
*.npy
*.h5

# Project meta
claude.md
CLAUDE.md
.env
configs/local/

# Results
results/
outputs/
*.log
```

---

## [PROJECT LOG] - EDITABLE BY CLAUDE

### Template Version: v1.0 (2025-06-03)

### Project Sessions

#### 2025-06-03 - Session 1: Project Setup & Claude Code Integration

**Decisions**:

- Updated CLAUDE.md with phoneme classification project specifics
- Confirmed .gitignore already excludes outputs/runs directories
- Project uses PyTorch VAE architecture with encoder/decoder for speaker-invariant phoneme classification

**Alternatives Rejected**:

- N/A (initial setup session)

**Next Steps**:

- [ ] Remove outputs/runs from git history to reduce repo size
- [ ] Review project structure for potential refactoring needs
- [ ] Add ruff formatting configuration if needed

#### Active Development

<!-- Current work - clean up when complete -->

- Working on: Initial Claude Code setup and git history cleanup
- Questions: None currently

---

## [QUICK REFERENCE]

### Common Commands

```bash
uv venv && uv pip install -e .
uv run pytest
uv run ruff check . --fix && uv run ruff format .
git add . && git commit -m "type: message" && git push origin main
```

### Development Philosophy

**KISS** > YAGNI > DRY Simple > Clever Explicit > Implicit Security > Everything