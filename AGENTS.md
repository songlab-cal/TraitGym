# Project Guidelines

## Project Overview

TraitGym is a benchmarking framework for evaluating DNA sequence models on causal regulatory variant prediction tasks in human genetics.
It compares genomic language models (gLMs) like GPN-MSA, Enformer, Borzoi, and others on mendelian and complex trait variant datasets.

## Important Notes

- **Installation and usage**: See README.md for installation commands and general usage

## Development Practices

- **Package management**: Use `uv` for Python dependencies
- **Testing**: Run `uv run pytest` before committing
- **Code quality**: Pre-commit hooks enforce ruff formatting and linting

### Type Annotations
- Use Python 3.11+ type annotation syntax throughout
- Include type hints for all function parameters and return values
- Use `typing` module imports only when necessary for complex types
- Prefer built-in generic types (e.g., `list[str]` instead of `List[str]`)

### Constants Over Magic Numbers
- Replace hard-coded values with named constants
- Use descriptive constant names that explain the value's purpose
- Keep constants at the top of the file or in a dedicated constants file

### Meaningful Names
- Variables, functions, and classes should reveal their purpose
- Names should explain why something exists and how it's used

### Smart Comments
- Don't comment on what the code does - make the code self-documenting
- Use comments to explain why something is done a certain way
- Document APIs, complex algorithms, and non-obvious side effects

### Clean Structure
- Keep related code together
- Organize code in a logical hierarchy
- Use consistent file and folder naming conventions

### Code Quality Maintenance
- Refactor continuously
- Fix technical debt early
- Leave code cleaner than you found it

## Behavioral Guidelines

### Verify Information
Always verify information before presenting it. Do not make assumptions or speculate without clear evidence.

### No Apologies
Never use apologies.

### No Understanding Feedback
Avoid giving feedback about understanding in comments or documentation.

### No Summaries Of Your Work
Don't summarize changes made.

### No Unnecessary Confirmations
Don't ask for confirmation of information already provided in the context.

### Preserve Existing Code
Don't remove unrelated code or functionalities. Pay attention to preserving existing structures.

### No Implementation Checks
Don't ask the user to verify implementations that are visible in the provided context.

### No Unnecessary Updates
Don't suggest updates or changes to files when there are no actual modifications needed.

### Provide Real File Links
Always provide links to the real files, not x.md.

### No Current Implementation
Don't show or discuss the current implementation unless specifically requested.

### No Premature Generalizations
If you are asked to implement a specific backend, just stick to that. Do not generalize to other common or related use-cases. You can offer to implement these, but only do so if explicitly instructed to.

## Snakemake Safety Rules

- **NEVER touch upstream files** to trigger downstream rules — use `--forcerun <rule>` instead
- Failed snakemake runs **delete incomplete outputs**, so triggering unnecessary upstream jobs can destroy existing data
- When only rule code changed (not data), use `--forcerun <rule>` on the specific rule
- **Always dry-run first** (`-n`) and verify the full job list before executing
- Ask the user before running snakemake if there's any risk of recomputing expensive upstream steps
