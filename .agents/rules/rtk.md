---
trigger: always_on
---

# RTK Token Optimization Rules

You are operating in a Windows environment with 'rtk' (Rust Token Killer) successfully installed. To minimize token usage and optimize response speeds, you MUST wrap specified shell commands using the `rtk` proxy prefix.

## Command Rewriting Instructions

Whenever you need to execute a command in the terminal, rewrite it according to these rules:

1. Git Operations:
   - `git status` -> `rtk git status`
   - `git log` -> `rtk git log`
   - `git diff` -> `rtk git diff`
   - `git add` -> `rtk git add`
   - `git commit` -> `rtk git commit`

2. Testing & Linting:
   - `pytest` -> `rtk pytest`
   - `cargo test` -> `rtk cargo test`
   - `tsc` -> `rtk tsc`
   - All standard testing or linter commands -> Prefix with `rtk`

3. File System Exploration:
   - Instead of standard file viewing, prefer using:
     - `rtk read <filename>` for smart, compressed file viewing.
     - `rtk grep "<pattern>"` for compressed search results.
     - `rtk ls .` for optimized directory listing.

## Goal

By using `rtk`, the output will be automatically compressed by 60-90%, filtering out redundant successful logs and focusing only on errors or actionable summaries.
