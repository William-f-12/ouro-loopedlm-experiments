# Ouro Colab Implementation Plan

Goal: ship a self-contained editable model project and upload-and-run Colab notebook.
Architecture: local Python model package + shared inference CLI + verified source ZIP.
Tech stack: Python, PyTorch supplied by runtime, Transformers 4.57.1, nbformat 4.
Spec: design.md.

## Global constraints

- Work only in this new project. No cloud provisioning or full-weight download.
- Preserve upstream source revision and attribution. Do not change model mathematics.
- Notebook inference uses a fresh subprocess and the local package.
- Do not install or replace the user's global packages; reuse the isolated CPU probe dependencies.

## Tasks

- [x] 1. Import fixed upstream source/config/license and record SHA256 provenance.
- [x] 2. Add meaningful bundle contract tests first; implement source-only deterministic packaging and verified extraction.
- [x] 3. Implement local-source inference CLI with revision pin, fresh DynamicCache, parameter checks, provenance and JSON output; add a no-download CPU smoke test.
- [x] 4. Create Colab notebook and Chinese instructions for ZIP upload, GPU checks, inference, results and repeated edits.
- [x] 5. Validate CPU cache correctness, packaging, extracted project and notebook upload/execution cells; perform final review and produce upload ZIP.

## Review focus

- Modified source must be imported, not remote model code or an old notebook module.
- Upload ZIP must preserve new source files while excluding weights/secrets/caches.
- Extracted package must work from directories containing spaces.
- GPU absence and oversized inputs must fail with actionable messages.
- CPU verification must not imply full pretrained GPU validation.

## Execution record

User explicitly requested implementation of the previous deployment proposal. Proceed inline with reversible local work. This directory is outside an existing Git repository, so there is no branch/worktree to integrate. Keep the deliverable directly in the requested new directory.

Completed 2026-10-11. Seven tests passed, including offline full-CLI inference with a tiny checkpoint. CPU smoke checks passed for 3/4 loops. Notebook schema/cells validated. Independent read-only review found no substantive workflow issues. The integration test exposed an unused token_type_ids field from a generic fast tokenizer; the entrypoint now explicitly excludes it. See validation.md for exact scope and limits.
