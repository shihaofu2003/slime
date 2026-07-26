# Commit convention

We follow [Conventional Commits](https://www.conventionalcommits.org/). Every commit
message starts with a **type** and an optional **scope** so the history reads as a
log of what changed and where.

## Message format

```
<type>(<scope>): <imperative summary>

<optional body — what & why, wrapped ~72 chars>

<optional footer>
```

- Summary: imperative mood (`add`, not `added`), lowercase, ≤ 72 chars, no trailing period.
- Body: explain motivation and what changed, not line-by-line diff prose.
- Footer: breaking changes (`BREAKING CHANGE:`) or co-authors.

## Types

| type       | use for                                                        |
|------------|----------------------------------------------------------------|
| `feat`     | new pipeline / capability / feature                            |
| `fix`      | bug fix                                                        |
| `docs`     | documentation only                                             |
| `refactor` | restructuring without behavior change                          |
| `perf`     | performance improvement                                        |
| `test`     | adding or fixing tests                                         |
| `build`    | build system, dependencies, docker images                      |
| `ci`       | CI / CD configuration                                          |
| `chore`    | repo tooling, `.gitignore`, maintenance, baseline imports      |
| `style`    | formatting / whitespace, no code change                        |

## Scopes

Common scopes in this repo: `gitignore`, `framework`, `scripts`, `tau2-sft`,
`tau2-rl`, `tau2-eval`, `tau2-opd`, `tau2-rft`, `tau-bench`, `repo`, `docs`.
Omit the scope when a change is genuinely repo-wide.

## Branching

Work proceeds in phases. Each phase develops on its own branch off `main`:

- Branch name: `<type>/<scope>` (e.g. `feat/tau2-sft`, `chore/import-framework`).
- Verify the branch (syntax-check scripts, parse modules) before merging.
- Merge with `--no-ff` so each phase is a visible merge commit on `main`.
- Open the next phase's branch from `main` after the previous one merges.
