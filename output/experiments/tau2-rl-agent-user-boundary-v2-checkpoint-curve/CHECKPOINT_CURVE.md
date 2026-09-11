# boundary-v2 checkpoint curve

## Scope

Seed 300, 100 test tasks × 4 trials, temperature 0.6, max_steps 200. Trend statements use only the same-lineage iter9/39/69/89/99 checkpoints; historical iter9 and raw Instruct are background controls. No model is selected automatically.

## Overall

| Model | Role | pass@1 | pass@4(any) | pass^4 | too_many_errors | max_steps |
|---|---|---:|---:|---:|---:|---:|
| selected_sft | baseline | 23.75% | 42.00% | 8.00% | 6 | 41 |
| lineage_iter9 | same_lineage | 25.00% | 52.00% | 8.00% | 9 | 36 |
| lineage_iter39 | same_lineage | 26.25% | 51.00% | 9.00% | 1 | 38 |
| lineage_iter69 | same_lineage | 24.00% | 47.00% | 8.00% | 1 | 27 |
| lineage_iter89 | same_lineage | 29.75% | 53.00% | 10.00% | 1 | 36 |
| lineage_iter99 | same_lineage | 27.50% | 53.00% | 9.00% | 2 | 35 |
| historical_iter9 | background | 26.25% | 52.00% | 10.00% | 3 | 39 |
| raw_instruct | background | 22.75% | 49.00% | 5.00% | 6 | 11 |

## Domains and telecom namespace

| Model | Domain | pass@1 | pass@4(any) | pass^4 | namespace affected/attempts/terminations |
|---|---|---:|---:|---:|---:|
| selected_sft | airline | 32.50% | 45.00% | 20.00% | — |
| selected_sft | retail | 24.38% | 37.50% | 10.00% | — |
| selected_sft | telecom | 18.75% | 45.00% | 0.00% | 54/326/4 |
| lineage_iter9 | airline | 32.50% | 45.00% | 20.00% | — |
| lineage_iter9 | retail | 24.38% | 50.00% | 10.00% | — |
| lineage_iter9 | telecom | 21.88% | 57.50% | 0.00% | 52/256/8 |
| lineage_iter39 | airline | 27.50% | 40.00% | 15.00% | — |
| lineage_iter39 | retail | 30.00% | 50.00% | 12.50% | — |
| lineage_iter39 | telecom | 21.88% | 57.50% | 2.50% | 10/19/0 |
| lineage_iter69 | airline | 30.00% | 55.00% | 10.00% | — |
| lineage_iter69 | retail | 25.62% | 45.00% | 12.50% | — |
| lineage_iter69 | telecom | 19.38% | 45.00% | 2.50% | 5/6/0 |
| lineage_iter89 | airline | 33.75% | 45.00% | 25.00% | — |
| lineage_iter89 | retail | 29.38% | 47.50% | 12.50% | — |
| lineage_iter89 | telecom | 28.12% | 62.50% | 0.00% | 4/11/0 |
| lineage_iter99 | airline | 35.00% | 65.00% | 15.00% | — |
| lineage_iter99 | retail | 30.00% | 47.50% | 12.50% | — |
| lineage_iter99 | telecom | 21.25% | 52.50% | 2.50% | 8/8/0 |
| historical_iter9 | airline | 33.75% | 45.00% | 30.00% | — |
| historical_iter9 | retail | 26.88% | 52.50% | 7.50% | — |
| historical_iter9 | telecom | 21.88% | 55.00% | 2.50% | 36/151/3 |
| raw_instruct | airline | 22.50% | 45.00% | 10.00% | — |
| raw_instruct | retail | 28.12% | 57.50% | 7.50% | — |
| raw_instruct | telecom | 17.50% | 42.50% | 0.00% | 74/289/6 |

## Same-lineage direction

| Adjacent checkpoints | Direction | Δ pass@1 | Δ pass@4(any) |
|---|---|---:|---:|
| lineage_iter39-lineage_iter9 | rise | +1.25 pp | -1.00 pp |
| lineage_iter69-lineage_iter39 | decline | -2.25 pp | -4.00 pp |
| lineage_iter89-lineage_iter69 | rise | +5.75 pp | +6.00 pp |
| lineage_iter99-lineage_iter89 | decline | -2.25 pp | +0.00 pp |

## Paired bootstrap intervals

| Type | Comparison | Scope | Metric | Delta | 95% CI |
|---|---|---|---|---:|---:|
| relative_to_sft | lineage_iter9-selected_sft | overall | pass@1 / pass^1 | +1.25 pp | [-3.25, +5.75] pp |
| relative_to_sft | lineage_iter9-selected_sft | overall | pass@4(any) | +10.00 pp | [+2.00, +19.00] pp |
| relative_to_sft | lineage_iter9-selected_sft | overall | pass^4 | +0.00 pp | [-4.00, +4.00] pp |
| relative_to_sft | lineage_iter9-selected_sft | airline | pass@1 / pass^1 | +0.00 pp | [-11.25, +11.25] pp |
| relative_to_sft | lineage_iter9-selected_sft | airline | pass@4(any) | +0.00 pp | [-15.00, +15.00] pp |
| relative_to_sft | lineage_iter9-selected_sft | airline | pass^4 | +0.00 pp | [-20.00, +20.00] pp |
| relative_to_sft | lineage_iter9-selected_sft | retail | pass@1 / pass^1 | +0.00 pp | [-7.50, +7.50] pp |
| relative_to_sft | lineage_iter9-selected_sft | retail | pass@4(any) | +12.50 pp | [-2.50, +27.50] pp |
| relative_to_sft | lineage_iter9-selected_sft | retail | pass^4 | +0.00 pp | [+0.00, +0.00] pp |
| relative_to_sft | lineage_iter9-selected_sft | telecom | pass@1 / pass^1 | +3.12 pp | [-3.12, +9.38] pp |
| relative_to_sft | lineage_iter9-selected_sft | telecom | pass@4(any) | +12.50 pp | [+0.00, +25.00] pp |
| relative_to_sft | lineage_iter9-selected_sft | telecom | pass^4 | +0.00 pp | [+0.00, +0.00] pp |
| relative_to_sft | lineage_iter39-selected_sft | overall | pass@1 / pass^1 | +2.50 pp | [-2.25, +7.50] pp |
| relative_to_sft | lineage_iter39-selected_sft | overall | pass@4(any) | +9.00 pp | [+0.00, +18.00] pp |
| relative_to_sft | lineage_iter39-selected_sft | overall | pass^4 | +1.00 pp | [-4.00, +6.00] pp |
| relative_to_sft | lineage_iter39-selected_sft | airline | pass@1 / pass^1 | -5.00 pp | [-15.00, +3.75] pp |
| relative_to_sft | lineage_iter39-selected_sft | airline | pass@4(any) | -5.00 pp | [-15.00, +0.00] pp |
| relative_to_sft | lineage_iter39-selected_sft | airline | pass^4 | -5.00 pp | [-25.00, +15.00] pp |
| relative_to_sft | lineage_iter39-selected_sft | retail | pass@1 / pass^1 | +5.62 pp | [-2.50, +13.12] pp |
| relative_to_sft | lineage_iter39-selected_sft | retail | pass@4(any) | +12.50 pp | [-2.50, +27.50] pp |
| relative_to_sft | lineage_iter39-selected_sft | retail | pass^4 | +2.50 pp | [+0.00, +7.50] pp |
| relative_to_sft | lineage_iter39-selected_sft | telecom | pass@1 / pass^1 | +3.12 pp | [-4.38, +11.25] pp |
| relative_to_sft | lineage_iter39-selected_sft | telecom | pass@4(any) | +12.50 pp | [-2.50, +27.50] pp |
| relative_to_sft | lineage_iter39-selected_sft | telecom | pass^4 | +2.50 pp | [+0.00, +7.50] pp |
| relative_to_sft | lineage_iter69-selected_sft | overall | pass@1 / pass^1 | +0.25 pp | [-4.75, +5.25] pp |
| relative_to_sft | lineage_iter69-selected_sft | overall | pass@4(any) | +5.00 pp | [-5.00, +15.00] pp |
| relative_to_sft | lineage_iter69-selected_sft | overall | pass^4 | +0.00 pp | [-6.00, +6.00] pp |
| relative_to_sft | lineage_iter69-selected_sft | airline | pass@1 / pass^1 | -2.50 pp | [-15.00, +10.00] pp |
| relative_to_sft | lineage_iter69-selected_sft | airline | pass@4(any) | +10.00 pp | [-10.00, +30.00] pp |
| relative_to_sft | lineage_iter69-selected_sft | airline | pass^4 | -10.00 pp | [-30.00, +10.00] pp |
| relative_to_sft | lineage_iter69-selected_sft | retail | pass@1 / pass^1 | +1.25 pp | [-5.62, +8.12] pp |
| relative_to_sft | lineage_iter69-selected_sft | retail | pass@4(any) | +7.50 pp | [-7.50, +22.50] pp |
| relative_to_sft | lineage_iter69-selected_sft | retail | pass^4 | +2.50 pp | [-5.00, +10.00] pp |
| relative_to_sft | lineage_iter69-selected_sft | telecom | pass@1 / pass^1 | +0.62 pp | [-7.50, +8.75] pp |
| relative_to_sft | lineage_iter69-selected_sft | telecom | pass@4(any) | +0.00 pp | [-17.50, +17.50] pp |
| relative_to_sft | lineage_iter69-selected_sft | telecom | pass^4 | +2.50 pp | [+0.00, +7.50] pp |
| relative_to_sft | lineage_iter89-selected_sft | overall | pass@1 / pass^1 | +6.00 pp | [+1.00, +11.00] pp |
| relative_to_sft | lineage_iter89-selected_sft | overall | pass@4(any) | +11.00 pp | [+2.00, +21.00] pp |
| relative_to_sft | lineage_iter89-selected_sft | overall | pass^4 | +2.00 pp | [-3.00, +7.00] pp |
| relative_to_sft | lineage_iter89-selected_sft | airline | pass@1 / pass^1 | +1.25 pp | [-8.75, +12.50] pp |
| relative_to_sft | lineage_iter89-selected_sft | airline | pass@4(any) | +0.00 pp | [-20.00, +20.00] pp |
| relative_to_sft | lineage_iter89-selected_sft | airline | pass^4 | +5.00 pp | [-10.00, +20.00] pp |
| relative_to_sft | lineage_iter89-selected_sft | retail | pass@1 / pass^1 | +5.00 pp | [-2.50, +12.50] pp |
| relative_to_sft | lineage_iter89-selected_sft | retail | pass@4(any) | +10.00 pp | [-2.50, +22.50] pp |
| relative_to_sft | lineage_iter89-selected_sft | retail | pass^4 | +2.50 pp | [-5.00, +10.00] pp |
| relative_to_sft | lineage_iter89-selected_sft | telecom | pass@1 / pass^1 | +9.38 pp | [+0.62, +18.12] pp |
| relative_to_sft | lineage_iter89-selected_sft | telecom | pass@4(any) | +17.50 pp | [+0.00, +35.00] pp |
| relative_to_sft | lineage_iter89-selected_sft | telecom | pass^4 | +0.00 pp | [+0.00, +0.00] pp |
| relative_to_sft | lineage_iter99-selected_sft | overall | pass@1 / pass^1 | +3.75 pp | [-1.00, +8.50] pp |
| relative_to_sft | lineage_iter99-selected_sft | overall | pass@4(any) | +11.00 pp | [+2.00, +20.00] pp |
| relative_to_sft | lineage_iter99-selected_sft | overall | pass^4 | +1.00 pp | [-3.00, +5.00] pp |
| relative_to_sft | lineage_iter99-selected_sft | airline | pass@1 / pass^1 | +2.50 pp | [-6.25, +10.00] pp |
| relative_to_sft | lineage_iter99-selected_sft | airline | pass@4(any) | +20.00 pp | [+5.00, +40.00] pp |
| relative_to_sft | lineage_iter99-selected_sft | airline | pass^4 | -5.00 pp | [-20.00, +10.00] pp |
| relative_to_sft | lineage_iter99-selected_sft | retail | pass@1 / pass^1 | +5.62 pp | [-1.88, +13.75] pp |
| relative_to_sft | lineage_iter99-selected_sft | retail | pass@4(any) | +10.00 pp | [-2.50, +22.50] pp |
| relative_to_sft | lineage_iter99-selected_sft | retail | pass^4 | +2.50 pp | [+0.00, +7.50] pp |
| relative_to_sft | lineage_iter99-selected_sft | telecom | pass@1 / pass^1 | +2.50 pp | [-5.62, +10.00] pp |
| relative_to_sft | lineage_iter99-selected_sft | telecom | pass@4(any) | +7.50 pp | [-7.50, +22.50] pp |
| relative_to_sft | lineage_iter99-selected_sft | telecom | pass^4 | +2.50 pp | [+0.00, +7.50] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | overall | pass@1 / pass^1 | +1.25 pp | [-3.25, +5.75] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | overall | pass@4(any) | -1.00 pp | [-10.00, +8.00] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | overall | pass^4 | +1.00 pp | [-3.00, +5.00] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | airline | pass@1 / pass^1 | -5.00 pp | [-18.75, +7.50] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | airline | pass@4(any) | -5.00 pp | [-20.00, +10.00] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | airline | pass^4 | -5.00 pp | [-20.00, +10.00] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | retail | pass@1 / pass^1 | +5.62 pp | [+0.00, +11.25] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | retail | pass@4(any) | +0.00 pp | [-15.00, +15.00] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | retail | pass^4 | +2.50 pp | [+0.00, +7.50] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | telecom | pass@1 / pass^1 | +0.00 pp | [-6.88, +6.88] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | telecom | pass@4(any) | +0.00 pp | [-17.50, +17.50] pp |
| adjacent_checkpoint | lineage_iter39-lineage_iter9 | telecom | pass^4 | +2.50 pp | [+0.00, +7.50] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | overall | pass@1 / pass^1 | -2.25 pp | [-7.00, +2.50] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | overall | pass@4(any) | -4.00 pp | [-14.00, +6.00] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | overall | pass^4 | -1.00 pp | [-5.00, +3.00] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | airline | pass@1 / pass^1 | +2.50 pp | [-8.75, +12.50] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | airline | pass@4(any) | +15.00 pp | [+0.00, +30.00] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | airline | pass^4 | -5.00 pp | [-15.00, +0.00] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | retail | pass@1 / pass^1 | -4.38 pp | [-10.00, +1.25] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | retail | pass@4(any) | -5.00 pp | [-17.50, +7.50] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | retail | pass^4 | +0.00 pp | [-7.50, +7.50] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | telecom | pass@1 / pass^1 | -2.50 pp | [-11.88, +6.88] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | telecom | pass@4(any) | -12.50 pp | [-32.50, +7.50] pp |
| adjacent_checkpoint | lineage_iter69-lineage_iter39 | telecom | pass^4 | +0.00 pp | [-7.50, +7.50] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | overall | pass@1 / pass^1 | +5.75 pp | [+0.25, +11.25] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | overall | pass@4(any) | +6.00 pp | [-4.00, +16.00] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | overall | pass^4 | +2.00 pp | [-3.00, +8.00] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | airline | pass@1 / pass^1 | +3.75 pp | [-11.25, +17.50] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | airline | pass@4(any) | -10.00 pp | [-35.00, +15.00] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | airline | pass^4 | +15.00 pp | [+0.00, +30.00] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | retail | pass@1 / pass^1 | +3.75 pp | [-5.00, +11.88] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | retail | pass@4(any) | +2.50 pp | [-10.00, +15.00] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | retail | pass^4 | +0.00 pp | [-10.00, +10.00] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | telecom | pass@1 / pass^1 | +8.75 pp | [+0.62, +17.50] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | telecom | pass@4(any) | +17.50 pp | [+0.00, +35.00] pp |
| adjacent_checkpoint | lineage_iter89-lineage_iter69 | telecom | pass^4 | -2.50 pp | [-7.50, +0.00] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | overall | pass@1 / pass^1 | -2.25 pp | [-6.75, +2.25] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | overall | pass@4(any) | +0.00 pp | [-10.00, +10.00] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | overall | pass^4 | -1.00 pp | [-6.00, +4.00] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | airline | pass@1 / pass^1 | +1.25 pp | [-7.50, +10.00] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | airline | pass@4(any) | +20.00 pp | [+5.00, +40.00] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | airline | pass^4 | -10.00 pp | [-25.00, +0.00] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | retail | pass@1 / pass^1 | +0.62 pp | [-7.50, +8.75] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | retail | pass@4(any) | +0.00 pp | [-15.00, +15.00] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | retail | pass^4 | +0.00 pp | [-10.00, +10.00] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | telecom | pass@1 / pass^1 | -6.88 pp | [-13.12, -0.62] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | telecom | pass@4(any) | -10.00 pp | [-27.50, +7.50] pp |
| adjacent_checkpoint | lineage_iter99-lineage_iter89 | telecom | pass^4 | +2.50 pp | [+0.00, +7.50] pp |
