# Training monitoring

`rollout/rewards` is the mean post-GRPO training reward and is expected to be
near zero after group centering. `Zero-signal` counts complete eight-sample
groups with neither binary-outcome variation nor nonzero turn-credit-v2
reallocation. Retry, `max_steps`, and tool-execution-error counts are cumulative
over trajectories written when the snapshot was taken.

## Snapshots

| Time (CST) | Arm | Stage / step | Latest raw / train reward | K2 loss | Latest truncation | Zero-signal groups | Retry / permanent overlong | `max_steps` / execution-error trajectories | Checkpoint / ETA | Health |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2026-08-15 01:24 | Airline | stage-a / 1 | 10.42% / 6.21e-10 | 0.000214 | 0.00% | 1 / 12 | 5 / 0 | 1 / 34 | none / 8.2h | finite; no OOM or traceback |
| 2026-08-15 01:24 | Retail | stage-a / 3 | 37.50% / 0 | 0.000214 | 0.00% | 3 / 24 | 1 / 0 | 0 / 50 | none / 4.8h | finite; no OOM or traceback |
| 2026-08-15 01:24 | Mixed | stage-a / 1 | 22.92% / -6.21e-10 | 0.000286 | 6.25% | 1 / 16 | 6 / 0 | 5 / 17 | none / 9.1h | finite; no OOM or traceback |
| 2026-08-15 01:24 | Telecom | stage-a / rollout 0 | pending | pending | pending | pending | 0 / 0 | 0 / 0 | none / pending | running; no OOM or traceback |
| 2026-08-15 02:27 | Airline | stage-b / 13 | 52.08% / 2.48e-09 | 0.001392 | 0.00% | 8 / 89 | 48 / 5 | 11 / 162 | iter9 / 7.6h | finite; stage-a succeeded; no OOM |
| 2026-08-15 02:27 | Retail | stage-final / 19 | 37.50% / 1.86e-09 | 0.001384 | 0.00% | 12 / 123 | 7 / 0 | 0 / 236 | iter19 / 4.6h | finite; stages a/b succeeded; no OOM |
| 2026-08-15 02:27 | Mixed | stage-b / 12 | 54.17% / 4.97e-09 | 0.000960 | 0.00% | 7 / 82 | 33 / 1 | 9 / 89 | iter9 / 8.7h | finite; stage-a succeeded; no OOM |
| 2026-08-15 02:27 | Telecom | stage-b / 10 | 4.17% / 1.24e-09 | 0.001695 | 0.00% | 1 / 67 | 38 / 1 | 11 / 25 | iter9 / 8.9h | finite; stage-a succeeded; no OOM |
| 2026-08-15 03:29 | Airline | stage-final / 22 | 35.42% / 3.73e-09 | 0.002254 | 0.00% | 12 / 146 | 73 / 6 | 19 / 269 | iter19 / 7.5h | finite; stages a/b succeeded; no OOM |
| 2026-08-15 03:29 | Retail | stage-final / 39 | 18.75% / 1.86e-09 | 0.003619 | 0.00% | 22 / 240 | 11 / 0 | 4 / 443 | iter39 / 3.5h | finite; no OOM |
| 2026-08-15 03:29 | Mixed | stage-final / 24 | 25.00% / 3.10e-09 | 0.003249 | 6.25% | 19 / 154 | 59 / 2 | 20 / 173 | iter19 / 6.9h | finite; stages a/b succeeded; no OOM |
| 2026-08-15 03:29 | Telecom | stage-final / 21 | 0.00% / 0 | 0.005919 | 4.17% | 13 / 133 | 69 / 1 | 21 / 45 | iter19 / 7.5h | finite; stages a/b succeeded; no OOM |
| 2026-08-15 04:30 | Airline | stage-final / 35 | 33.33% / 0 | 0.004022 | 0.00% | 26 / 225 | 105 / 14 | 25 / 418 | iter29 / 5.8h | finite; no OOM |
| 2026-08-15 04:30 | Retail | stage-final / 55 | 27.08% / -2.48e-09 | 0.006528 | 0.00% | 37 / 341 | 18 / 0 | 7 / 658 | iter49 / 2.6h | finite; no OOM |
| 2026-08-15 04:30 | Mixed | stage-final / 38 | 50.00% / -3.10e-09 | 0.004894 | 4.17% | 35 / 239 | 87 / 2 | 36 / 291 | iter29 / 5.2h | finite; no OOM |
| 2026-08-15 04:30 | Telecom | stage-final / 31 | 0.00% / 0 | 0.010319 | 4.17% | 37 / 199 | 94 / 2 | 32 / 57 | iter29 / 6.5h | finite; no OOM |
| 2026-08-15 05:38 | Airline | stage-final / 48 | 43.75% / -2.48e-09 | 0.008749 | 0.00% | 40 / 303 | 169 / 20 | 33 / 532 | iter39 / 4.6h | finite; no OOM |
| 2026-08-15 05:38 | Retail | stage-final / 75 | 56.25% / -3.73e-09 | 0.010150 | 0.00% | 59 / 454 | 27 / 0 | 10 / 842 | iter69 / 1.4h | finite; no OOM |
| 2026-08-15 05:38 | Mixed | stage-final / 54 | 12.50% / 4.97e-09 | 0.007946 | 6.25% | 56 / 335 | 127 / 6 | 48 / 412 | iter49 / 3.6h | finite; no OOM |
| 2026-08-15 05:38 | Telecom | stage-final / 42 | 20.83% / 0 | 0.019764 | 0.00% | 57 / 264 | 138 / 3 | 39 / 67 | iter39 / 5.6h | finite; no OOM |
| 2026-08-15 06:39 | Airline | interrupted / 50 | 31.25% / 5.59e-09 | 0.004859 | 0.00% | 41 / 314 | 178 / 20 | 33 / 550 | iter49 / interrupted | finite; no OOM; remote failure |
| 2026-08-15 06:39 | Retail | interrupted / 79 | 29.17% / 1.24e-09 | 0.009393 | 0.00% | 66 / 485 | 29 / 0 | 10 / 902 | iter79 / interrupted | finite; no OOM; remote failure |
| 2026-08-15 06:39 | Mixed | interrupted / 57 | 35.42% / -6.21e-10 | 0.008976 | 0.00% | 62 / 351 | 133 / 6 | 48 / 420 | iter49 / interrupted | finite; no OOM; remote failure |
| 2026-08-15 06:39 | Telecom | interrupted / 45 | 12.50% / 0 | 0.017537 | 0.00% | 70 / 284 | 150 / 3 | 42 / 70 | iter39 / interrupted | finite; no OOM; remote failure |
| 2026-08-15 09:30 | Airline | stage-final / 59 | 70.83% / -6.21e-09 | 0.008187 | 10.42% | 53 / 381 | 225 / 24 | 40 / 625 | iter59 / 3.8h | resumed; finite; no OOM |
| 2026-08-15 09:30 | Retail | stage-final / 93 | 68.75% / -6.21e-10 | 0.013041 | 0.00% | 87 / 574 | 35 / 0 | 13 / 1022 | iter89 / 0.5h | resumed; finite; no OOM |
| 2026-08-15 09:30 | Mixed | stage-final / 63 | 33.33% / 3.10e-09 | 0.007233 | 0.00% | 84 / 439 | 146 / 6 | 58 / 534 | iter59 / 2.5h | resumed; finite; no OOM |
| 2026-08-15 09:30 | Telecom | queued / 39 | pending | pending | pending | 70 / 284 | 150 / 3 | 42 / 70 | iter39 / pending | resume job 1945 queued |
| 2026-08-15 10:34 | Airline | stage-final / 68 | 22.92% / 6.21e-10 | 0.008862 | 0.00% | 64 / 439 | 272 / 31 | 43 / 749 | iter59 / 3.1h | finite; no OOM |
| 2026-08-15 10:34 | Retail | complete / 99 | 50.00% / -3.10e-09 | 0.012364 | 0.00% | 93 / 605 | 46 / 0 | 14 / 1077 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 10:34 | Mixed | stage-final / 74 | 22.92% / 6.21e-10 | 0.008140 | 4.17% | 103 / 503 | 176 / 8 | 63 / 602 | iter69 / 2.1h | finite; no OOM |
| 2026-08-15 10:34 | Telecom | stage-final / 44 | 29.17% / 1.86e-09 | 0.016957 | 0.00% | 86 / 318 | 160 / 3 | 45 / 75 | iter39 / 5.9h | resumed; finite; no OOM |
| 2026-08-15 11:34 | Airline | stage-final / 80 | 41.67% / 0 | 0.014060 | 2.08% | 84 / 509 | 328 / 39 | 46 / 837 | iter79 / 1.9h | finite; no OOM |
| 2026-08-15 11:34 | Retail | complete / 99 | 50.00% / -3.10e-09 | 0.012364 | 0.00% | 93 / 605 | 46 / 0 | 14 / 1077 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 11:34 | Mixed | stage-final / 87 | 41.67% / 1.86e-09 | 0.009642 | 2.08% | 129 / 585 | 207 / 8 | 74 / 707 | iter79 / 1.0h | finite; no OOM |
| 2026-08-15 11:34 | Telecom | stage-final / 56 | 4.17% / 1.24e-09 | 0.021630 | 0.00% | 132 / 392 | 184 / 3 | 63 / 84 | iter49 / 3.8h | finite; no OOM |
| 2026-08-15 12:37 | Airline | stage-final / 90 | 50.00% / -1.86e-09 | 0.014034 | 0.00% | 95 / 572 | 361 / 41 | 48 / 922 | iter89 / 0.9h | finite; no OOM |
| 2026-08-15 12:37 | Retail | complete / 99 | 50.00% / -3.10e-09 | 0.012364 | 0.00% | 93 / 605 | 46 / 0 | 14 / 1077 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 12:37 | Mixed | complete / 99 | 37.50% / 0 | 0.012817 | 2.08% | 146 / 652 | 230 / 8 | 80 / 788 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 12:37 | Telecom | stage-final / 68 | 0.00% / 0 | 0.019868 | 4.17% | 167 / 464 | 218 / 3 | 78 / 98 | iter59 / 2.7h | finite; no OOM |
| 2026-08-15 13:39 | Airline | complete / 99 | 33.33% / 1.24e-09 | 0.014369 | 2.08% | 108 / 626 | 385 / 42 | 50 / 1010 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 13:39 | Retail | complete / 99 | 50.00% / -3.10e-09 | 0.012364 | 0.00% | 93 / 605 | 46 / 0 | 14 / 1077 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 13:39 | Mixed | complete / 99 | 37.50% / 0 | 0.012817 | 2.08% | 146 / 652 | 230 / 8 | 80 / 788 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 13:39 | Telecom | stage-final / 77 | 20.83% / -1.86e-09 | 0.028099 | 6.25% | 195 / 520 | 264 / 6 | 94 / 108 | iter69 / 1.9h | finite; no OOM |
| 2026-08-15 14:38 | Airline | complete / 99 | 33.33% / 1.24e-09 | 0.014369 | 2.08% | 108 / 626 | 385 / 42 | 50 / 1010 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 14:38 | Retail | complete / 99 | 50.00% / -3.10e-09 | 0.012364 | 0.00% | 93 / 605 | 46 / 0 | 14 / 1077 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 14:38 | Mixed | complete / 99 | 37.50% / 0 | 0.012817 | 2.08% | 146 / 652 | 230 / 8 | 80 / 788 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 14:38 | Telecom | stage-final / 87 | 6.25% / 6.21e-10 | 0.046272 | 4.17% | 225 / 577 | 298 / 7 | 114 / 114 | iter79 / 1.0h | finite; no OOM |
| 2026-08-15 15:38 | Airline | complete / 99 | 33.33% / 1.24e-09 | 0.014369 | 2.08% | 108 / 626 | 385 / 42 | 50 / 1010 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 15:38 | Retail | complete / 99 | 50.00% / -3.10e-09 | 0.012364 | 0.00% | 93 / 605 | 46 / 0 | 14 / 1077 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 15:38 | Mixed | complete / 99 | 37.50% / 0 | 0.012817 | 2.08% | 146 / 652 | 230 / 8 | 80 / 788 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 15:38 | Telecom | stage-final / 96 | 22.92% / -2.48e-09 | 0.085977 | 4.17% | 267 / 631 | 298 / 7 | 140 / >=114 | iter89 / 0.3h | finite; no OOM; K2 elevated |
| 2026-08-15 16:38 | Airline | complete / 99 | 33.33% / 1.24e-09 | 0.014369 | 2.08% | 108 / 626 | 385 / 42 | 50 / 1010 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 16:38 | Retail | complete / 99 | 50.00% / -3.10e-09 | 0.012364 | 0.00% | 93 / 605 | 46 / 0 | 14 / 1077 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 16:38 | Mixed | complete / 99 | 37.50% / 0 | 0.012817 | 2.08% | 146 / 652 | 230 / 8 | 80 / 788 | iter99 / done | succeeded; checkpoint complete |
| 2026-08-15 16:38 | Telecom | complete / 99 | 18.75% / -2.48e-09 | 0.137748 | 2.08% | 278 / 649 | 298 / 7 | 142 / >=114 | iter99 / done | succeeded; K2 elevated but finite |

The stage-boundary tracebacks are W&B `atexit` connection-reset messages after
Ray reports each stage job as succeeded; the parent jobs continue into the next
stage. They are cleanup noise, not training failures.

At 04:30 the last-10 raw rewards were Airline `34.38%`, Retail `42.92%`,
Mixed `35.63%`, and Telecom `10.83%`; the corresponding last-10 truncation
rates were `0.63/0.63/2.71/1.67%`. Telecom's single-batch zero is therefore not
a ten-update collapse signal, though its K2 loss remains the largest and is
tracked as a diagnostic.

At 05:38 the last-10 raw rewards were Airline `41.25%`, Retail `45.83%`,
Mixed `32.29%`, and Telecom `16.25%`; last-10 truncation was
`1.67/0.42/1.88/1.25%`, and mean K2 loss was
`0.006459/0.008587/0.006881/0.015196`. All values remained finite; Telecom's
K2 loss continued upward and remains the main health diagnostic.

All four original cluster jobs stopped in the same second while generating
rollouts and were later reported as `REMOTE_FAILED`; their logs contain no
training exception, NaN, or OOM. Complete checkpoints at Airline iter49,
Retail iter79, Mixed iter49, and Telecom iter39 were retained. At 08:26, 18
superseded intermediate checkpoints were removed after approval, releasing
928 GB for resume checkpoints. Resume jobs 1894--1896 were queued for Airline,
Retail, and Mixed; Telecom submission waits for the 24-GPU queue limit to clear.

At 09:30, Airline, Retail, and Mixed had resumed normally. Their last-10 raw
rewards were `51.25/52.50/28.54%`, last-10 truncation was
`1.46/0.42/1.25%`, and mean K2 loss was `0.008221/0.011081/0.007143`.
Telecom resume job 1945 is queued behind the active 24-GPU allocation. A wider
checkpoint cleanup retained all active resume points and validated HF outputs
while increasing shared free space to 6.7 TB.

At 10:34, Retail job 1895 had succeeded and its iter99 checkpoint contained
all 12 distributed shards plus `common.pt` and metadata. Airline, Mixed, and
Telecom remained finite. Their four-arm last-10 raw rewards were
`51.46/57.92/32.71/16.25%`, last-10 truncation was
`1.46/0.42/0.83/1.25%`, and mean K2 loss was
`0.008704/0.011922/0.009051/0.017107`.

At 11:34, all parsed training metrics remained finite. Last-10 raw rewards
were `42.71/57.92/26.25/9.38%`, last-10 truncation was
`0.42/0.42/1.25/3.33%`, and mean K2 loss was
`0.011935/0.011922/0.011477/0.020455`. The only text match for `NaN` was an
Airline task instruction containing the literal word; parsed metrics had zero
non-finite values.

At 12:37, Mixed job 1896 had succeeded and its iter99 checkpoint was complete.
All four parsed metric streams still had zero non-finite values. Last-10 raw
rewards were `41.25/57.92/27.08/20.83%`, last-10 truncation was
`0.42/0.42/1.25/2.08%`, and mean K2 loss was
`0.012067/0.011922/0.011487/0.022258`.

At 13:39, Airline job 1894 had succeeded and its iter99 checkpoint was
complete. Last-10 raw rewards were `35.21/57.92/27.08/13.96%`, last-10
truncation was `0.42/0.42/1.25/3.33%`, and mean K2 loss was
`0.013731/0.011922/0.011487/0.025831`. Telecom remained finite while its K2
diagnostic continued upward.

At 14:38, Telecom's last-10 raw reward was `17.50%`, truncation `3.54%`, and
mean K2 loss `0.035590`; the latest K2 value was `0.046272`. All parsed values
remained finite. Literal out-of-memory text in a Telecom user utterance was not
a runtime OOM. Six corrected all-domain evaluation jobs for Mixed, Airline,
and Retail were running normally.

At 15:38, Telecom had reached step 96 and checkpoint iter89. Its last-10 raw
reward was `8.75%`, truncation `5.83%`, and mean K2 loss `0.049249`; the latest
K2 value was `0.085977`. All parsed values remained finite and no runtime OOM
or traceback appeared. The six all-domain evaluations remained healthy at
`271--304/400` completed simulations each. The execution-error trajectory
count in the row is a lower bound because the exact post-step87 total is not
emitted in aggregate form while the current rollout is live.

At 16:38, Telecom job 1945 had succeeded and its iter99 checkpoint contained
all 12 distributed shards plus `common.pt` and metadata. Its final last-10 raw
reward was `13.33%`, truncation `4.58%`, and mean K2 loss `0.063370`; step 99
K2 was `0.137748`. Training remained finite with no runtime OOM. Both Airline
all-domain evaluations succeeded, while the Mixed and Retail evaluations were
still finishing. Telecom conversion job 2129 subsequently succeeded and its
two all-domain evaluations were submitted as jobs 2140 and 2139.

At 18:55, Telecom expert all-domain evaluation jobs 2139 (seed301) and 2140
(seed300) were both running. Airline and Retail were complete in both jobs;
seed301 had completed `287/400` simulations (`47/160` Telecom), and seed300 had
completed `245/400` (`5/160` Telecom). Job-manager reported no failure. The
repeated LiteLLM model-cost mapping messages are non-fatal accounting warnings.

At 19:51, both Telecom expert evaluations remained healthy and running.
Seed301 had completed `385/400` simulations (`145/160` Telecom); seed300 had
completed `347/400` (`107/160` Telecom). Job-manager still reported no failure.
