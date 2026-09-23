# Tau2 OPD status

Updated2026-09-23:the user fixed the selected recipe and submitted four-domain
full distillation. Old experts23045 are
STOPPED; replacement8-GPU experts23191 are RUNNING with all four endpoints ready.
User23043 remains running. The
[four-domain pipeline](../experiments/tau2-opd-four-domain-20260923/README.md)
was submitted as8-GPU job23193 at00:59 CST and is RUNNING:2524 tasks,160 updates at batch16/K1,iter159 HF,
then official four-domain seeds300/301 evaluation. The160-update budget corrects
the initial60-step pilot proposal after the user clarified full-data coverage.
Startup passed20 OPD +119 continuous tests and entered `train_async.py` model initialization.

The completed [bounded LR study](../experiments/tau2-opd-bounded-lr-20260922/README.md):
LR2e-6/5e-6 jobs23062/23049 both SUCCEEDED:60 updates,iter59 HF export and
seeds300/301 official evaluation. All1,600 new simulations completed with zero
infrastructure errors. The replacement User/expert services remain persistent;
keep them running after subsequent training/evaluation. Fixed-context scoring
remains deferred.
Original LR2e-6 job23048 stopped before training on an ordering-dependent unit
test; its fixture was corrected and23062 completed from the still-fresh root.

## Current recipe and findings

SFT4505 student,Airline iter29/Retail iter9 teachers,current-student logprob
advantage plus TIS,LR2e-6,batch16,K1,60 updates,buffer32. Current candidate is
`tau2-opd-bounded-lr-20260922/pilot-lr2e6-seed1235-43/checkpoints/iter_0000059_hf`.
The paired LR study used common train/rollout seeds1235/43. Both arms had finite
losses/gradients,mean policy lag1.02/1.03 and maximum admitted groups32.

Current2e-6 versus5e-6 pass@1:Airline50.00/43.75%,Retail60.3125/55.625%,
Telecom52.50/46.875%,overall55.125/49.75%. Overall difference+5.375pp has
task-paired95% CI[+1.50,+9.375]; both evaluation seeds favor2e-6 in every domain.
Keep2e-6 for subsequent work. Overall pass^4 is25.00/24.50%,without a clear gap.

Retail2e-6 pass^4 is33.75% in both independent training runs (seeds1234/42 and
1235/43),versus SFT22.50%. The latest increase has CI[+3.75,+18.75]pp.
Airline2e-6 pass@1 exceeds its teacher point estimate45.625%,but the difference
CI[-0.625,+10.00] includes zero. Airline pass^4 remains25.00%,equal to the teacher
and below SFT32.50%; consistency remains an unresolved limitation.

Airline teacher pass@1 exceeds matched SFT by only1.25pp,CI[-6.25,+8.75].
The historical four-domain seed300 teacher score53.75% is not a fixed target
for this matched two-seed comparison. Teacher quality and write-action signal
now deserve attention alongside OPD optimization.

[Current results and next-run recommendations](../experiments/tau2-opd-bounded-lr-20260922/RESULTS.md),
[previous buffer A/B results](../experiments/tau2-opd-current-buffer-ab/RESULTS.md).

## Interpretation limits

The LR comparison has one paired training/rollout seed; the two2e-6 training
runs differ by5.375pp in overall pass@1. Two evaluation seeds do not substitute
for independent training repeats. Bootstrap intervals cluster by task,keeping
both evaluation seeds/trials together; they do not measure training-seed variance
and are not corrected for multiple comparisons.

In the older buffer A/B study,B's job22931 was interrupted by the user-confirmed cluster crash. Recovery22970
restored iter49 and sampler state; pending trajectories were regenerated and
updates50–59 had mean lag2.69. B therefore is not an uninterrupted unbounded
control. A evaluation was elastically requeued22972→22979 before producing trials.
All4,000 older simulations completed. A controls lag,but A/B task-score superiority
was not established.

The earlier behavior-logprob v3 surrogate is opt-in only. With stale behavior
and unchanged TIS it is not the current-student reverse-KL gradient; the
[code review](../experiments/tau2-opd-airline-retail-pilot/REVIEW_20260922.md)
contains the numerical counterexample. Negative unweighted log-ratio logs alone
do not demonstrate a wrong TIS-weighted gradient.

## Next work

The submitted four-domain run starts from SFT4505 with Airline29/Retail9/Telecom9/
Banking9 teachers,LR2e-6,buffer32,train/rollout seeds1235/43 and160 updates.
This expands both source domains and trajectory budget,so it is not a domain-only
ablation against the60-step pilot. Record actual consumed-domain counts and
unique-task coverage. Four-domain evaluation uses the expert protocol with BM25
and concurrency1/2/2/4; existing four-domain seed300 controls are available,
while seed301 controls remain unevaluated. Do not substitute the three-domain
controls as a matched comparison. Historical LR uncertainty and Airline
consistency limitations still apply. Fixed-context scoring remains deferred.
