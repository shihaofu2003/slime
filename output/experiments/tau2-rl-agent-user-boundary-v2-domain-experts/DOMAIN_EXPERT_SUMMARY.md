# Domain expert summary

| Domain | Selected | Original classification | OPD eligible |
|---|---:|---|---|
| airline | iter129 | reject | false |
| retail | iter119 | advantage | true |
| telecom | iter129 | advantage | true |

No OPD or distillation job is started by this experiment.

Retail and Telecom are teacher candidates. Airline is not selected because its
two-seed target-domain comparison is weaker than mixed RL; this is a useful
negative-transfer result, not an invalid checkpoint.
