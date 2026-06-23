# Runs

Task lineage records: what was requested, what was retrieved, what was tried,
what was validated, and whether the toolbox should change.

Create one for meaningful tasks:

```bash
python3 scripts/yax.py new-run <slug>
```

Then fill in the outcome and validation evidence. Runs are the evidence trail
that lets future similar tasks reuse and improve prior work.
