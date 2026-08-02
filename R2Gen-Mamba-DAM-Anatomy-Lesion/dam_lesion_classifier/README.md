# DAM lesion classifier

Copy this directory into `/data/ruida/LLM/R2Gen-DAM/dam_lesion_classifier`.

It reuses the repository's `VisualExtractor` and `GatedGlobalLocalAdapter` and exactly follows the DAM global/focal RGB+mask representation.

Run:

```bash
cd /data/ruida/LLM/R2Gen-DAM
bash dam_lesion_classifier/run_prepare_and_train.sh
```

Outputs:
- `results/dam_lesion_classifier_oracle04/best.pt`
- `results/dam_lesion_classifier_oracle04/test_metrics.json`
- console classification reports for primary lesion and attribute classes.
