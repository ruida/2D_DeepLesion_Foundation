# Simplified Oracle-Pair RAG Pipeline

## New directories

- Code: `oracle_pair_rag_pipeline/`
- Generated annotations: `data/oracle_pair_rag_annotations/`
- Metrics/results: `results/oracle_pair_rag/`

This pipeline does **not** retrieve reports and does **not** retrain R2Gen-Mamba.
It retrieves/corrects only the oracle-style pair:

`rough TotalSeg anatomy + crop-classifier lesion type -> oracle anatomy + oracle lesion type`

The frozen high-performing oracle anatomy+lesion R2Gen-Mamba checkpoint is then tested using the mapped pair.

## Stage 0: install the new code directory

From the project root:

```bash
cp -r /path/to/oracle_pair_rag_pipeline .
mkdir -p data/oracle_pair_rag_annotations results/oracle_pair_rag logs
```

## Stage 1: inspect the source JSON

The default source is:

```text
data/generated_annotations/rag_ready_statmapped_anatomy_predicted_lesion/deeplesion_rag_ready_statmapped_anatomy_predicted_lesion_retrained8.json
```

It should have `train`, `val`, and `test` arrays. The preparation script auto-detects the likely rough-anatomy, predicted-lesion, and oracle target fields and prints the first canonical sample.

## Stage 2: run preparation and gated/agent RAG

```bash
./oracle_pair_rag_pipeline/run_pair_rag.sh
```

Outputs:

```text
data/oracle_pair_rag_annotations/01_pair_rag_canonical.json
data/oracle_pair_rag_annotations/02_pair_rag_oracle_style_r2gen.json
results/oracle_pair_rag/summary.json
results/oracle_pair_rag/validation_grid.json
```

The retrieval memory is built from **train only**. Validation selects gate thresholds. Test never contributes to retrieval or threshold tuning.

## Retrieval and gating logic

The structured similarity score is:

- 0.45: rough anatomy exact match
- 0.30: predicted lesion type exact match
- 0.25: Jaccard overlap of detailed TotalSeg names

The rule agent tries, in order:

1. high-support exact query lookup;
2. gated joint-pair top-k weighted vote;
3. partial anatomy and/or lesion correction;
4. raw-pair fallback.

Validation optimizes joint oracle-pair accuracy.

## Stage 3: verify improvement

```bash
cat results/oracle_pair_rag/summary.json
```

Compare, especially on validation:

- `raw.anatomy_accuracy`
- `raw.lesion_accuracy`
- `raw.joint_accuracy`
- `after_rag.anatomy_accuracy`
- `after_rag.lesion_accuracy`
- `after_rag.joint_accuracy`

Proceed to report generation only when validation joint accuracy improves.

## Stage 4: test the existing oracle-trained R2Gen model

Find the checkpoint used by the high-performing oracle anatomy+lesion experiment, then run:

```bash
CHECKPOINT=results/YOUR_ORACLE_MODEL/current_checkpoint.pth \
./oracle_pair_rag_pipeline/test_pair_rag_with_oracle_model.sh
```

The test annotation is:

```text
data/oracle_pair_rag_annotations/02_pair_rag_oracle_style_r2gen.json
```

The mapped fields include:

```json
{
  "pair_rag_mapped_anatomy": "abdomen",
  "pair_rag_mapped_lesion_type": "lymph node",
  "pair_rag_method": "joint_rag",
  "rough_anatomy_names": ["abdomen", "lymph node"]
}
```

## Optional Slurm run

The RAG mapping itself is CPU work and does not require a GPU, but the included Slurm template follows the current project convention:

```bash
sbatch oracle_pair_rag_pipeline/train_pair_rag.slurm
```

For report testing, use the same GPU allocation and module settings as the prior successful oracle model test job.

## Important leakage controls

- Retrieval database: train only.
- Gate tuning: validation only.
- Test oracle labels: used only for offline metric reporting, never retrieval or tuning.
- Existing R2Gen checkpoint remains frozen.
- No retrieved report text enters the model.
