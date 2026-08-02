#!/bin/bash
set -euo pipefail

PROJECT=${PROJECT:-/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text}
CODE_ROOT=${CODE_ROOT:-oracle_pair_rag_pipeline}
DATA_ROOT=${DATA_ROOT:-data/oracle_pair_rag_annotations}
RESULT_ROOT=${RESULT_ROOT:-results/oracle_pair_rag}

# This existing combined file already contains train/val/test rough anatomy,
# predicted lesion type, oracle targets, image paths, boxes, and reports.
SOURCE_JSON=${SOURCE_JSON:-data/generated_annotations/rag_ready_statmapped_anatomy_predicted_lesion/deeplesion_rag_ready_statmapped_anatomy_predicted_lesion_retrained8.json}

cd "$PROJECT"
mkdir -p "$DATA_ROOT" "$RESULT_ROOT" logs

CANONICAL="$DATA_ROOT/01_pair_rag_canonical.json"
FINAL="$DATA_ROOT/02_pair_rag_oracle_style_r2gen.json"

python "$CODE_ROOT/01_prepare_pair_rag_dataset.py" \
  --input_json "$SOURCE_JSON" \
  --out_json "$CANONICAL" \
  --allow_test_oracle_for_evaluation

python "$CODE_ROOT/02_train_validate_apply_gated_pair_rag.py" \
  --input_json "$CANONICAL" \
  --out_json "$FINAL" \
  --out_dir "$RESULT_ROOT"

echo "Final R2Gen-ready JSON: $FINAL"
echo "Metrics: $RESULT_ROOT/summary.json"
