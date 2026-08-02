# Simple dual-head lesion classifier

1. Build dataset:
`python simple_dualhead_lesion_classifier/00_build_dualhead_dataset.py`

2. Train:
`sbatch simple_dualhead_lesion_classifier/train_dualhead_classifier.slurm`

3. Predict:
`sbatch simple_dualhead_lesion_classifier/predict_dualhead_classifier.slurm`

4. New RAG input:
`data/simple_pair_rag_annotations/01b_raw_simple_inputs_dualhead.json`
