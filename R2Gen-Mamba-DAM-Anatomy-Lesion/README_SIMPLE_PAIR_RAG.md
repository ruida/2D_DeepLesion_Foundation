# Simple Pair RAG

Project: `/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text`

Run preprocessing:
```bash
chmod +x simple_pair_rag_pipeline/*.sh
./simple_pair_rag_pipeline/run_all_preprocessing.sh
```
Train:
```bash
./simple_pair_rag_pipeline/train_oracle_simple_r2gen.sh
```
Test:
```bash
./simple_pair_rag_pipeline/test_oracle_simple.sh
./simple_pair_rag_pipeline/test_raw_simple.sh
./simple_pair_rag_pipeline/test_simple_pair_rag.sh
```
