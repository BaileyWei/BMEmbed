# BMEmbed: Private Embedding Adaptation Pipeline

BMEmbed is an **unsupervised pipeline** for adapting general-purpose text embedding models to **private or proprietary datasets**. It leverages **BM25-based ranking signals** as pseudo-supervision to fine-tune models in a **label-free** manner, making it ideal for **enterprise-specific information retrieval tasks**.

## Overview of the Pipeline

The training pipeline consists of three main stages:

1. **Synthetic Query Generation**
   - Uses OpenAI GPT models to generate **event-based structured queries** from private corpora.
   - Saves structured queries and their evidence.

2. **Data Sampling via BM25-Based Ranking Score**
   - BM25 is used to rank **retrieved document chunks** for each query.
   - Samples **training data** for fine-tuning the embedding model.

3. **Model List-wise Fine-Tuning**
   - Fine-tunes the embedding model using **listwise ranking loss**.

**Evaluation Script**
   - Evaluates the model on retrieval performance (if have evaluation dataset).

---

## Installation

**Requirements:**
- Python 3.9+
- PyTorch
- Transformers (Hugging Face)
- Accelerate

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Synthetic Query Generation

```bash
python private_data_processing/corpus2chunks.py \
  --base_model qwen-7b \
  --corpus_file_path ./data/sythetic_data/multihop-rag/raw_corpus.json \
  --dataset_name multihop-rag # chunk original corpus
```
This script:

- Tokenizes text into sentences while maintaining structure.
- Splits documents into overlapping chunks (default: max_length=256).
- Saves the output in `./data/sythetic_data/multihop-rag/chunked_corpus.json`.
```bash
python private_data_processing/corpus2qa_processing.py \
  --corpus_file_path ./data/sythetic_data/multihop-rag/chunked_corpus.json \
  --dataset_name multihop-rag # generate synthetic query
  ```
This script:

- Extracts structured events from raw corpora using GPT.
- Generates event-based queries.
- Saves structured query-evidence pairs for subsequent training in `./data/sythetic_data/multihop-rag/doc2query.json`.



### Step 2. Data Sampling via BM25-Based Ranking Score

1. Make sure your **raw data** (e.g., JSON or text files) is accessible to the script.
2. Run the following command to generate a score-labeled dataset:

   ```bash
    python ranking_sampling/data_sampler.py \
      --dataset multihop-rag \
      --topk 1000 \
      --strategy fine-to-coarse \
      --interval_multiplier 1.2 \
      --num_samples 1

   ```
This script:

- Retrieves top-k ranked candidates for each query using BM25.
- Adjust the sampling strategy via args `--strategy`, `--num_samples`, `--topk`, and `--interval_multiplier`
- Saves the queries and candidates' scores in `./data/sythetic_data/multihop-rag/{timestamp}/bm25_dataset.json`.
### Step 3. Model List-wise Fine-Tuning
With your newly generated data in place, fine-tune your model:


```bash
    accelerate launch listwise_finetuning/qwen_trainer.py \
  --model_name_or_path qwen-7b \
  --dataset_name multihop-rag \
  --dataset_file_path ./data/sythetic_data/multihop-rag/{timestamp}/ \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-5 \
  --max_seq_length 256 \
  --bf16 \
  --gradient_accumulation_steps 8 \
  --listwise \
  --label_scaling 0.2 \
  --output_dir ./output/multihop-rag/
      
  ```
Your trained model is saved in `./output/multihop-rag/`.

### Step 4. Evaluation (Use your own dataset with queries and answers)
After training completes, evaluate your model with:

```bash
        python simple_retrieval.py \
          --base_model qwen-7b \
          --dataset_name multihop-rag \
          --dataset_file_path ../data/sythetic_data/multihop-rag/{timestamp} \
          --cache_dir ./ \
          --output_dir ./output/multihop-rag/ \
          --listwise \
          --save_retrieval_result \
          --save_embedding_database \
          --label_scaling 0.2
   ```
### Repository Structure
```
├── .gitignore
├── README.md
├── data
│   ├── evaluation_set
│   │   └── multihop-rag
│   │       └── retrieval_dataset.json
│   └── sythetic_data
│       └── multihop-rag
│           ├── chunked_corpus.json
│           └── doc2query.json
├── listwise_finetuning
│   ├── accelerate_config.yaml
│   ├── all_mini_l6_v2_trainer.py
│   ├── lora.json
│   ├── loss.py
│   ├── models
│   │   ├── __init__.py
│   │   └── qwen_embedding.py
│   └── qwen_trainer.py
├── private_data_processing
│   ├── corpus2chunks.py
│   ├── corpus2qa_processing.py
│   ├── prompts.py
│   └── utils.py
├── ranking_sampling
│   ├── bm25_retrieval.py
│   ├── data_sampler.py
│   ├── data_sampler_scores.py
│   └── partitioning_strategy.py
├── requirements.txt
├── retrieval_evaluating.py
└── simple_retrieval.py
```