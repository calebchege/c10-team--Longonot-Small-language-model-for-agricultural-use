# SLM for Agriculture and Climate

Fine-tuning a small language model (SLM) to generate concise, factsheet-grounded answers to smallholder farmers' agricultural questions, built for the [Agriculture & Climate SLM Challenge](https://www.kaggle.com/competitions/agriculture-climate-slm-challenge) on Kaggle. To access the data click on the link and accept terms and conditions after reviewing them.

## Dataset

The dataset was provided by the competition and consists of three files:

- **`documents.csv`** (24 rows): a corpus of short (~250-character) extension factsheets, each tagged with `topic` (8 categories: crop diseases, pests, soil health, climate adaptation, water management, fertiliser, livestock, post-harvest), `crop` (7 categories: maize, beans, cassava, sorghum, groundnuts, livestock, general), and `agro_zone` (semi-arid, sub-humid, highland). All 24 documents are synthetic (`origin = synthetic`), licensed CC0-1.0, with no `source_url`.
- **`train_qa.csv`** (45 rows): farmer-style questions, each linked to its source `document_id`, with a short (mean ~10 words) `reference_answer` that paraphrases and compresses specific clauses from the linked document rather than copying it verbatim.
- **`test_questions.csv`** (12 rows): held-out questions with `topic`/`crop`/`agro_zone` metadata but no `document_id`.

EDA confirmed that `(topic, crop, agro_zone)` reliably maps each question to its correct source document — verified against `train_qa`'s ground-truth `document_id` (one ambiguous combination out of 45 train rows) and confirmed unique for all 12 test rows. This informed the core design decision: retrieve the matched document via metadata lookup, then fine-tune the model to compress it into a short, grounded answer, rather than relying on open-book retrieval or closed-book generation alone.

## Training Pipeline

**Data preparation:** Each training example was built as `(prompt, target)`, where the prompt combines an instruction, the matched document's full text, and the question, and the target is the `reference_answer`. Prompts were tokenized with the Gemma-2 tokenizer; loss was masked over the prompt tokens so the model only learns to predict the answer.

**Model and fine-tuning:** We fine-tuned **Gemma-2-2b-it** using **LoRA** (`r=16`, `alpha=32`, `dropout=0.05`, targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`), chosen over Phi-3-mini for its smaller size relative to the very limited (45-example) training set, reducing overfitting risk and speeding iteration within Kaggle's GPU limits. Training ran in `bf16` on a single T4 GPU (`CUDA_VISIBLE_DEVICES` pinned to avoid `device_map="auto"` splitting the model across both available GPUs, which conflicted with `Trainer`'s multi-GPU handling), with gradient checkpointing enabled to manage memory. Final hyperparameters: batch size 1 with gradient accumulation of 8 (effective batch size 8), learning rate `2e-4`, 8 epochs.

**Key design choices:** No held-out validation split was used, given how few labeled examples (45) were available; instead, we prioritized inspecting qualitative predictions on both train and test examples to check for overfitting and hallucination before finalizing.

## Evaluation

The competition scores submissions by **mean Levenshtein distance** between each generated answer and a hidden reference answer across the 12 test questions (lower is better), benchmarked against a topic-filtered TF-IDF baseline (~54 mean Levenshtein on v1).

Since no labeled test set was available for direct scoring, we validated the pipeline in two stages:
1. **Training-set inspection:** After fine-tuning, generated answers on sampled training questions matched their references closely, confirming the model learned the intended compression task.
2. **Test-set qualitative review:** Generated answers on all 12 test questions were manually reviewed for specificity, correct grounding in the matched document, and appropriate length (~8-14 words), rather than generic or off-topic phrasing. All 12 answers passed format validation (correct `QuestionId`/`Answer` columns, preserved row order, non-empty text).

## Reproduction

Run the following in a Kaggle Notebook with the competition data attached and GPU enabled:

1. **Setup:** Pin GPU visibility (`os.environ["CUDA_VISIBLE_DEVICES"] = "0"`) before any other imports; install/upgrade `bitsandbytes` and `torchao` if versions are outdated.
2. **Load data:** Read `documents.csv`, `train_qa.csv`, `test_questions.csv` from `/kaggle/input/competitions/agriculture-climate-slm/`.
3. **Build prompts:** Merge `train_qa` with `documents` on `document_id`; construct the instruction-style prompt template combining factsheet text and question.
4. **Load model:** Load `gemma-2-2b-it` (Transformers framework) in `bf16`, pinned to a single GPU (`device_map={"": 0}`); attach a LoRA adapter via `peft`.
5. **Tokenize:** Tokenize prompts and targets with loss masking over the prompt portion.
6. **Train:** Fine-tune with `Trainer`/`TrainingArguments` using the hyperparameters above.
7. **Generate:** For each test question, match its `(topic, crop, agro_zone)` to the correct document, build the same prompt template, and generate an answer.
8. **Submit:** Write predictions to `/kaggle/working/submission.csv` with columns `QuestionId,Answer`, validate row count/order/non-empty text, then **Save Version → Save & Run All** and submit the committed notebook's output.

## Appendix

**Contributors:**
- Janet Kisee
- Caleb Chege
- Winnie Chikhwaya
- Anita Otoo

**Mentors:**
- Samuel Taiwo
- John Evans Okyere
