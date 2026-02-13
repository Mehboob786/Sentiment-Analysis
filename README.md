# Sentiment-Aware Customer Support Chatbot

A text-based customer support chatbot that combines:
- Rule-based response handling for support intents
- Machine learning sentiment classification (`negative`, `neutral`, `positive`)
- A Streamlit web UI for interactive chatting

This project is designed as a practical NLP baseline that is easy to extend into more advanced conversational AI or emotion detection work.

## 1. Project Goals

- Detect sentiment in each incoming user message
- Generate a support response based on both:
  - message intent keywords (billing, delivery, technical issues)
  - predicted sentiment and confidence score
- Provide an interactive web interface for demo and evaluation

## 2. Tech Stack

- Python 3.10+
- scikit-learn (TF-IDF + Logistic Regression)
- Hugging Face `datasets` (for public labeled sentiment data)
- pandas
- joblib
- Streamlit

## 3. Current Repository Structure

```text
Sentiment-Analysis/
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   └── sentiment_model.joblib
└── src/
    ├── app.py
    ├── infer.py
    ├── prepare_data.py
    ├── responder.py
    └── train.py
```

Note:
- `data/` and `models/` are generated after running preparation and training scripts.

## 4. End-to-End Workflow

Run in this exact order:

1. Create/activate virtual environment and install dependencies
2. Download + convert dataset
3. Train model
4. Launch Streamlit app

## 5. Setup Instructions

### 5.1 Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 5.2 Install dependencies

```bash
pip install -r requirements.txt
```

## 6. Data Preparation

Script: `src/prepare_data.py`

What it does:
- Downloads dataset: `cardiffnlp/tweet_eval` with config `sentiment`
- Converts numeric labels to string labels:
  - `0 -> negative`
  - `1 -> neutral`
  - `2 -> positive`
- Writes local CSV files:
  - `data/train.csv`
  - `data/val.csv`
  - `data/test.csv`

Run:

```bash
python src/prepare_data.py
```

Expected terminal output (shape counts may vary only if dataset changes):

```text
Saved:
- data/train.csv (45615 rows)
- data/val.csv (2000 rows)
- data/test.csv (12284 rows)
```

## 7. Model Training and Evaluation

Script: `src/train.py`

### 7.1 Model pipeline

- Vectorizer: `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)`
- Classifier: `LogisticRegression(max_iter=1200, class_weight='balanced')`
- Wrapped in a scikit-learn `Pipeline`

### 7.2 Inputs

Reads:
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

Each CSV must have columns:
- `text`
- `label`

### 7.3 Outputs

Prints:
- validation classification report
- validation confusion matrix
- test classification report
- test confusion matrix

Saves model:
- `models/sentiment_model.joblib`

Run:

```bash
python src/train.py
```

## 8. Rule-Based Response Logic

Script: `src/responder.py`

The chatbot first checks intent-style keywords:
- Billing: `refund`, `charge`, `billing`, `invoice`, etc.
- Delivery: `late`, `delay`, `shipping`, `tracking`, etc.
- Technical: `error`, `bug`, `not working`, `crash`, etc.

If no intent keyword is matched:
- Uses sentiment and confidence
- If confidence is low (`< 0.45`), asks for clarification
- Otherwise responds with sentiment-aware support tone

This makes behavior more practical for customer support than sentiment-only responses.

## 9. Streamlit Chat Application

Script: `src/app.py`

Features:
- Loads trained model from `models/sentiment_model.joblib`
- Accepts user input via form
- Predicts sentiment + confidence for each message
- Calls rule-based responder
- Displays conversation history
- Renders newest messages at the top (recent-first order)

Run:

```bash
streamlit run src/app.py
```

After running, open the local URL shown in terminal (usually `http://localhost:8501`).

Important:
- Do not run UI with `python src/app.py`
- `app.py` must be launched with `streamlit run ...` for session state and UI context to work correctly

## 10. CLI Inference Mode (Optional)

Script: `src/infer.py`

Run:

```bash
python src/infer.py
```

Usage:
- Type messages in terminal
- Get `sentiment + confidence`
- Type `quit` to exit

## 11. Command Reference (Quick Copy)

```bash
# from project root
source .venv/bin/activate
python src/prepare_data.py
python src/train.py
streamlit run src/app.py
```

## 12. Troubleshooting

### 12.1 Streamlit warning: missing `ScriptRunContext`

Symptom:
- warnings like `missing ScriptRunContext` or
- `Session state does not function when running a script without streamlit run`

Cause:
- launching app with `python src/app.py`

Fix:

```bash
streamlit run src/app.py
```

### 12.2 Model file missing

Symptom:
- error in app: model file missing

Fix:

```bash
python src/prepare_data.py
python src/train.py
```

### 12.3 Dataset download issues

Symptom:
- errors during `load_dataset(...)`

Fix:
- verify internet access
- retry command
- ensure dependencies installed:

```bash
pip install -r requirements.txt
```

### 12.4 Import errors from Streamlit app

If running from root with `streamlit run src/app.py`, local imports should work.
If you run commands from another directory, switch back to project root first.

## 13. Reproducibility Notes

- Model behavior can vary slightly across dependency versions
- Keep `requirements.txt` pinned as needed for strict reproducibility
- You can add a random seed if you introduce stochastic training steps later

## 14. How to Extend This Project

### 14.1 Better NLP/modeling

- Replace TF-IDF with transformer embeddings
- Fine-tune a pretrained sentiment model
- Add class probability thresholds per class

### 14.2 Better support behavior

- Add more intent categories (returns, account, subscription, outage)
- Maintain multi-turn context and slot filling
- Integrate escalation logic to human support

### 14.3 Better evaluation

- Add held-out domain-specific support dataset
- Track macro-F1 and per-class confusion trends
- Save error analysis examples (sarcasm, mixed sentiment)

### 14.4 Deployment

- Deploy Streamlit app on Streamlit Community Cloud or Render
- Add lightweight API layer (FastAPI/Flask) if needed

## 15. File-by-File Summary

- `src/prepare_data.py`: pulls public sentiment dataset and writes local CSVs
- `src/train.py`: trains TF-IDF + Logistic Regression pipeline and evaluates it
- `src/infer.py`: terminal-based sentiment prediction loop
- `src/responder.py`: rule-based support response engine with confidence fallback
- `src/app.py`: Streamlit chatbot interface and message rendering

## 16. Suggested Next Improvements for Professor Demo

1. Add chat transcripts export (`.csv`/`.json`) for analysis
2. Add confidence color badges in UI
3. Add profanity/frustration detection layer
4. Add a small custom labeled customer-support dataset and compare metrics before/after fine-tuning

## 17. Results (Current Run)

Values below are from the current local run of `python src/train.py` on February 13, 2026.

### 17.1 Dataset Summary

| Split | Rows | Source |
|---|---:|---|
| Train | `45,615` | `cardiffnlp/tweet_eval` |
| Validation | `2,000` | `cardiffnlp/tweet_eval` |
| Test | `12,284` | `cardiffnlp/tweet_eval` |

### 17.2 Model Configuration

| Component | Setting |
|---|---|
| Vectorizer | `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)` |
| Classifier | `LogisticRegression(max_iter=1200, class_weight='balanced')` |
| Labels | `negative`, `neutral`, `positive` |

### 17.3 Validation Metrics

| Metric | Score |
|---|---:|
| Accuracy | `0.6620` |
| Macro Precision | `0.6339` |
| Macro Recall | `0.6632` |
| Macro F1 | `0.6415` |

Per-class (validation):

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | `0.4600` | `0.6635` | `0.5433` | `312` |
| Neutral | `0.6889` | `0.6191` | `0.6521` | `869` |
| Positive | `0.7529` | `0.7070` | `0.7292` | `819` |

### 17.4 Test Metrics

| Metric | Score |
|---|---:|
| Accuracy | `0.5957` |
| Macro Precision | `0.5883` |
| Macro Recall | `0.6020` |
| Macro F1 | `0.5924` |

Per-class (test):

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | `0.5598` | `0.6589` | `0.6053` | `3,972` |
| Neutral | `0.6477` | `0.5550` | `0.5978` | `5,937` |
| Positive | `0.5575` | `0.5920` | `0.5742` | `2,375` |

### 17.5 Confusion Matrix

Label order used in code: `negative`, `neutral`, `positive`.

Validation confusion matrix:

```text
[[207, 74, 31],
 [172, 538, 159],
 [71, 169, 579]]
```

Test confusion matrix:

```text
[[2617, 1112, 243],
 [1769, 3295, 873],
 [289, 680, 1406]]
```

### 17.6 Error Analysis (Short)

- Common false positives: negative text often predicted as neutral; neutral text often predicted as negative.
- Common false negatives: positive class confusion with neutral and vice versa.
- Hard examples: sarcasm, mixed-sentiment sentences, short context-poor messages.
- Planned fix: fine-tune on customer-support-specific labeled data and add confidence-based fallback handling.

### 17.7 Example Chats

Use `python src/infer.py` or the Streamlit UI to generate live examples and record exact confidence scores for your report.

## 18. License

Add a license file if required by your course or lab policy.
