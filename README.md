---

````markdown
# Deploying a Scalable Machine Learning Pipeline with FastAPI

This project builds and deploys a complete machine learning pipeline for predicting income levels using the U.S. Census dataset. The pipeline includes model training, automated testing, continuous integration, and API deployment.  
The project follows modern MLOps practices and demonstrates reproducibility, automation, and maintainability.

---

## 📦 Table of Contents
- [Project Overview](#-project-overview)
- [Environment Setup](#-environment-setup)
- [Continuous Integration](#-continuous-integration-ci)
- [Project Structure](#-project-structure)
- [Model Summary](#-model-summary)
- [API Usage](#-api-usage)
- [Local API Testing](#-local-api-testing)
- [Model Card](#-model-card)
- [Rubric Alignment Summary](#-rubric-alignment-summary)
- [Author](#author)

---

## 📦 Project Overview

The goal is to develop a reusable ML pipeline that can:
1. Train and evaluate a model on tabular census data.  
2. Automate tests and linting through GitHub Actions.  
3. Serve predictions in real time through a FastAPI REST API.  
4. Include slice-based model performance analysis and documentation.

This end-to-end solution reflects real-world ML DevOps workflows — from data ingestion to production inference.

---

## ⚙️ Environment Setup

You can create the environment in two ways:

**Option 1 – Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate fastapi310
````

**Option 2 – pip:**

```bash
pip install -r requirements.txt
```

Then verify setup:

```bash
python --version
pytest -v
flake8 .
```

---

## 🧪 Continuous Integration (CI)

The CI pipeline is implemented with **GitHub Actions** in `.github/workflows/manual.yml`.
Each push or manual trigger runs:

* `flake8` linting for style and syntax compliance
* `pytest` for unit test validation
* Python 3.10 setup consistency check

✅ A screenshot of the passing CI run is included in the `screenshots/` folder.

---

## 🧩 Project Structure

```
Deploying-a-Scalable-ML-Pipeline-with-FastAPI/
├── data/                     # Census dataset
├── ml/
│   ├── data.py               # Data processing functions
│   ├── model.py              # Model training and inference logic
├── model/                    # Trained model & encoders
├── screenshots/              # CI and testing screenshots
│   ├── continuous_integration.png
│   ├── unit_test.png
│   └── local_api.png
├── test_ml.py                # Unit tests for core ML functions
├── train_model.py            # End-to-end training and slice analysis
├── main.py                   # FastAPI app for inference
├── local_api.py              # Client script calling the API
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🧠 Model Summary

The model uses scikit-learn’s `RandomForestClassifier` trained on the **Census Income dataset** to predict whether a person earns more than $50K/year.

Performance metrics on the test set:

| Metric    | Score |
| :-------- | :---: |
| Precision | ~0.74 |
| Recall    | ~0.64 |
| F1 Score  | ~0.68 |

Slice-based results are logged in `slice_output.txt` to analyze model fairness across categorical groups such as race, gender, and occupation.

---

## 🚀 API Usage

Run the FastAPI app locally:

```bash
uvicorn main:app --reload
```

**Endpoints:**

| Method | Endpoint | Description                                |
| :----- | :------- | :----------------------------------------- |
| GET    | `/`      | Returns a welcome message                  |
| POST   | `/data/` | Performs inference on a single data record |

Example JSON payload:

```json
{
  "age": 37,
  "workclass": "Private",
  "fnlgt": 178356,
  "education": "HS-grad",
  "education-num": 10,
  "marital-status": "Married-civ-spouse",
  "occupation": "Prof-specialty",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}
```

The API returns:

```json
{"result": "<=50K"}
```

---

## 🧭 Local API Testing

The `local_api.py` script sends both GET and POST requests to verify live inference.
A successful run produces:

```
Status Code: 200
Result: {'message': 'Welcome! Census Income API is live.'}
Status Code: 200
Result: {'result': '<=50K'}
```

A screenshot (`local_api.png`) is included under `screenshots/`.

---

## 📈 Model Card

A model card template (`model_card_template.md`) is included to document model purpose, assumptions, ethical considerations, and performance across data slices.

---

## ✅ Rubric Alignment Summary

* **Code quality:** modular, well-commented, passes flake8
* **Testing:** ≥3 unit tests implemented and automated
* **Pipeline:** complete train-to-deploy flow
* **CI/CD:** integrated GitHub Actions for test automation
* **API:** FastAPI endpoints functional and validated
* **Documentation:** model card and markdown-formatted README provided

---

### Author

**Raquel Rambo**
Machine Learning DevOps Student – WGU / Udacity
📂 GitHub: [1105blue](https://github.com/1105blue)

```

---