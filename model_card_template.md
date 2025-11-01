# 🧾 Model Card – Census Income Classification Model

---

## Model Details

- **Model Type:** Random Forest Classifier  
- **Framework:** scikit-learn  
- **Training Script:** `train_model.py`  
- **Inference API:** FastAPI app (`main.py`)  
- **Encoders:** OneHotEncoder and LabelBinarizer (stored in `model/` folder)  
- **Developer:** Raquel Rambo  
- **Version:** 1.0.0  
- **Date:** November 1, 2025  

The model was trained as part of a machine learning DevOps project to demonstrate an end-to-end workflow — from preprocessing and model training to deployment with continuous integration and RESTful inference.

---

## Intended Use

- **Purpose:**  
  To predict whether a person earns more than \$50K per year based on demographic and employment data from the U.S. Census Bureau.

- **Intended Users:**  
  Students, data analysts, and engineers learning to build, test, and deploy machine learning pipelines using MLOps best practices.

- **Out of Scope:**  
  This model is not designed or approved for real-world financial, hiring, or policy decisions.  
  It should only be used in an educational or experimental environment.

---

## Training Data

- **Source:** [UCI Adult Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)  
- **Total Rows:** ~48,000 (after cleaning)  
- **Split:** 80% training / 20% test  
- **Label:** `salary` (binary: `>50K` or `<=50K`)  
- **Feature Columns:**
```

age, workclass, fnlgt, education, education-num,
marital-status, occupation, relationship, race, sex,
capital-gain, capital-loss, hours-per-week, native-country

```

All categorical features were one-hot encoded, and the label was binarized using `LabelBinarizer`.

---

## Evaluation Data

The model was tested on the 20% hold-out portion of the dataset.  
Preprocessing was applied consistently with the same encoders used during training.

---

## Metrics

| Metric | Score |
| :------ | :----: |
| Precision | 0.74 |
| Recall | 0.64 |
| F1 Score | 0.68 |

Performance values are based on the model’s predictions on the test set.  
These metrics reflect the balance between correctly identifying higher earners and minimizing misclassifications.

---

## Data Slice Performance

Performance was also evaluated across key categorical slices to check for bias and consistency.  
A few examples from `slice_output.txt`:

| Slice Feature | Example Value | Precision | Recall | F1 |
| :------------- | :------------- | :--------: | :------: | :----: |
| sex | Female | 0.72 | 0.61 | 0.66 |
| sex | Male | 0.74 | 0.64 | 0.69 |
| race | White | 0.75 | 0.65 | 0.70 |
| race | Black | 0.71 | 0.60 | 0.65 |

While results are relatively balanced, continued monitoring of slice performance is recommended before any real deployment.

---

## Ethical Considerations

- The dataset includes **sensitive demographic variables** such as race and gender, which may encode existing social and economic biases.  
- Predictions should **not** be used to influence employment, financial, or policy decisions.  
- This model is developed for **educational demonstration only**, with the goal of practicing responsible MLOps principles.

---

## Caveats and Recommendations

- Model interpretability is limited — Random Forests provide good performance but are not highly transparent.  
- Feature drift or demographic imbalance over time could reduce accuracy if retraining is not performed regularly.  
- Potential improvements:
- Implement fairness metrics and bias-reduction techniques  
- Add automated retraining and monitoring for model drift  
- Deploy via Docker or Kubernetes for production-grade scaling  

---

## Maintenance and Monitoring

- **Retraining:** Recommended every 4–6 weeks or when new data becomes available  
- **CI/CD:** Automated via GitHub Actions (`pytest`, `flake8`, FastAPI tests)  
- **Version Control:** Git and GitHub  
- **Monitoring:** Slice performance logs (`slice_output.txt`) and CI workflow history  

---

## Version History

| Version | Date | Summary |
| :------ | :---- | :------- |
| 1.0.0 | Nov 1, 2025 | Initial release with trained model, CI/CD setup, API deployment, and documentation. |

---

### Author

**Raquel Rambo**  
B.S. Data Analytics Candidate – Western Governors University  
Machine Learning DevOps Portfolio Project  
📂 GitHub: [1105blue](https://github.com/1105blue)
```
