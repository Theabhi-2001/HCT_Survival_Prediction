# HCT_Survival_Prediction
Machine learning-based survival outcome prediction system for Hematopoietic Cell Transplantation (HCT) using ensemble models with SHAP explainability.
Excellent 👏 Let’s make a **professional and clean `README.md`** for your mini project so it looks great on GitHub.
Here’s a complete version tailored for your project — you can copy and paste this directly into a file named **`README.md`** in your GitHub repo 👇

---

##  Survival Outcome Prediction for Hematopoietic Cell Transplantation Using Machine Learning Ensemble Model

###  Project Overview

This project focuses on predicting **Event-Free Survival (EFS)** outcomes for patients undergoing **Hematopoietic Cell Transplantation (HCT)** using a **machine learning ensemble approach**.
The system integrates **Random Forest**, **Logistic Regression**, and **XGBoost** models using a **soft voting ensemble**, improving accuracy and robustness.
To make the model interpretable for clinicians, **SHAP (Shapley Additive Explanations)** is incorporated to visualize how each clinical feature influences the prediction.

---

###  Objective

* To predict patient survival outcomes after HCT based on clinical and donor-related data.
* To improve accuracy and stability using an ensemble of multiple ML models.
* To enhance interpretability through SHAP-based feature explanation.
* To provide a user interface where clinicians can upload patient data and obtain survival predictions.

---

###  Methodology

1. **Data Preprocessing**

   * Handling missing values using imputation.
   * One-hot encoding for categorical variables.
   * Feature normalization for uniform scaling.

2. **Model Training**

   * Individual models: Random Forest, Logistic Regression, XGBoost.
   * Hyperparameter tuning using GridSearchCV.
   * Ensemble through **soft voting** to combine prediction probabilities.

3. **Explainability**

   * **SHAP** was used to identify and visualize top influencing features.
   * Enhances model transparency for clinical decision-making.

---

###  Dataset

* Source: Public dataset from **Kaggle** (HCT clinical dataset)
* **Total records:** 28,800
* **Features:** 60 (Demographic, Clinical, Donor-related, and Disease-related)
* **Target variable:** `efs` (Event-Free Survival)

---

###  Technologies Used

* **Language:** Python
* **Libraries:** scikit-learn, XGBoost, pandas, numpy, SHAP
* **Environment:** Jupyter Notebook / VS Code
* **Model Files:**

  * `voting_model.pkl` – Trained ensemble model
  * `preprocessor.pkl`, `num_cols.pkl`, `cat_cols.pkl` – Preprocessing pipeline files

---

###  Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 91.35% |
| Precision | 0.8877 |
| Recall    | 0.9646 |
| F1-Score  | 0.9245 |
| ROC-AUC   | 0.9787 |

---

###  How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:

   ```bash
   python app.py
   ```
4. Open your browser at **[http://localhost:5000/](http://localhost:5000/)** and upload a `.csv` file containing patient data.



###  Project Structure
---
📁 HCT_Survival_Prediction/
│
├── app.py                     # Flask web app for prediction
├── model_training.py          # ML model training code
├── voting_model.pkl           # Final ensemble model
├── preprocessor.pkl           # Preprocessing pipeline
├── num_cols.pkl               # Numerical columns used
├── cat_cols.pkl               # Categorical columns used
├── README.md                  # Project description
└── LICENSE                    # MIT License

---

### Results Visualization

* SHAP Summary plots highlight top influencing features such as:

  * `age_at_hct`
  * `dri_score`
  * `comorbidity_score`
  * `hla_match_c_high`

These help clinicians understand **why** a certain prediction was made.


### License

This project is licensed under the **MIT License**.
See the [LICENSE](./LICENSE) file for more information.


**Author**

**Bala Abhishek Udagandla**
*M.Tech in Software Engineering, VNR Vignana Jyothi Institute of Engineering & Technology, Hyderabad, India*
abhishekbala789@gmail.com
