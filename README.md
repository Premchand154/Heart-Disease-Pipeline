# Heart-Disease-Pipeline
End-to-End Machine Learning Pipeline using Heart Disease UCI Dataset (Data Cleaning → EDA → Feature Engineering → Model → FastAPI Deployment).

**Project Overview**

This project builds a complete end-to-end **Machine Learning pipeline** to predict the presence of heart disease using the **Heart Disease UCI** dataset.

It includes:
    Data cleaning, Exploratory data analysis (EDA), Feature engineering
    Model training (Logistic Regression & Random Forest), Model evaluation
    Saving the final model, API deployment using **FastAPI**

# **1. Dataset Used**

**Dataset:** Heart Disease UCI
**Rows (before cleaning):** 920
**Rows (after cleaning):** 702
**Features:** 15 (mixed categorical + numerical)
**Target:** `num` (converted to binary: 0 = no disease, 1 = disease)

# **2. Data Cleaning Steps**
1.Removed `id` column.
2.Converted datatypes.
    * sex → Male/Female → 1/0.
    * fbs & exang → True/False → 1/0.
3.Missing value handling.
    * Numerical: Median imputation.
    * Categorical: Mode imputation.
4.Duplicate removal.
5.Outlier removal using IQR(removed ~218 rows).
6.Target binarization.
    num = 0 → No disease.
    num = 1,2,3,4 → Disease.

# **3. Exploratory Data Analysis**

Key plots include:

* Histograms
* Boxplots
* Countplots
* Correlation Heatmap
* Scatterplots

### **Top 10 EDA insights:**

1. Age is nearly normally distributed around 50–60 years.
2. Oldpeak is heavily right-skewed.
3. Cholesterol has the widest spread (~100–350).
4. Majority patients are male.
5. Asymptomatic chest pain type is most common.
6. Flat slope is highly dominant.
7. Higher oldpeak strongly correlates with disease.
8. Thalch decreases with age.
9. oldpeak (0.45) has the highest positive correlation with disease.
10. Scatterplots show diseased patients cluster at high oldpeak and low thalch.

# **4. Feature Engineering**
        1.One-Hot Encoding
                 Applied to: cp, restecg, slope, thal, dataset
        2.Label Encoding
                 sex, fbs, exang → numeric
        3.Standardization
                 Applied to all numerical features
        4.Created new feature
               **Heart Rate Reserve (HRR):**
                           HRR = 220 - age - thalch 
        5.Final Preprocessor
                 Built using **ColumnTransformer** + **Pipeline**

# **5. Model Training**

Two models trained:

1. **Logistic Regression**
2. **Random Forest (100 trees)**

Both models used a **Pipeline** containing preprocessing steps.


# **6. Model Evaluation**

| Model                   | Accuracy  | Precision | Recall    | F1        | ROC-AUC   |
| ----------------------- | --------- | --------- | --------- | --------- | --------- |
| **Logistic Regression** | **0.766** | **0.75**  | **0.738** | **0.744** | **0.834** |
| Random Forest           | 0.752     | 0.734     | 0.723     | 0.728     | **0.839** |

### Final Model Selected: **Logistic Regression**

**Reasons:**

* Best F1-score
* Higher recall (lower false negatives → critical for medical data)
* More stable and interpretable
* Nearly equal ROC-AUC

Model saved as:
     models/heart_disease_lr.pkl

# **7. FastAPI Deployment**

The model is deployed using **FastAPI**.

### **Start the server**

```bash
uvicorn main:app --reload
```

### **POST /predict example input**

```json
{
  "age": 50,
  "trestbps": 130,
  "chol": 250,
  "thalch": 150,
  "oldpeak": 1.5,
  "ca": 0,
  "sex": 1,
  "cp": "asymptomatic",
  "restecg": "normal",
  "slope": "flat",
  "thal": "normal",
  "dataset": "Cleveland",
  "exang": 0,
  "fbs": 0
}
```

### **Output**

```json
{
  "prediction": 1,
  "probability_of_disease": 0.78
}
```

# **8. Challenges & Learnings**

### **Challenges**

* Handling many missing values
* Outlier removal caused large row reduction
* Encoding mixed categorical variables
* Avoiding data leakage
* Choosing between interpretability vs performance

### **Learnings**

* Proper EDA reveals medically meaningful patterns
* Logistic Regression performs strongly on clean tabular data
* Pipelines ensure clean deployment-ready workflows
* FastAPI is simple for packaging ML models
* Confusion matrix is crucial for medical datasets

---

# **9. How to Run This Project Locally**

### 1. Clone repository

```bash
git clone https://github.com/your-username/heart-disease-ml-pipeline.git
cd heart-disease-ml-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start FastAPI

```bash
cd api
uvicorn main:app --reload
```

# **10. Repository Link**

Add after pushing:

👉 **[https://github.com/your-username/heart-disease-ml-pipeline](https://github.com/your-username/heart-disease-ml-pipeline)**




