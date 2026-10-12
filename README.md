# Predictive Customer Churn & Retention Modeling

## Overview

**Predictive Customer Churn & Retention Modeling** is a machine learning project designed to identify customers who are likely to discontinue their telecommunications services.

The project uses the Telco Customer Churn dataset containing **7,043 customer records and 21 attributes**. It applies exploratory data analysis, feature engineering, and gradient boosting algorithms to identify churn patterns and estimate customer churn probabilities.

Built using Python, Pandas, NumPy, CatBoost, LightGBM, and Optuna, the solution evaluates predictive performance and identifies important churn drivers to support proactive retention planning.

## Project Objectives

- Predict customers who are likely to discontinue their services.
- Analyze customer demographics, account information, and service usage.
- Identify patterns associated with customer churn.
- Develop predictive models using CatBoost and LightGBM.
- Optimize model hyperparameters using Optuna.
- Evaluate model performance using classification metrics.
- Identify important churn drivers through feature importance analysis.
- Support data-driven customer retention strategies.

## Problem Statement

Customer churn is a significant challenge in the telecommunications industry. Losing customers can reduce revenue and increase the cost of acquiring replacements.

Customers may discontinue services because of contract conditions, pricing concerns, service requirements, or other factors affecting their experience. Traditional retention strategies may identify these issues only after customers have decided to leave.

This project addresses the problem by analyzing historical customer data and building machine learning models to estimate churn probability. The resulting predictions and feature importance insights can help businesses identify at-risk customers and plan targeted retention activities.

## Dataset Description

The project uses the **Telco Customer Churn dataset**, which contains customer demographic information, account details, service subscriptions, billing information, and churn labels.

| Dataset Attribute | Description |
|---|---|
| Dataset Name | Telco Customer Churn |
| Total Records | 7,043 |
| Total Columns | 21 |
| Target Variable | Churn |
| Target Classes | Yes, No |
| Domain | Telecommunications |
| Data Type | Structured Tabular Data |
| Reported Churn Rate | Approximately 26.5% |

### Dataset Categories

The dataset contains information across the following categories:

- **Customer Demographics:** Gender, senior citizen status, and other demographic attributes.
- **Account Information:** Customer tenure, contract type, payment method, and billing details.
- **Service Information:** Phone service, internet service, streaming services, online security, and technical support.
- **Billing Information:** Monthly charges and related account values.
- **Churn Information:** Indicates whether a customer discontinued the service.

The dataset is used to explore customer behavior, identify churn patterns, and train predictive models.

## Key Features

- **Exploratory Data Analysis:** Examines customer characteristics and churn distribution.
- **Data Preprocessing:** Cleans the dataset and prepares features for modeling.
- **Feature Engineering:** Creates derived features such as tenure groups and payment categories.
- **Churn Prediction:** Estimates the probability that a customer will leave.
- **CatBoost Modeling:** Uses gradient boosting optimized for categorical data.
- **LightGBM Modeling:** Uses an efficient gradient boosting framework for structured data.
- **Hyperparameter Optimization:** Uses Optuna to search for suitable model configurations.
- **Model Evaluation:** Assesses predictions using classification metrics and ROC-AUC.
- **Feature Importance Analysis:** Identifies factors associated with customer churn.
- **Prediction Reporting:** Produces churn probabilities and interpretable results for business users.
- **Retention Planning:** Supports the identification of customers who may benefit from targeted retention activities.

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas |
| Numerical Computing | NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Predictive Modeling | CatBoost, LightGBM |
| Machine Learning Utilities | Scikit-learn |
| Hyperparameter Optimization | Optuna |
| Interactive Visualization | Plotly |
| Model Evaluation | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| Model Visualization | Graphviz |
| Development Environment | Jupyter Notebook / JupyterLab |
| Version Control | Git, GitHub |

## Project Workflow

The project follows a structured machine learning workflow, from data exploration to prediction and reporting.

### 1. Data Loading

Load the Telco Customer Churn dataset from the provided CSV file.

### 2. Exploratory Data Analysis

Examine the dataset structure, data types, customer distribution, and churn patterns.

Analyze relationships between churn and factors such as tenure, contract type, monthly charges, and service subscriptions.

### 3. Data Preprocessing

Prepare the data for machine learning by:

- Checking for missing values and duplicate records.
- Encoding categorical variables.
- Preparing numerical features.
- Applying appropriate transformations and scaling where required.
- Separating input features from the target variable.

### 4. Feature Engineering

Create additional features that help represent customer behavior and account characteristics.

Examples include:

- Customer tenure groups.
- Payment categories.
- Derived billing-related features.
- Relevant service and account attributes.

### 5. Model Training

Train predictive models using CatBoost and LightGBM.

These gradient boosting algorithms are used to learn patterns in structured customer data and estimate the probability of churn.

### 6. Hyperparameter Optimization

Use Optuna to search for suitable model hyperparameters, including learning rate, tree depth, iteration count, and regularization parameters.

### 7. Model Evaluation

Evaluate model performance using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and cross-validation where configured.

### 8. Churn Prediction

Generate churn probabilities for customers and identify customers who may be at higher risk of leaving.

### 9. Feature Importance Analysis

Examine the relative importance of customer attributes to understand factors associated with churn predictions.

### 10. Reporting and Retention Planning

Present prediction results and analytical insights to support business decisions and proactive retention strategies.

## Algorithms Used

### 1. Gradient Boosting Decision Trees

Gradient Boosting Decision Trees (GBDT) build an ensemble of decision trees sequentially, with each subsequent tree attempting to improve on the errors of previous trees.

This approach is suitable for structured customer data and serves as the foundation for the CatBoost and LightGBM models used in the project.

### 2. CatBoost

CatBoost is a gradient boosting algorithm designed to work effectively with categorical features.

In this project, it is used to:

- Predict customer churn.
- Learn patterns from customer and account attributes.
- Analyze feature importance.
- Support churn probability estimation.

### 3. LightGBM

LightGBM is a gradient boosting framework designed for efficient model training on structured datasets.

It is used to:

- Train churn prediction models.
- Generate churn probability estimates.
- Compare predictive performance with CatBoost.
- Analyze feature importance.

### 4. Optuna

Optuna is used for hyperparameter optimization.

The optimization process explores model configurations such as:

- Learning rate.
- Number of iterations or boosting rounds.
- Tree depth.
- Regularization parameters.

The objective is to identify model configurations that improve predictive performance while managing overfitting.

### 5. Classification Metrics

The project evaluates predictive performance using multiple classification metrics.

- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Measures the proportion of predicted churn cases that are actual churn cases.
- **Recall:** Measures the proportion of actual churn cases identified by the model.
- **F1-Score:** Combines precision and recall into a single metric.
- **ROC-AUC:** Measures how effectively the model distinguishes between churn and non-churn customers.
- **Confusion Matrix:** Summarizes correct and incorrect predictions across the target classes.

## Data Preprocessing and Feature Engineering

Data preprocessing prepares the dataset for reliable model training.

The documented workflow includes:

- Inspecting dataset structure and data types.
- Checking data completeness and duplicates.
- Encoding categorical attributes.
- Preparing numerical features.
- Applying transformations where necessary.
- Creating derived features such as tenure groups and payment categories.
- Separating the target variable from the input features.

Preprocessing steps should be fitted on training data and applied consistently to validation and test data to avoid data leakage.

## Exploratory Data Analysis

Exploratory data analysis helps reveal patterns and relationships within the customer dataset.

The project examines:

- Churn distribution between customers who leave and those who remain.
- Customer tenure and churn behavior.
- Contract type and churn relationships.
- Monthly charges and churn patterns.
- Payment methods and account characteristics.
- Service subscriptions and churn behavior.
- Relationships between customer attributes and the target variable.

### Visualization Techniques

The project uses Matplotlib, Seaborn, and Plotly for visual analysis.

Common visualization categories include:

- Churn distribution charts.
- Categorical feature comparisons.
- Correlation analysis.
- Customer tenure analysis.
- Feature importance plots.
- Confusion matrices.
- ROC curves and precision-recall curves.

These visualizations help interpret model behavior and communicate analytical findings.

## Model Evaluation Strategy

The models are evaluated using multiple metrics to assess classification performance and generalization.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall prediction correctness |
| Precision | Evaluates the reliability of predicted churn cases |
| Recall | Measures how many actual churn cases are identified |
| F1-Score | Balances precision and recall |
| ROC-AUC | Evaluates the model's ability to distinguish churn from non-churn customers |
| Confusion Matrix | Shows correct and incorrect classifications |
| Cross-Validation | Assesses model performance across multiple data splits |
| Feature Importance | Identifies influential input attributes |

### Performance Criteria

The PRD specifies the following project success criteria:

- Model accuracy greater than 80%.
- ROC-AUC greater than 0.85.
- Evaluation using precision, recall, and F1-score.
- Identification of important churn drivers.
- Potential support for reducing churn through proactive business actions.

These are the documented performance criteria. Exact achieved values should be reported only when confirmed by the final model evaluation results.

## Project Modules

| Module | Responsibility |
|---|---|
| Data Preprocessing | Cleans data and prepares features |
| Exploratory Data Analysis | Examines customer distributions and churn patterns |
| Feature Engineering | Creates derived features for model training |
| Model Training | Trains CatBoost and LightGBM models |
| Hyperparameter Optimization | Uses Optuna to tune model configurations |
| Evaluation | Calculates classification metrics and generates evaluation plots |
| Prediction | Estimates churn probabilities for customers |
| Feature Importance | Identifies influential customer attributes |
| Reporting | Presents predictions and analytical findings |

## Project Structure

The following is a logical project structure based on the modules described in the PRD. Actual filenames and directory names should be verified against the repository.

```text
customer-churn-prediction/
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_catboost.py
│   ├── train_lightgbm.py
│   ├── optimize_hyperparameters.py
│   ├── evaluate.py
│   └── predict.py
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── reports/
├── requirements.txt
└── README.md
```

## Hardware Requirements

| Component | Minimum Requirement | Recommended Requirement |
|---|---|---|
| Operating System | Windows 10/11, macOS, or Linux | Windows 10/11, macOS, or Linux |
| Processor | Intel Core i5/i7 or equivalent AMD processor | Higher-performance multicore processor |
| RAM | 8 GB | 16 GB or higher |
| Storage | 2 GB free disk space | Additional space for outputs and model artifacts |
| GPU | Optional | NVIDIA GPU with CUDA support, where supported |

CPU execution is sufficient for the dataset size described in the PRD.

## Software Requirements

- Python 3.10
- pip
- Jupyter Notebook or JupyterLab
- Visual Studio Code, PyCharm, or another Python editor
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- CatBoost
- LightGBM
- Optuna
- Plotly
- Graphviz

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd customer-churn-prediction
```

Replace `<your-repository-url>` with the actual GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

**Linux or macOS:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

If the repository contains a `requirements.txt` file, install the dependencies using:

```bash
pip install -r requirements.txt
```

Alternatively, install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost lightgbm optuna plotly jupyter
```

Graphviz may require a separate system installation for rendering certain visualizations.

### 5. Prepare the Dataset

Place the dataset in the expected project directory.

The documented default filename is:

```text
Telco-Customer-Churn.csv
```

Ensure the notebook or data-loading script references the correct dataset path.

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Alternatively, launch JupyterLab:

```bash
jupyter lab
```

### 7. Execute the Workflow

Follow the project notebook or application scripts in this order:

1. Load and inspect the dataset.
2. Perform exploratory data analysis.
3. Preprocess the input features.
4. Perform feature engineering.
5. Train CatBoost and LightGBM models.
6. Optimize model hyperparameters using Optuna.
7. Evaluate model performance.
8. Generate churn probabilities.
9. Review feature importance and evaluation visualizations.

Use the actual notebook names and execution instructions available in the repository.

## Expected Outputs

The project is designed to produce:

- Exploratory data analysis summaries.
- Customer churn distribution visualizations.
- Processed features for machine learning.
- Trained CatBoost and LightGBM models.
- Hyperparameter optimization results.
- Accuracy, precision, recall, F1-score, and ROC-AUC metrics.
- Confusion matrices and model evaluation curves.
- Feature importance visualizations.
- Customer-level churn probability estimates.
- Analytical insights to support retention planning.

Actual output files and numerical results depend on the implementation and model execution.

## Business Applications

The project can support telecommunications businesses in the following areas:

- **Proactive Retention:** Identify customers who may be likely to discontinue services.
- **Customer Risk Analysis:** Estimate churn probabilities using historical customer data.
- **Churn Driver Identification:** Understand which customer characteristics contribute to model predictions.
- **Retention Planning:** Help prioritize customer outreach and retention initiatives.
- **Business Reporting:** Communicate predictive insights through analytical reports and visualizations.
- **Decision Support:** Use model outputs to inform customer relationship management strategies.

The project provides predictive insights; actual reductions in churn or improvements in customer lifetime value require separate business implementation and outcome measurement.

## Future Enhancements

Potential extensions include:

- Integrating churn predictions with CRM platforms.
- Developing interactive dashboards for business stakeholders.
- Automating model retraining as new customer data becomes available.
- Monitoring model performance over time.
- Improving explainability of customer-level predictions.
- Evaluating additional models and feature engineering strategies.
- Deploying prediction services for integration with business applications.

These enhancements should be treated as future possibilities unless they are implemented in the project.

## Conclusion

Predictive Customer Churn & Retention Modeling applies machine learning to identify patterns associated with customer attrition in the telecommunications industry.

By combining exploratory data analysis, feature engineering, CatBoost, LightGBM, and Optuna-based hyperparameter optimization, the project develops a structured approach to churn prediction and model evaluation.

The resulting churn probabilities and feature importance insights can help businesses identify at-risk customers, understand churn drivers, and make more informed retention decisions.

## License

Add the appropriate license information based on the repository owner's licensing decision.
