# Customer Churn Prediction & Risk Segmentation Dashboard

![Dashboard Preview](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.12-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

A comprehensive, interactive Machine Learning dashboard built with **Streamlit** and **Plotly** to predict customer churn, identify high-risk segments, and provide actionable business recommendations. 

The dashboard serves as a professional-grade analytics tool for a telecom retention team to quickly visualize churn patterns and model performances.

## 🚀 Features

- **Interactive KPIs:** Live tracking of Total Customers, Churn Rate, High-Risk Customers, and the Best Model ROC-AUC score.
- **Dynamic Filtering:** Filter data dynamically by Contract Type, Internet Service, Payment Method, Senior Citizen status, and Tenure Range.
- **Machine Learning Models:** Trains and compares multiple ML models on the fly:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- **Risk Segmentation:** Categorizes customers into High, Medium, and Low risk tiers based on their predicted probability of churning.
- **Feature Importance:** Identifies the top drivers of churn using the best-performing tree-based model.
- **Premium UI:** Custom dark-themed UI built with custom CSS, Font Awesome vector icons, and beautifully styled Plotly charts.

## 🛠️ Technology Stack

- **Frontend:** Streamlit, Custom CSS, HTML, Font Awesome 6
- **Data Visualization:** Plotly Express & Graph Objects
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, SciPy

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/parth-ladage/customer-churn-dashboard.git
   cd customer-churn-dashboard
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Dashboard

Ensure your virtual environment is active, then run:

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

## 📂 Project Structure

```text
dashboard/
│
├── app.py               # Main Streamlit application and ML logic
├── style.css            # Custom CSS for dark theme and UI enhancements
├── requirements.txt     # Python package dependencies
└── .gitignore           # Git ignore rules
```

*(Note: The raw dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` is expected to be placed one directory level above this folder.)*

## 💡 Business Recommendations

Based on the model insights, the dashboard automatically generates business strategies such as:
- **Improving Onboarding:** Focusing retention efforts on the critical first 6-12 months.
- **Promoting Long-Term Contracts:** Incentivizing 1-year and 2-year contracts.
- **Optimizing Pricing:** Addressing price sensitivity for customers with high monthly charges.

---
*Built by **Parth Ladage***
