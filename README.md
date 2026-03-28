# 🇬🇭 Loan Default Risk Assessment App

An AI-powered loan assessment tool for the Ghanaian market, featuring real-time Ghana Reference Rate (GRR) integration, machine learning predictions, and comprehensive affordability analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
  <img src="screenshots/app_demo.png" alt="App Demo" width="800">
</p>

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
- [Ghana Reference Rate](#-ghana-reference-rate)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This application helps users in Ghana assess their loan default risk before applying for credit. It combines:

- **Machine Learning**: XGBoost model trained on historical lending data with FinBERT sentiment analysis
- **Ghana Reference Rate**: Real-time GRR integration for accurate interest rate estimation
- **Affordability Analysis**: Comprehensive budget assessment to ensure loan sustainability
- **Historical Statistics**: Compare your profile against historical loan performance data

### Why This Matters

Access to credit is crucial for economic growth, but high default rates hurt both lenders and borrowers. This tool helps:

- **Borrowers**: Understand their approval likelihood before applying
- **Lenders**: Pre-screen applications efficiently
- **Financial Literacy**: Educate users about factors affecting creditworthiness

## ✨ Features

### Core Features

| Feature | Description |
|---------|-------------|
| 🎯 **Risk Assessment** | ML-powered default probability prediction |
| 💰 **Affordability Calculator** | Check if you can actually afford the monthly payments |
| 📊 **GRR Integration** | Auto-fetch current Ghana Reference Rate from GAB |
| 📈 **Historical Stats** | Compare against market performance data |
| 💾 **Save Assessments** | Store and compare multiple loan scenarios |
| 📱 **Mobile Responsive** | Works seamlessly on phones and tablets |

### Risk Factors Analyzed

- Loan amount and term
- Credit score and grade
- Debt-to-income ratio
- Employment length
- Credit utilization
- Payment history (delinquencies)
- Loan purpose
- Application sentiment (FinBERT)

## 🚀 Demo

### Assessment Flow

1. **Enter Loan Details**: Amount, term, purpose
2. **Provide Financial Info**: Income, credit score, DTI
3. **Check Affordability**: Monthly expenses analysis
4. **Get Results**: Approval probability + recommendations

### Sample Output

```
📊 Assessment Results
├── Approval Likelihood: 73%
├── Risk Level: MODERATE
├── Monthly Payment: GH₵1,250
├── Interest Rate: 18.5% (GRR 11.71% + 6.79% premium)
└── Affordability: ✅ Good
```

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/loan-default-assessment-ghana.git
cd loan-default-assessment-ghana
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
python train_and_save.py
```

This will generate three model files:
- `model.joblib` - Trained XGBoost classifier
- `imputer.joblib` - Fitted imputer for missing values
- `config.joblib` - Feature configuration and medians

### Step 5: Run the App

```bash
streamlit run loan_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

### For End Users

1. Open the app in your browser
2. Fill in the **Assessment** tab with your information:
   - Loan amount (GH₵1,000 - GH₵500,000)
   - Loan term (12-60 months)
   - Monthly income and expenses
   - Credit score (300-850)
   - Employment and credit details
3. View your results:
   - Approval probability
   - Affordability status
   - Risk factors and recommendations
4. Save assessments to compare different scenarios

### For Developers

```python
# Load the trained model
import joblib

model = joblib.load("model.joblib")
imputer = joblib.load("imputer.joblib")
config = joblib.load("config.joblib")

# Make predictions
features = config["features"]
X = [[...]]  # Your feature values
X_imputed = imputer.transform(X)
probability = model.predict_proba(X_imputed)[0][1]
```

## 📁 Project Structure

```
loan-default-assessment-ghana/
│
├── 📄 loan_app.py              # Main Streamlit application
├── 📄 train_and_save.py        # Model training script
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md                # This file
│
├── 📊 Data Files
│   └── lending_club_sample_scored.parquet  # Training data (not included)
│
├── 🤖 Model Files (generated after training)
│   ├── model.joblib            # Trained XGBoost model
│   ├── imputer.joblib          # Fitted imputer
│   └── config.joblib           # Configuration
│
├── 📓 Notebooks
│   └── Loan_Application.ipynb  # Original analysis notebook
│
└── 📸 screenshots/             # App screenshots for README
```

## 🤖 Model Details

### Algorithm

- **Model**: XGBoost Classifier
- **Training Data**: Lending Club loans (2007-2018) with FinBERT sentiment scores
- **Performance**: ~72% AUC on test set

### Features Used

| Feature | Description | Type |
|---------|-------------|------|
| `loan_amnt` | Loan amount requested | Numeric |
| `term` | Loan duration (months) | Categorical |
| `int_rate` | Interest rate | Numeric |
| `grade` | Credit grade (A-G → 1-7) | Ordinal |
| `emp_length` | Years at current job | Numeric |
| `annual_inc` | Annual income | Numeric |
| `dti` | Debt-to-income ratio | Numeric |
| `delinq_2yrs` | Delinquencies (past 2 years) | Numeric |
| `open_acc` | Open credit accounts | Numeric |
| `revol_util` | Credit utilization % | Numeric |
| `finbert_pos` | Positive sentiment score | Numeric |
| `finbert_neg` | Negative sentiment score | Numeric |
| `finbert_neu` | Neutral sentiment score | Numeric |

### Hyperparameters

```python
XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    early_stopping_rounds=30
)
```

## 🇬🇭 Ghana Reference Rate

### What is GRR?

The Ghana Reference Rate (GRR) is the benchmark lending rate set by the Bank of Ghana and the Ghana Association of Banks. Commercial banks use it as a base to determine their lending rates.

### How It's Used

```
Your Interest Rate = GRR + Risk Premium

Risk Premium is based on your credit score:
- 750+ score: +5%
- 700-749:    +7%
- 670-699:    +9%
- 640-669:    +11%
- 600-639:    +14%
- 560-599:    +17%
- Below 560:  +20%
```

### GRR Trend (2025-2026)

| Month | GRR |
|-------|-----|
| Jan 2025 | 29.72% |
| Jun 2025 | 22.50% |
| Dec 2025 | 15.90% |
| Mar 2026 | 11.71% |

The GRR has declined significantly, making borrowing more affordable!

### Auto-Fetch Feature

The app automatically fetches the latest GRR from the [Ghana Association of Banks website](https://www.gab.com.gh/grr-historic-data). If the fetch fails, it falls back to stored historical data.

## 📊 Historical Statistics

The app includes historical performance data:

| Metric | Value |
|--------|-------|
| Overall Approval Rate | 67.5% |
| Overall Default Rate | 18.2% |
| Average Loan Amount | GH₵25,000 |

### Default Rate by Credit Grade

| Grade | Approval Rate | Default Rate |
|-------|---------------|--------------|
| A | 95% | 5.2% |
| B | 88% | 9.8% |
| C | 75% | 15.3% |
| D | 58% | 22.7% |
| E | 42% | 31.5% |
| F | 28% | 40.2% |
| G | 15% | 48.8% |

## 🛠️ Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
pyarrow>=14.0.0
requests>=2.31.0
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Ideas for Contributions

- [ ] Add more Ghanaian banks' specific criteria
- [ ] Implement user authentication for persistent storage
- [ ] Add PDF report generation
- [ ] Multi-language support (Twi, Ga, Ewe)
- [ ] Integration with Ghana's credit bureaus
- [ ] Mobile app version (React Native/Flutter)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It does not guarantee loan approval or specific terms. Actual lending decisions depend on individual bank policies, additional verification, and regulatory requirements. This is not financial advice.

## 🙏 Acknowledgments

- [Ghana Association of Banks](https://www.gab.com.gh/) for GRR data
- [Bank of Ghana](https://www.bog.gov.gh/) for monetary policy information
- [Lending Club](https://www.lendingclub.com/) for historical loan data
- [FinBERT](https://github.com/ProsusAI/finBERT) for financial sentiment analysis
- [Streamlit](https://streamlit.io/) for the amazing framework
- [XGBoost](https://xgboost.readthedocs.io/) for the ML library

## 📧 Contact

For questions or feedback:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/loan-default-assessment-ghana/issues)
- **Email**: your.email@example.com

---

<p align="center">
  Made with ❤️ for Ghana 🇬🇭
</p>
