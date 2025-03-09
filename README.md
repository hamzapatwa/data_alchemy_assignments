# Phishing URL Detection System

## Overview
This project aims to identify phishing websites using machine learning models trained on extracted URL features. Phishing websites mimic legitimate ones to steal sensitive information from users. Our system uses advanced feature extraction techniques, machine learning models, and a Streamlit dashboard for analysis and real-time URL checking.

## Features & Capabilities
- **Feature Extraction**: Analyzes URLs for address bar, domain, HTML/JavaScript, lexical, and statistical features.
- **Machine Learning Models**: Implements Decision Trees, Random Forests, XGBoost, SVM, and Multilayer Perceptrons (MLP).
- **Model Performance**: Evaluates models using accuracy, precision, and confusion matrices.
- **Visualization & Analysis**: Provides detailed exploratory data analysis (EDA) and feature importance.
- **Real-time Phishing Detection**: Accepts user input and classifies URLs as phishing or legitimate.
- **Streamlit Dashboard**: User-friendly interface for model evaluation and live URL analysis.

## Dataset
The dataset contains legitimate and phishing URLs sourced from:
- **PhishTank**: A repository of verified phishing URLs.
- **University of New Brunswick**: A dataset of legitimate and phishing URLs.

### Feature Engineering
We extract several features from each URL:
#### **Address Bar Features:**
- **Having_IP**: Whether the URL contains an IP instead of a domain.
- **Have_At**: Presence of '@' symbol.
- **URL_Length**: URLs longer than 54 characters are suspicious.
- **URL_Depth**: Count of '/' in the path.
- **Redirection**: Checks occurrences of '//' beyond protocol.
- **HTTPS_Domain**: Detects if 'https' appears in the domain.
- **TinyURL**: Identifies usage of link-shortening services.
- **Prefix_Suffix**: Checks for '-' in domain names.

#### **Lexical & Statistical Features:**
- **URL Entropy**: Measures randomness in the URL.
- **Domain Entropy**: Measures randomness in the domain name.
- **Subdomain Count**: Counts additional subdomains.
- **Digit Count**: Number of numerical characters in the URL.
- **Special Character Count**: Identifies non-alphanumeric symbols.
- **Uppercase Ratio**: Percentage of uppercase letters.
- **Domain Age**: Computes domain registration age via WHOIS.

#### **HTML & JavaScript Features:**
- **iFrame**: Detects iframe redirection abuse.
- **Mouse Over**: Checks for JavaScript altering links.
- **Right Click Disable**: Detects websites preventing right-click actions.
- **Web Forwarding**: Counts redirections on the site.

## Model Training & Evaluation
### **Models Implemented:**
1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **Multilayer Perceptrons (MLP)**
4. **XGBoost Classifier**
5. **Support Vector Machine (SVM)**

### **Performance Metrics:**
- **Accuracy**: Measures overall correctness.
- **Precision**: Determines how many predicted phishing URLs were actually phishing.
- **Confusion Matrix**: Visual representation of model performance.
- **Feature Importance Analysis**: Evaluates the impact of different features on predictions.

### **Results:**
- The best-performing model was **XGBoost** with **93% accuracy and precision**.
- Decision Trees and Random Forests also provided strong results, with interpretable decision paths.
- The feature importance analysis highlighted `URL Entropy`, `Domain Age`, and `TinyURL` usage as key indicators of phishing.

## Streamlit Dashboard
The Streamlit dashboard provides an interactive interface with the following features:
1. **Data Overview**: Displays dataset statistics and feature explanations.
2. **Exploratory Data Analysis (EDA)**: Allows users to visualize distributions and correlations.
3. **Train & Evaluate Models**: Enables training and validation of multiple models.
4. **Model Comparison**: Summarizes and ranks models based on performance.
5. **Live URL Checker**: Accepts URLs and predicts their legitimacy using the trained model.

## Installation & Setup
### **Prerequisites:**
- Python 3.8+
- Pip & Virtual Environment
- Streamlit, Scikit-learn, XGBoost, Pandas, NumPy, Plotly

### **Installation Steps:**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourrepo/phishing-detector.git
   cd phishing-detector
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
### **Running the Phishing Detector**
- Open the Streamlit dashboard.
- Go to the "Phishing URL Detector" tab.
- Enter a URL and click **Check URL**.
- The model will predict whether the URL is **Phishing** or **Legit**.

## Troubleshooting
- **Issue: Model always predicts "Legit"**
  - Verify that feature extraction outputs match training data.
  - Check if the trained model is properly loaded in the Streamlit app.
- **Issue: WHOIS lookup failing for new domains**
  - Some domain registrars block WHOIS queries. Consider using an API for better reliability.
- **Issue: URL Checker is slow**
  - Reduce the number of requests or optimize parallel execution.

## Future Improvements
- **Enhance Feature Engineering:** Incorporate content-based features by scraping webpage text.
- **Improve WHOIS Reliability:** Use a dedicated WHOIS API for real-time domain age retrieval.
- **Deep Learning Models:** Experiment with LSTMs or CNNs for URL pattern analysis.
- **Expand Dataset:** Integrate more real-world phishing URL sources.


## License
This project is licensed under the MIT License.

## References
- PhishTank (https://www.phishtank.com/)
- University of New Brunswick (https://www.unb.ca/cic/datasets/url-2016.html)
- Scikit-learn documentation (https://scikit-learn.org/)
- XGBoost documentation (https://xgboost.readthedocs.io/)

