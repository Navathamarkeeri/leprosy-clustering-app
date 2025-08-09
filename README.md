# 🩺 Leprosy Clustering & Risk Assessment App

This is a **Streamlit web application** for clustering leprosy patient data and assessing risk based on patient information and foot measurements.

## 🚀 Live Demo
[![Streamlit App](https://img.shields.io/badge/Launch-Streamlit%20App-brightgreen?logo=streamlit)](https://leprosy-clustering-app-cpc2c9jzgxjishmujatmvj.streamlit.app/)

Click the badge above or this link to try it out:  
**🔗 https://leprosy-clustering-app-cpc2c9jzgxjishmujatmvj.streamlit.app/**

## 📌 Features
- **Clustering Analysis** using K-Means to group patient data.
- **Risk Prediction** based on patient details and foot measurements.
- **Interactive Visualizations** for cluster distributions and PCA plots.
- **Model Save & Load** functionality for reusing trained clustering models.
- **File Upload** support for CSV/XLSX datasets.
- **Manual Entry Form** for single patient risk screening.

## 🛠 Installation & Local Run
```bash
# Clone the repository
git clone https://github.com/Navathamarkeeri/leprosy-clustering-app.git
cd leprosy-clustering-app

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

