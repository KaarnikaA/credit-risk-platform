
# Credit Risk Prediction Platform (MLOps + Explainable AI)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) 
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green?logo=fastapi) 
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange) 
![Airflow](https://img.shields.io/badge/Airflow-Orchestration-red?logo=apacheairflow) 
![Feast](https://img.shields.io/badge/Feast-Feature%20Store-black) 
![Redis](https://img.shields.io/badge/Redis-Cache-red?logo=redis)


### Project Summary

Built as a personal end-to-end implementation of a complete machine learning system, focused on understanding how production-grade credit risk platforms are designed, trained, and deployed in real-world environments.

The goal was to move beyond isolated model training and implement a complete MLOps workflow, where every stage of the machine learning lifecycle is connected and operationally meaningful.

---
### Project Objective

The objective of this project was to design and implement a complete machine learning system with a production-style MLOps architecture, focusing on:

- Workflow orchestration  
- Model governance and versioning  
- Real-time inference  
- Feature consistency  
- Explainable AI  

This reflects a practical attempt to simulate how credit risk systems are built and operated in real production environments.

---

### End-to-End Workflow

- **Data Ingestion**  
  Raw loan application data is collected and structured for downstream processing.

- **Feature Engineering**  
  Financial attributes are transformed into meaningful risk indicators such as income ratios, debt behavior signals, and interaction-based features.

- **Feature Store Layer**  
  A centralized feature store ensures consistency between training and inference by reusing the same feature logic across the system.

- **Training Pipeline (Airflow Orchestration)**  
  An automated pipeline trains an XGBoost model and evaluates it using predefined performance metrics.

- **Model Validation**  
  Only models that meet the required performance threshold are considered for deployment.

- **Model Registry (MLflow)**  
  Validated models are registered with version control for traceability and reproducibility.

- **Real-Time Serving (FastAPI)**  
  The approved model is deployed as an API service for real-time inference.

- **Low-Latency Feature Retrieval (Redis)**  
  Runtime features are fetched from a fast feature store to support real-time scoring.

- **Prediction & Decisioning**  
  The model generates probability of default, which is converted into a credit score and combined with rule-based logic to produce the final decision.

- **Explainability Layer (SHAP)**  
  Each prediction is explained by identifying the most influential features driving the model output.

---

