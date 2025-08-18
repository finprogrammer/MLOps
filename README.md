## Project 1: Intrusion Detection System (UNSW-NB15 Dataset)

### Key Features

- **Exploratory Data Analysis (EDA)**  

- **ETL Pipeline**  

- **Model Training & Evaluation**
  - Trains multiclass classification models using `GridSearchCV`.
  - Evaluates models using `F1-score` (weighted).
  - Saves the best model and preprocessing objects as `.pkl` files to AWS S3.

- **CI/CD Pipeline**
  - **GitHub Actions** for:
    - Code linting
    - Building Docker containers
    - Pushing Docker images to **AWS Elastic Container Registry (ECR)**
    - Deploying containers to **AWS EC2**

- **MLOps with DagsHub & MLflow**
  - Tracks experiments, models, parameters, and metrics.
  - Uses `MLflow` integrated with `DagsHub` for version control and collaboration.

- **Web-Based Triggering**
  - Trains and predicts via a **FastAPI** interface with **Jinja2 templating**.

- **Cloud Integration**
  - Stores raw and processed data, models, and transformers in **AWS S3**.

---

## Project 2: Blog Generation with GenAI (Meta LLaMA 3)

### Features

- **Amazon Bedrock Integration**
  - Uses **Meta's LLaMA 3** model for blog content generation via `boto3`.

- **AWS Lambda + API Gateway**
  - AWS Lambda function triggers blog generation on HTTP requests.
  - Blog content is stored automatically in an S3 bucket.

![blog](https://github.com/user-attachments/assets/9972447d-3091-4477-bd88-9d2ec9d60010)

## Project 3: ETL Pipeline Automation with Apache Airflow
- **ETL Pipeline**  
  Structured modular ETL workflow with the following:
  - **Data Ingestion:** Extracts raw data from MongoDB.
  - **Data Validation:** Detects data drift using the Kolmogorov–Smirnov (KS) test and ensures schema conformity.
  - **Data Transformation:** 
    - Handles missing values, encoding, scaling.
    - Class imbalance handled with SMOTE.
    - Saves transformed objects and pipelines for reproducibility.

