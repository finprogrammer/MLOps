## Project 1: MLOps

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




