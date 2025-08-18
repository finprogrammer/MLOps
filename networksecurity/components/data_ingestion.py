import os
import sys
import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
from typing import List
from sklearn.model_selection import train_test_split

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

load_dotenv()
MONGO_DB_URL = os.getenv("MONGODB_URL_KEY")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Connect to MongoDB and load the specified collection into a DataFrame.
        """
        try:
            db_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            client = pymongo.MongoClient(MONGO_DB_URL)

            collection = client[db_name][collection_name]
            records = list(collection.find())
            df = pd.DataFrame(records)

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace({"na": np.nan}, inplace=True)
            logging.info(f"Data loaded from MongoDB collection: {collection_name}, rows: {df.shape[0]}")
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Save the DataFrame into a local CSV file (feature store).
        """
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)
            dataframe.to_csv(feature_store_path, index=False)
            logging.info(f"Data exported to feature store at: {feature_store_path}")
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Split the input dataframe into train and test sets and save them as CSV files.
        """
        try:
            train_data, test_data = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                stratify=dataframe[TARGET_COLUMN],
                random_state=42,
            )

            train_path = self.data_ingestion_config.training_file_path
            test_path = self.data_ingestion_config.testing_file_path

            os.makedirs(os.path.dirname(train_path), exist_ok=True)

            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

            logging.info(f"Train and test datasets saved at: {train_path}, {test_path}")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute full data ingestion: load from DB, store to CSV, split into train/test, return artifact.
        """
        try:
            df = self.export_collection_as_dataframe()
            df = self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)

            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info("Data ingestion completed and artifact created.")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
