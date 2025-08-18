import os
import sys
import json
from dotenv import load_dotenv
import certifi
import pandas as pd
import pymongo
from networksecurity.exception.exception import NetworkSecurityException

load_dotenv()

MONGO_DB_URL = os.getenv("MONGODB_URL_KEY")
if not MONGO_DB_URL:
    raise EnvironmentError("MONGODB_URL_KEY is not set in environment variables")

ca = certifi.where()

class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def insert_data_mongodb(self, records, database, collection, batch_size=1000):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.database = self.mongo_client[database]
            self.collection = self.database[collection]

            total_inserted = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                self.collection.insert_many(batch)
                total_inserted += len(batch)

            return total_inserted
        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__ == '__main__':
    FILE_PATH = "Network_Data/covtype.csv"
    DATABASE = "trialset_covtype"
    collection = "Data_covtype"
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, collection)
    print("Number of records inserted:", no_of_records)
    print("Number of documents in DB:", networkobj.collection.count_documents({}))
