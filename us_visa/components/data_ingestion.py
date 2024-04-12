import os 
import sys

from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.data_access.usvisa_data import USvisaData
from  us_visa.logger import logging
from us_visa.exception import USvisaException

from pandas import DataFrame
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise USvisaException(e, sys)
        

    def export_data_into_feature_store(self)->DataFrame:
        try:
            logging.info(f"Export data from mongodb")
            usvisa_data = USvisaData()
            dataframe = usvisa_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)

            logging.info(f"shape of the dataframe: {dataframe.shape}")
            Feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(Feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"saving exported data to feature store_file_path: {Feature_store_file_path}")
            dataframe.to_csv(Feature_store_file_path, index=False,header=True)

            return dataframe
        
        except Exception as e:
            raise USvisaException(e, sys)
        
    def split_data_as_train_test(self, dataframe: DataFrame)->None:

        logging.info("Entered split_data_as_train_test method of dataingestion class")

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio,shuffle=True)
            logging.info("Performned train test split on the dataframe")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info("Exported train and test file path")
        except Exception as e:
            raise USvisaException(e, sys)
    
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method of data_ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataframe")

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                            test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f"Created the data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise USvisaException(e, sys)
