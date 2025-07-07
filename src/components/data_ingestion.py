import os
import zipfile
import logging
import requests
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> str:
        os.makedirs(os.path.dirname(self.config.feature_store_file_path), exist_ok=True)
        logging.info("Downloading dataset from: %s", self.config.training_pipeline_config.download_url)
        response = requests.get(self.config.training_pipeline_config.download_url, stream=True)
        if response.status_code == 200:
            with open(self.config.feature_store_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            logging.info("Downloaded dataset to: %s", self.config.feature_store_file_path)
            return self.config.feature_store_file_path
        else:
            raise Exception("Failed to download dataset")

    def extract_zip(self, zip_path: str) -> str:
        os.makedirs(self.config.extracted_data_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.config.extracted_data_dir)
        logging.info("Extracted dataset to: %s", self.config.extracted_data_dir)
        return self.config.extracted_data_dir

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        zip_path = self.download_data()
        extract_path = self.extract_zip(zip_path)
        return DataIngestionArtifact(zip_path, extract_path)
