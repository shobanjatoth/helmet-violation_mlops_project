import os
import logging
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact
from src.constants import *

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_required_directories(self):
        dirs = [
            os.path.join(self.config.extracted_data_dir, TRAIN_DIR_NAME, IMAGES_DIR_NAME),
            os.path.join(self.config.extracted_data_dir, TRAIN_DIR_NAME, LABELS_DIR_NAME),
            os.path.join(self.config.extracted_data_dir, VAL_DIR_NAME, IMAGES_DIR_NAME),
            os.path.join(self.config.extracted_data_dir, VAL_DIR_NAME, LABELS_DIR_NAME),
        ]
        for d in dirs:
            if not os.path.exists(d):
                raise FileNotFoundError(f"Missing directory: {d}")
        logging.info("All required directories exist.")

    def validate_label_file_content(self):
        label_dir = os.path.join(self.config.extracted_data_dir, TRAIN_DIR_NAME, LABELS_DIR_NAME)
        with open(os.path.join(self.config.extracted_data_dir, CLASSES_FILE_NAME)) as f:
            num_classes = len(f.readlines())

        for f_name in os.listdir(label_dir):
            with open(os.path.join(label_dir, f_name)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != ANNOTATION_FORMAT_YOLO:
                        raise ValueError(f"Incorrect format in {f_name}")
                    class_id = int(parts[0])
                    if not (0 <= class_id < num_classes):
                        raise ValueError(f"Invalid class_id {class_id} in {f_name}")
        logging.info("Label file content validation passed.")

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            self.validate_required_directories()
            self.validate_label_file_content()
            return DataValidationArtifact(True, self.config.extracted_data_dir)
        except Exception as e:
            logging.error("Data validation failed: %s", str(e))
            return DataValidationArtifact(False, self.config.extracted_data_dir, str(e))



