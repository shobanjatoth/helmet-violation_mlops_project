import os
from dataclasses import dataclass, field
from src.constants import *

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = ARTIFACT_DIR
    download_url: str = ""
    zip_file_name: str = FILE_NAME
    extracted_dir: str = "helmet_dataset"

@dataclass
class DataIngestionConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    data_ingestion_dir: str = field(init=False)
    feature_store_file_path: str = field(init=False)
    extracted_data_dir: str = field(init=False)
    collection_name: str = field(default=DATA_INGESTION_COLLECTION_NAME)

    def __post_init__(self):
        self.data_ingestion_dir = os.path.join(self.training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, self.training_pipeline_config.zip_file_name)
        self.extracted_data_dir = os.path.join(self.data_ingestion_dir, self.training_pipeline_config.extracted_dir)

@dataclass
class DataValidationConfig:
    extracted_data_dir: str

@dataclass
class ModelTrainerConfig:
    data_yaml_path: str
    pretrained_model_path: str = PRETRAINED_MODEL_PATH
    epochs: int = DEFAULT_EPOCHS
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    lr0: float = DEFAULT_LR0
    patience: int = DEFAULT_PATIENCE
    device: str = DEFAULT_DEVICE
    optimizer: str = DEFAULT_OPTIMIZER
    seed: int = DEFAULT_SEED
    output_dir: str = TRAIN_OUTPUT_DIR
    run_name: str = TRAIN_RUN_NAME
    exist_ok: bool = EXIST_OK
   # Augmentation
    auto_augment: str = DEFAULT_AUTO_AUGMENT
    scale: float = DEFAULT_SCALE
    translate: float = DEFAULT_TRANSLATE
    erasing: float = DEFAULT_ERASING
    copy_paste: float = DEFAULT_COPY_PASTE



@dataclass
class ModelEvaluationConfig:
    data_yaml_path: str
    image_size: int = DEFAULT_EVAL_IMAGE_SIZE
    device: str = DEFAULT_EVAL_DEVICE
