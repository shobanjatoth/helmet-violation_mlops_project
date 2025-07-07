from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    extracted_data_dir: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    validated_data_path: str
    error_message: str = ""

@dataclass
class ModelTrainerArtifact:
    model_dir: str


@dataclass
class ModelEvaluationArtifact:
    success: bool
    message: str