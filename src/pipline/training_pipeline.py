# -----------------------
# training_pipeline.py
# -----------------------

import yaml
import logging
import os
from src.entity.config_entity import *
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer, patch_coco_yaml
from src.components.model_evaluation import ModelEvaluator
from src.entity.artifact_entity import *

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def start_training_pipeline():
    try:
        with open("config/config.yaml") as f:
            config_data = yaml.safe_load(f)

        ingestion_cfg = config_data.get("data_ingestion", {})

        pipeline_config = TrainingPipelineConfig(
            download_url=ingestion_cfg.get("download_url"),
            zip_file_name=ingestion_cfg.get("zip_file_name", FILE_NAME),
            extracted_dir=ingestion_cfg.get("extracted_dir", "data")
        )

        data_ingestion_config = DataIngestionConfig(training_pipeline_config=pipeline_config)
        ingestion = DataIngestion(config=data_ingestion_config)
        ingestion_artifact = ingestion.initiate_data_ingestion()

        validation_config = DataValidationConfig(extracted_data_dir=ingestion_artifact.extracted_data_dir)
        validator = DataValidation(config=validation_config)
        validation_artifact = validator.initiate_data_validation()

        if not validation_artifact.validation_status:
            raise Exception(f"Validation failed: {validation_artifact.error_message}")

        yaml_path = os.path.join(validation_artifact.validated_data_path, "coco128.yaml")
        patched_yaml_path = patch_coco_yaml(yaml_path)

        with open(patched_yaml_path) as f:
            content = f.read()
            logging.info("📄 FINAL coco128.yaml before training:\n%s", content)

        trainer_config = ModelTrainerConfig(data_yaml_path=patched_yaml_path)
        trainer = ModelTrainer(config=trainer_config, data_artifact=validation_artifact)
        trainer_artifact = trainer.train_model()

        logging.info("✅ Training completed. Model saved at: %s", trainer_artifact.model_dir)

        # Model Evaluation
        evaluation_config = ModelEvaluationConfig(
            data_yaml_path=patched_yaml_path,
            image_size=trainer_config.image_size,
            device=trainer_config.device
        )
        evaluator = ModelEvaluator(config=evaluation_config, trainer_artifact=trainer_artifact)
        evaluation_artifact = evaluator.evaluate_model()

        if evaluation_artifact.success:
            logging.info("✅ Evaluation successful.")
        else:
            logging.warning("⚠️ Evaluation reported issues: %s", evaluation_artifact.message)

    except Exception as e:
        logging.exception("❌ Pipeline execution failed")




