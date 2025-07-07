
import os
import subprocess
import logging
import yaml
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig, trainer_artifact: ModelTrainerArtifact):
        self.config = config
        self.trainer_artifact = trainer_artifact

    def evaluate_model(self) -> ModelEvaluationArtifact:
        command = [
            "yolo", "detect", "val",
            f"model={self.trainer_artifact.model_dir}",
            f"data={self.config.data_yaml_path}",
            f"imgsz={self.config.image_size}",
            f"device={self.config.device}"
        ]

        logging.info("📊 Running model evaluation with command: %s", ' '.join(command))
        try:
            subprocess.run(command, check=True)

            # Path to Ultralytics default results file
            results_path = os.path.join("runs", "detect", "val", "results.csv")
            if os.path.exists(results_path):
                logging.info("📈 Found results at: %s", results_path)
            else:
                logging.warning("⚠️ No results.csv found. Evaluation may have run without logging to file.")

            return ModelEvaluationArtifact(success=True, message="Evaluation completed.")

        except subprocess.CalledProcessError as e:
            logging.error("❌ Evaluation failed: %s", str(e))
            return ModelEvaluationArtifact(success=False, message=str(e))