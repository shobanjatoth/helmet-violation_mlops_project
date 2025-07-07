# -----------------------
# model_trainer.py
# -----------------------
import os
import subprocess
import yaml
import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataValidationArtifact, ModelTrainerArtifact

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def patch_coco_yaml(yaml_path: str) -> str:
    """
    Updates the 'train' and 'val' image directory paths in the provided YAML file.
    """
    logging.info(f"🔧 Patching YAML: {yaml_path}")
    
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    base_dir = os.path.abspath(os.path.dirname(yaml_path)).replace("\\", "/")
    data['train'] = os.path.join(base_dir, "train", "images").replace("\\", "/")
    data['val'] = os.path.join(base_dir, "val", "images").replace("\\", "/")

    with open(yaml_path, 'w') as file:
        yaml.dump(data, file)

    logging.info("✅ Patched paths:\n  train: %s\n  val: %s", data['train'], data['val'])
    return yaml_path

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, data_artifact: DataValidationArtifact):
        self.config = config
        self.data_artifact = data_artifact

    def train_model(self) -> ModelTrainerArtifact:
        """
        Trains a YOLO model using the specified configuration and returns the artifact.
        """
        command = [
            "yolo", "detect", "train",
            f"data={self.config.data_yaml_path}",
            f"model={self.config.pretrained_model_path}",
            f"epochs={self.config.epochs}",
            f"imgsz={self.config.image_size}",
            f"batch={self.config.batch_size}",
            f"lr0={self.config.lr0}",
            f"patience={self.config.patience}",
            f"device={self.config.device}",
            f"optimizer={self.config.optimizer}",
            f"seed={self.config.seed}",
            f"project={self.config.output_dir}",
            f"name={self.config.run_name}",
            f"exist_ok={str(self.config.exist_ok).lower()}",
            f"auto_augment={self.config.auto_augment}",
            f"scale={self.config.scale}",
            f"translate={self.config.translate}",
            f"erasing={self.config.erasing}",
            f"copy_paste={self.config.copy_paste}",
            "freeze=10"  # To reduce compute on CPU and speed up convergence
        ]

        logging.info("🚀 Starting YOLO training with command:\n%s", ' '.join(command))
        subprocess.run(command, check=True)

        model_path = os.path.join(
            self.config.output_dir,
            self.config.run_name,
            "weights",
            "best.pt"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

        logging.info("✅ Model training completed. Model saved at: %s", model_path)
        return ModelTrainerArtifact(model_dir=model_path)

