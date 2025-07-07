
import os
import logging
import subprocess
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

class PredictionPipeline:
    def __init__(self, model_path: str, source_path: str, output_dir: str = "artifact/predictions"):
        self.model_path = os.path.normpath(model_path)
        self.source_path = os.path.normpath(source_path)
        self.output_dir = os.path.normpath(output_dir)
        self.run_name = f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run_inference(self):
        os.makedirs(self.output_dir, exist_ok=True)

        command = [
            "yolo", "detect", "predict",
            f"model={self.model_path}",
            f"source={self.source_path}",
            f"project={self.output_dir}",
            f"name={self.run_name}",
            "conf=0.25",
            "save=True",
            "save_txt=True"
        ]

        logging.info("🔍 Running prediction: %s", ' '.join(command))
        try:
            subprocess.run(command, check=True)
            predicted_dir = os.path.join(self.output_dir, self.run_name)
            logging.info("✅ Prediction completed. Results saved at: %s", predicted_dir)
            return predicted_dir
        except subprocess.CalledProcessError as e:
            logging.error("❌ Prediction failed: %s", str(e))
            return None