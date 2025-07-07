# constants.py
# -----------------------

PIPELINE_NAME = "helmet_violation_detection"
ARTIFACT_DIR = "artifact"
FILE_NAME = "dataset.zip"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_COLLECTION_NAME = "helmet_data"

TRAIN_DIR_NAME = "train"
VAL_DIR_NAME = "val"
IMAGES_DIR_NAME = "images"
LABELS_DIR_NAME = "labels"
CLASSES_FILE_NAME = "classes.txt"
ANNOTATION_FORMAT_YOLO = 5

PRETRAINED_MODEL_PATH = "yolov8s.pt"  # from scratch
DEFAULT_EPOCHS = 30
DEFAULT_IMAGE_SIZE = 416
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR0 = 0.01
DEFAULT_PATIENCE = 5
DEFAULT_DEVICE = "cpu"
DEFAULT_OPTIMIZER = "SGD"
DEFAULT_SEED = 42
TRAIN_OUTPUT_DIR = "artifact/model_trainer"
TRAIN_RUN_NAME = "helmet_detector"
EXIST_OK = True

# Augmentation defaults
DEFAULT_AUTO_AUGMENT = "randaugment"
DEFAULT_SCALE = 0.5
DEFAULT_TRANSLATE = 0.1
DEFAULT_ERASING = 0.4
DEFAULT_COPY_PASTE = 0.2


# Model evaluation
DEFAULT_EVAL_IMAGE_SIZE = 640
DEFAULT_EVAL_DEVICE = "cpu"
