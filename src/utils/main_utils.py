import os
import sys
import dill
import yaml
import numpy as np
from pandas import DataFrame
from src.logger import logging
from src.exception import HelmetDetectionException


def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HelmetDetectionException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write content to a YAML file. Replace existing file if specified.
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise HelmetDetectionException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to a file using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise HelmetDetectionException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Load a Python object from a dill file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise HelmetDetectionException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Save a numpy array to a binary .npy file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise HelmetDetectionException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load a numpy array from a .npy file.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise HelmetDetectionException(e, sys) from e


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drop specified columns from a DataFrame.
    """
    try:
        return df.drop(columns=cols, axis=1)
    except Exception as e:
        raise HelmetDetectionException(e, sys) from e
