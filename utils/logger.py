import logging
from pathlib import Path


def build_logger(output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tumor_synthesis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(Path(output_dir) / "train.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
