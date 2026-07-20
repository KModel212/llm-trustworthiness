import os
import tempfile
import pandas as pd
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    Ensures 'id' and 'query' columns exist.
    """
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found at {file_path}")
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
        
    logger.info(f"Loading dataset from {file_path}")
    tmp_path = None
    try:
        df = pd.read_csv(
            file_path,
            dtype={"id": str, "query": str},
            keep_default_na=False,
        )
        
        if 'id' not in df.columns or 'query' not in df.columns:
            logger.error("Dataset must contain 'id' and 'query' columns.")
            raise ValueError("Dataset missing required columns.")
            
        logger.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def save_submission(results: List[Dict[str, Any]], file_path: str):
    """
    Saves the results to a CSV file.
    Ensures that only 'id' and 'response' columns are saved.
    Maintains the exact row order of the input.
    """
    logger.info(f"Saving submission to {file_path}")
    
    # Ensure directory exists
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    try:
        df = pd.DataFrame(results)
        if 'id' not in df.columns or 'response' not in df.columns:
            raise ValueError("Results missing 'id' or 'response' keys.")
            
        # Select only required columns in the correct order
        df = df[['id', 'response']]
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=dir_name or ".",
            prefix=".submission.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp, index=False)
        os.replace(tmp_path, file_path)
        logger.info(f"Successfully saved {len(df)} rows to {file_path}.")
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error(f"Error saving submission: {e}")
        raise
