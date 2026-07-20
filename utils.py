import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any
import os


def setup_logger(name: str = "rag_assistant", log_file: str = "logs/rag_assistant.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def generate_document_id() -> str:
    return str(uuid.uuid4())


def get_timestamp() -> str:
    return datetime.now().isoformat()


def clean_text(text: str) -> str:
    text = ' '.join(text.split())
    return text.strip()


def format_source_citation(chunk: Dict[str, Any], index: int) -> str:
    metadata = chunk.get('metadata', {})
    source = metadata.get('source', 'Unknown')
    page = metadata.get('page', 'N/A')
    
    return f"[{index + 1}] Source: {source} (Page {page})"


def calculate_token_count(text: str, chars_per_token: int = 4) -> int:
    return len(text) // chars_per_token


def truncate_text(text: str, max_length: int = 500) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def validate_api_key(api_key: str) -> bool:
    return bool(api_key and api_key.strip() and api_key != "your_api_key_here")


def format_chat_message(role: str, content: str, timestamp: str = None) -> Dict[str, str]:
    message = {
        "role": role,
        "content": content,
        "timestamp": timestamp or get_timestamp()
    }
    return message


def merge_metadata(base_metadata: Dict, additional_metadata: Dict) -> Dict:
    merged = base_metadata.copy()
    merged.update(additional_metadata)
    return merged


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default
