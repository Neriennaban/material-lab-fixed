from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_hash(data: Any, algorithm: str = "sha256") -> str:
    """Generate hash from data
    
    Args:
        data: Data to hash (will be JSON serialized if not string)
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
    
    Returns:
        Hex digest of hash
    """
    if not isinstance(data, (str, bytes)):
        data = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    
    if isinstance(data, str):
        data = data.encode("utf-8")
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data)
    return hash_obj.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if not
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str, max_length: int = 255) -> str:
    """Make filename safe for filesystem
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
    
    Returns:
        Safe filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, "_")
    
    # Limit length
    if len(filename) > max_length:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        max_name_length = max_length - len(ext) - 1
        filename = f"{name[:max_name_length]}.{ext}" if ext else name[:max_length]
    
    return filename


def timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """Get current timestamp as string
    
    Args:
        format: strftime format string
    
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(format)


def format_bytes(size: int) -> str:
    """Format bytes as human-readable string
    
    Args:
        size: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {seconds:.0f}s"
    
    hours = minutes // 60
    minutes = minutes % 60
    
    return f"{hours}h {minutes}m {seconds:.0f}s"


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value between min and max
    
    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value
    
    Returns:
        Clamped value
    """
    return max(min_value, min(max_value, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b
    
    Args:
        a: Start value
        b: End value
        t: Interpolation factor (0-1)
    
    Returns:
        Interpolated value
    """
    return a + (b - a) * clamp(t, 0.0, 1.0)


def map_range(
    value: float,
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float,
) -> float:
    """Map value from one range to another
    
    Args:
        value: Input value
        in_min: Input range minimum
        in_max: Input range maximum
        out_min: Output range minimum
        out_max: Output range maximum
    
    Returns:
        Mapped value
    """
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)


def deep_merge(dict1: dict, dict2: dict) -> dict:
    """Deep merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
    
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split list into chunks
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def retry(
    func: callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """Retry function on exception
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        exceptions: Tuple of exceptions to catch
    
    Returns:
        Function result
    
    Raises:
        Last exception if all attempts fail
    """
    import time
    
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(delay)
    
    raise last_exception
