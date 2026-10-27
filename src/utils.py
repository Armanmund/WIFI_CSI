import os
import json
import numpy as np
from typing import Dict, Any


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def ensure_dir(path: str) -> None:
    """Creates directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Saves results dict as JSON with numpy types converted."""
    ensure_dir(os.path.dirname(os.path.abspath(filepath)))
    with open(filepath, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)
