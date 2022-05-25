import yaml

from typing import Any, Dict, Optional
from pathlib import Path
from .dataclass import DataClass
from .model import Model
from .optimizer import Optimizer
from .dataset import Dataset

def init_class(instance: DataClass, config: Dict[str, Any]):
    for name in dir(instance):
        if name.startswith("_") or name.endswith("_") or name not in config:
            continue
        attr = getattr(instance, name)
        if isinstance(attr, DataClass):
            init_class(attr, config[name])
            continue
        setattr(instance, name, config[name])

class Eval(DataClass):
    cache: bool = False

class Context(DataClass):
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[Path] = None):
        self.optimizer = Optimizer()
        self.dataset = Dataset()
        self.model = Model()
        self.eval = Eval()

        if config_path is not None:
            config = yaml.safe_load(config_path.read_text())

        if config is not None:
            init_class(self, config)