import yaml

from typing import Any, Dict, Optional
from pathlib import Path
from data_class import DataClass
from data_class.model import Model
from data_class.optimizer import Optimizer
from data_class.dataset import Dataset

def init_class(instance: DataClass, config: Dict[str, Any]):
    for name in dir(instance):
        if name.startswith("_") or name.endswith("_") or name not in config:
            continue
        attr = getattr(instance, name)
        if isinstance(attr, DataClass):
            init_class(attr, config[name])
            continue
        setattr(instance, name, config[name])


class Context(DataClass):
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[Path] = None):
        self.optimizer = Optimizer()
        self.dataset = Dataset()
        self.model = Model()

        if config_path is not None:
            config = yaml.safe_load(config_path.read_text())

        if config is not None:
            init_class(self, config)