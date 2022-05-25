from .dataclass import DataClass

class Dataset(DataClass):
    file_name: str = "out.tensor"
    classes: int = 256
    num_workers: int = 4
    pin_memory: bool = False
    prefetch_factor: int = 256  # 256 (Prefetch) * 8 (Long) * 2048 (GPT context) * 256 (High Batch) = 1GiB RAM