from dataclass import DataClass

class Dataset(DataClass):
    file_name: str = "out.tensor"
    classes: int = 256
