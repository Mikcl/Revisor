from typing import Any, Callable, Dict, Union

class DataClass:
    def serialize(self):
        return serialize(self)


def serialize(instance: Union[DataClass, Dict[str, Any]]):
    if isinstance(instance, DataClass):
        attributes = {key: getattr(instance, key) for key in dir(instance)
                      if not key.startswith('_') and not key.endswith('_')}
        return serialize({key: value for key, value in attributes.items() if not isinstance(value, Callable)})
    return {k: serialize(v) if isinstance(v, DataClass) else v for k, v in instance.items()}

