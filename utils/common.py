import importlib


def load_class(cls, *args, **kwargs):
    if cls is None:
        return None
    module_name, class_name = cls.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)(*args, **kwargs)
