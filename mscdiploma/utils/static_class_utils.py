import inspect


def static_init_class_decorator(cls):
    static_constructor = getattr(cls, 'static_init')
    static_constructor()
    return cls


def get_class_attrs_names(cls, constant_startswith: str):
    assert inspect.isclass(cls)
    assert isinstance(constant_startswith, str)
    constants_attrs = []
    for attr in cls.__dict__:
        if attr.startswith(constant_startswith):
            constants_attrs.append(attr)
    return constants_attrs


def get_class_attrs(cls, constant_startswith: str):
    constants_attrs = get_class_attrs_names(cls, constant_startswith)
    constants = []
    for attr in constants_attrs:
        constants.append(getattr(cls, attr))
    return constants


class StaticConstantsList:
    constants_list = []
