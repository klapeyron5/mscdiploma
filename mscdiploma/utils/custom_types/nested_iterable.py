from .common_types import is_iterable


def getitem(arr, indexes):
    for i in indexes:
        arr = arr.__getitem__(i)
    return arr


def setitem(arr, indexes, new_v):
    getitem(arr, indexes[:-1])[indexes[-1]] = new_v
    return arr


def _init__is_item(is_item=None):
    if is_item is not None:
        assert callable(is_item)
        return is_item
    else:
        def is_item(item):
            return not is_iterable(item)
        return is_item


def transform_nested(a, is_item=None, transform_item=None):
    is_item = _init__is_item(is_item)
    if transform_item is not None:
        assert callable(transform_item)
    else:
        transform_item = lambda x: x

    itr = iter(a)
    checkpoints = [itr]
    checkpoints_i = [-1]
    while True:
        try:
            item = next(itr)
            checkpoints_i[-1] += 1
        except StopIteration:
            del checkpoints[-1]
            del checkpoints_i[-1]
            if len(checkpoints) == 0: break
            itr = iter(checkpoints[-1])
            continue

        if is_item(item):
            setitem(a, checkpoints_i, transform_item(item))
            continue
        else:
            itr = iter(item)
            checkpoints.append(itr)
            checkpoints_i.append(-1)
            assert is_iterable(item)
    return a


def flatten_nested(a, is_item=None, transform_item_to_into_nested_list=None):
    is_item = _init__is_item(is_item)

    itr = iter(a)
    checkpoints = [itr]
    while True:
        try:
            item = next(itr)
        except StopIteration:
            del checkpoints[-1]
            if len(checkpoints) == 0: break
            itr = iter(checkpoints[-1])
            continue

        if is_item(item):
            yield item
            if transform_item_to_into_nested_list is None:
                continue
            else:
                item = transform_item_to_into_nested_list(item)
        itr = iter(item)
        checkpoints.append(itr)
        assert is_iterable(item)


def ut_0():
    a = [[1, 2], [[3, 4]], [[[[[5, 6]]], [99]], [111]]]

    true_items = [1, 2, 3, 4, 5, 6, 99, 111]

    def is_item(x):
        return isinstance(x, int)

    def tranform_item(x):
        x += 1
        return x

    flattened_a = [x for x in flatten_nested(a, is_item=None)]
    assert set(flattened_a) - set(true_items) == set()
    assert all([x==y for x,y in zip(true_items, flattened_a)])
    transform_nested(a, is_item, tranform_item)
    flattened_transformed_a = [x for x in flatten_nested(a, is_item)]
    true_transformed_items = [tranform_item(x) for x in flattened_a]
    assert set(flattened_transformed_a) - set(true_transformed_items) == set()
    pass


if __name__ == '__main__':
    ut_0()
