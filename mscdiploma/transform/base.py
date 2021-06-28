from abc import ABC, abstractmethod
from multiprocessing import Pool as ThreadPool
import multiprocessing
from typing import Optional, Callable
import numpy as np
from ..utils.timings import get_time
from copy import deepcopy
import inspect
import os
import pickle
from ..utils.custom_types.nested_iterable import flatten_nested
from ..utils.hash import get_md5_adler32
CPU_CORES = multiprocessing.cpu_count()
# TODO full list of all zero-lvl call keys
# TODO unmutable


"""
Framework "Transform":

functional pipeline
full control of data-flow through any transformations

Base unit is Transform class
class Transform:
    part 1:
    formalize __init__ process
        static keys:
        KEY_INIT_x = 'x'
    part 2:
    formalize __call__ process
        static keys:
        KEY_CALL_x = 'x'
        KEY_OUT_x = 'x'

class BatchTransformMultiThread:
    allows to process samples using all cpu cores
    each sample is processed with any Transform

Series-augmentation opportunity
"""


# def _check_lvl_key(cls, attr_name, constant_startswith):
#     if attr_name.startswith(constant_startswith):
#         attr = getattr(cls, attr_name)
#         assert isinstance(attr, str)
#
#         def has_key_lvl_number(attr_name: str, constant_startswith: str, cls):
#             assert attr_name.startswith(constant_startswith)
#             k = attr_name[len(constant_startswith):]
#             c0 = k.startswith("lvl")
#             if c0:
#                 k = k[len("lvl"):]
#                 split = k.split("_")
#                 assert len(split) > 1, f"Follow this format for key {cls}.{attr_name}: 'lvl{{X}}_{{key_name}}'"
#                 lvl_v = split[0]
#                 try:
#                     lvl_v = int(lvl_v)
#                     assert lvl_v >= 0
#                     c1 = True
#                 except Exception:
#                     c1 = False
#                 return {
#                     'has_lvl': c0 and c1,
#                     'lvl_v': lvl_v,
#                     'key_v': k[len(str(lvl_v)) + len("_"):],
#                 }
#             else:
#                 return {
#                     'has_lvl': False,
#                     'key_v': attr_name[len(constant_startswith):],
#                 }
#
#         out = has_key_lvl_number(attr_name, constant_startswith, cls)
#         if out['has_lvl']:
#             lvl_v = out['lvl_v']
#         key_v = out['key_v']
#         assert attr == key_v, f"Problems with static attr {cls.__name__}.{attr_name}"
#         return attr
#     return None

def _check_key(cls, attr_name, constant_startswith):
    if attr_name.startswith(constant_startswith):
        attr = getattr(cls, attr_name)
        assert isinstance(attr, str)
        assert attr == attr_name[len(constant_startswith):], \
            "Problems with static attr '{}; in class '{}'".format(attr_name, cls)
        return attr
    return None

def _get_static_keys(cls, constant_startswith, check_key_func: Optional[Callable]):
    if not inspect.isclass(cls):
        cls = cls.__class__
    assert inspect.isclass(cls)
    assert isinstance(constant_startswith, str)
    constants_attr_vals = []

    def get_attr_vals(cls):
        assert inspect.isclass(cls)
        constants_attr_vals = []
        for attr_name in cls.__dict__:
            attr = check_key_func(cls, attr_name, constant_startswith)
            if attr is not None: constants_attr_vals.append(attr)
        return constants_attr_vals

    for out in flatten_nested([cls, ], is_item=lambda item: inspect.isclass(item),
                              transform_item_to_into_nested_list=lambda item: item.__bases__):
        constants_attr_vals.extend(get_attr_vals(out))
    constants_attr_vals_set = set(constants_attr_vals)
    assert len(constants_attr_vals) == len(constants_attr_vals_set)
    return constants_attr_vals_set


class KeysChecker(ABC):
    def __init__(self, keys: set):
        assert isinstance(keys, set)
        assert all([isinstance(x, str) for x in keys])
        self.keys = keys

    def __call__(self, **kwargs):
        data_internal, data_external = {}, {}
        for k in self.keys:
            try:
                data_internal[k] = kwargs[k]
            except KeyError:
                print("ERROR: {} not in kwargs\nself.keys: {}\nkwargs.keys(): {}".format(k, self.keys, kwargs.keys()))
                raise KeyError
            del kwargs[k]
        data_external = kwargs
        return data_internal, data_external


class _PipeLight(ABC):
    """call/out (or pipe) KeysChecker"""
    def __init__(self, transform):
        self.__set_body(transform)

    def __call__(self, **data):
        return self._transform(**data)

    def __set_body(self, transform):
        self._transform = transform  # TODO check name of func; check func class is Transform uinstance


class _Pipe(_PipeLight):
    def __init__(self, transform, keys_call_immut, keys_call_mut, keys_out):
        assert set(keys_call_immut) & set(keys_call_mut) == set()
        assert set(keys_out) & set(keys_call_immut) == set()

        self.call_keys_checker = KeysChecker(keys=set(keys_call_immut)|set(keys_call_mut))
        self.call_mut_keys_checker = KeysChecker(keys=keys_call_mut)
        self.out_keys_checker = KeysChecker(keys=keys_out)
        _PipeLight.__init__(self, transform)

    def __call__(self, **data):
        data, data_call_immut, data_call_mut = self.before(data)
        # data_call_mut and data_call_immut
        for k in data_call_immut:
            data_call_mut[k] = deepcopy(data_call_immut[k])
        data_out = _PipeLight.__call__(self, **data_call_mut)
        data = self.after(data, data_call_immut, data_out)
        return data
    def before(self, data):
        data_call, data_ext = self.call_keys_checker(**data)
        assert set(data_call.keys()) & set(data_ext.keys()) == set()  # TODO delete
        assert set(data_call.keys()) | set(data_ext.keys()) == set(data.keys())  # TODO delete
        data = data_ext

        data_call_mut, data_call_immut = self.call_mut_keys_checker(**data_call)
        assert set(data_call_mut.keys()) & set(data_call_immut.keys()) == set()  # TODO delete
        assert set(data_call_mut.keys()) | set(data_call_immut.keys()) == set(data_call.keys())  # TODO delete
        return data, data_call_immut, data_call_mut
    def after(self, data, data_call_immut, data_out):
        data_out, _data = self.out_keys_checker(**data_out)
        assert len(_data.keys()) == 0, _data.keys()
        assert set(data_out.keys()) & set(data_call_immut.keys()) == set()  # TODO delete
        common_keys = set(data_out.keys()) & set(data.keys())
        assert common_keys == set(), 'Common keys: {}'.format(common_keys)
        data.update(data_call_immut)
        data.update(data_out)
        return data


class Pipe(_Pipe):  # TODO make calling assertions checking as flag (cas of deepcopy)
    """call/out (or pipe) KeysChecker"""

    def __init__(self, transform):
        _Pipe.__init__(self,
                       transform=transform,
                       keys_call_immut=_get_static_keys(self, 'KEY_CALL_IMMUT_', _check_key),
                       keys_call_mut=_get_static_keys(self, 'KEY_CALL_MUT_', _check_key),
                       keys_out=_get_static_keys(self, 'KEY_OUT_', _check_key))

    def __call__(self, **data):
        data, data_call_immut, data_call_mut = self.before(data)
        self.before_keys_assertions(**deepcopy(data_call_mut))
        # data_call_mut and data_call_immut
        for k in data_call_immut:
            data_call_mut[k] = deepcopy(data_call_immut[k])
        data_out = _PipeLight.__call__(self, **data_call_mut)
        self.after_keys_assertions(**deepcopy(data_out))
        data = self.after(data, data_call_immut, data_out)
        return data

    @abstractmethod
    def before_keys_assertions(self, **data):
        pass
    @abstractmethod
    def after_keys_assertions(self, **data):
        pass


class PipeDefault(Pipe):
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):pass


class PipeInnerRename:
    class _PipeInnerRename(_PipeLight):
        def __init__(self, transform, dict_ext_int_keys: dict):
            self.dict_ext_int_keys = dict_ext_int_keys
            self.dict_int_ext_keys = dict(zip(self.dict_ext_int_keys.values(), self.dict_ext_int_keys.keys()))
            _PipeLight.__init__(self, transform)
        def __call__(self, **data):
            data = self._replace(data, self.dict_ext_int_keys)
            data = _PipeLight.__call__(self, **data)
            data = self._replace(data, self.dict_int_ext_keys)
            return data
        @staticmethod
        def _replace(data, D):
            new_keys = []
            old_keys = []
            for data_k in data:
                if data_k in D:
                    k = D[data_k]
                    assert k not in data, "{} should not be in {}".format(k, data.keys())
                    new_keys.append(k)
                    old_keys.append(data_k)
            for new_k, old_k in zip(new_keys, old_keys):
                data[new_k] = data[old_k]
                del data[old_k]
            return data
    def __init__(self, dict_ext_int_keys: dict):
        exts = list(dict_ext_int_keys.keys())
        ints = list(dict_ext_int_keys.values())
        assert all([isinstance(x, str) for x in exts+ints]), exts+ints
        assert len(set(ints))==len(ints), ints
        self.dict_ext_int_keys = dict_ext_int_keys
    def __call__(self, transform):
        return self._PipeInnerRename(transform=transform, dict_ext_int_keys=self.dict_ext_int_keys)


class PipeReproduceInBatch:
    class _PipeReproduceInBatch(_PipeLight):
        def __init__(self, transform, keys_to_reproduce: set):
            self.keys_to_reproduce = keys_to_reproduce
            assert isinstance(self.keys_to_reproduce, set)
            assert all([isinstance(x, str) for x in self.keys_to_reproduce])
            _PipeLight.__init__(self, transform)
        def __call__(self, **data):
            result = set(self.keys_to_reproduce|{KEY_batch,})-set(data.keys())
            assert result==set(), result

            for s in data[KEY_batch]:
                for k in self.keys_to_reproduce:
                    assert k not in s, "{} should not be in {}".format(k, s.keys())
                    s[k] = data[k]
            data = _PipeLight.__call__(self, **data)  # TODO check transforms in batch (sample transforms) are using these reproduced keys only as IMMUT
            for s in data[KEY_batch]:
                for k in self.keys_to_reproduce:
                    # assert k in s, "{} should be in {}".format(k, s.keys())
                    if k in s:
                        del s[k]
            return data
    def __init__(self, keys_to_reproduce: set):
        self.keys_to_reproduce = keys_to_reproduce
    def __call__(self, transform):
        return self._PipeReproduceInBatch(transform=transform, keys_to_reproduce=self.keys_to_reproduce)


class PipeClearData:
    class _PipeClearData(_PipeLight):
        def __init__(self, transform, remaining_data: set):
            self.remaining_data = remaining_data
            assert isinstance(self.remaining_data, set)
            assert all([isinstance(x, str) for x in self.remaining_data])
            _PipeLight.__init__(self, transform)
        def __call__(self, **data):
            data = _PipeLight.__call__(self, **data)
            keys = set(data.keys())
            results = self.remaining_data - keys
            assert results==set(), "{} these keys should be in output data".format(results)
            for k in keys-self.remaining_data:
                del data[k]
            assert set(data.keys())==self.remaining_data
            return data
    def __init__(self, remaining_data: set):
        self.remaining_data = remaining_data
    def __call__(self, transform):
        return self._PipeClearData(transform=transform, remaining_data=self.remaining_data)


class Transform(ABC):
    """
    0th pipe is the outermost one, -1th pipe is the innermost one
    """
    KEY_INIT_pipes = 'pipes'

    PIPE_DEFAULT = _PipeLight

    def __init__(self, **config):
        config.setdefault(self.KEY_INIT_pipes, [])

        config_keys_checker = KeysChecker(keys=_get_static_keys(self, 'KEY_INIT_', _check_key))
        config, config_external = config_keys_checker(**config)
        assert set(config_external.keys()) == set(), config_external.keys()
        # TODO separate pipes key from _init_
        self._init_(**config)

        # config.update(config_external)

        # setup default pipe
        self.transform = self.PIPE_DEFAULT(transform=self.transform)
        assert isinstance(self.transform, _PipeLight)

        # setup custom pipe
        for pipe in config[self.KEY_INIT_pipes][::-1]:
            if pipe is not None:
                self.transform = pipe(transform=self.transform)
        assert isinstance(self.transform, _PipeLight)

    @abstractmethod
    def _init_(self, **config):
        """"""

    def __call__(self, randomness_src=None, **data):
        self._setup_data_independent_randomness(randomness_src=randomness_src)
        return self.transform(**data)

    def transform(self, **data):  # TODO make abstract for TransformNoDataIndependentRandomness (exception when child does not have this method)
        """"""

    @abstractmethod
    def generate_data_independent_randomness(self) -> dict:
        """"""

    @abstractmethod
    def get_data_independent_randomness(self) -> dict:
        """"""

    @abstractmethod
    def set_data_independent_randomness(self, **randomness):
        """"""

    def _setup_data_independent_randomness(self, randomness_src: Optional["Transform"] = None):
        if randomness_src is None:
            return self.set_data_independent_randomness(**self.generate_data_independent_randomness())
        else:
            return self._copy_data_independent_randomness_from(randomness_src)

    def _copy_data_independent_randomness_from(self, randomness_src: "Transform"):
        assert type(randomness_src) == type(self)
        self.set_data_independent_randomness(**randomness_src.get_data_independent_randomness())


class TransformNoDataIndependentRandomness(Transform, ABC):
    """
    Template for Transform without data independent randomness.
    """
    def _init_(self, **config): pass

    def generate_data_independent_randomness(self) -> dict:
        return dict()

    def get_data_independent_randomness(self) -> dict:
        return dict()

    def set_data_independent_randomness(self, **randomness):
        """"""

def init_transform(op_class_config):
    op_class, op_config = op_class_config
    assert hasattr(op_class, '__init__')
    try:
        transform = op_class(**op_config)
    except Exception as e:
        print('op_class:', op_class)
        print('op_config:', op_config)
        print(e)
        raise Exception
    assert isinstance(transform, Transform)
    return transform


class TransformPlug(TransformNoDataIndependentRandomness):
    """
    Blank Transform to use your custom Pipe.
    """
    def transform(self, **data): return data


class Forker(Transform):
    """
    Runs one fork between many forks with equal probability.
    """
    KEY_INIT_forks = 'forks'

    def _init_(self, **config):
        self.__init_forks(config[self.KEY_INIT_forks])

    def transform(self, *args, **kwargs):
        return self.fork(*args, **kwargs)

    def generate_data_independent_randomness(self) -> dict:
        return dict(
            fork = np.random.choice(self.forks)
        )

    def get_data_independent_randomness(self) -> dict:
        return dict(
            fork = self.fork
        )

    def set_data_independent_randomness(self, fork):
        self.fork = fork

    def __init_forks(self, forks):
        assert isinstance(forks, list)
        self.forks = []
        for fork in forks:
            fork = TransformPipeline(**{
                TransformPipeline.KEY_INIT_transforms: fork,
            })
            self.forks.append(fork)


class TransformPipeline(Transform):
    """
    Runs sequence of Transforms.
    """
    PIPE_DEFAULT = _PipeLight

    KEY_INIT_transforms = 'transforms'

    def _init_(self, **config):
        ops = config[self.KEY_INIT_transforms]
        self.ops = [init_transform(op) for op in ops]

    def transform(self, **data):
        for op in self.ops:
            data = op.transform(**data)
        return data

    def generate_data_independent_randomness(self) -> dict:
        return dict(randomness=[op.generate_data_independent_randomness() for op in self.ops])

    def get_data_independent_randomness(self) -> dict:
        return dict(randomness=[op.get_data_independent_randomness() for op in self.ops])

    def set_data_independent_randomness(self, randomness):
        assert len(randomness) == len(self.ops)
        for op, r in zip(self.ops, randomness):
            op.set_data_independent_randomness(**r)


KEY_batch = 'batch'


class BatchPipe(Pipe):
    KEY_CALL_MUT_batch = KEY_batch
    KEY_OUT_batch = KEY_batch
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):pass


class InitSampleTransform:
    KEY_INIT_sample_transform_cls_cnfg = 'sample_transform_cls_cnfg'


class BatchTransformSingleThread(TransformNoDataIndependentRandomness, InitSampleTransform):
    """
    Processes batch with a single cpu-process.
    """
    PIPE_DEFAULT = BatchPipe

    def _init_(self, **config):
        assert len(config[self.KEY_INIT_sample_transform_cls_cnfg]) == 2
        sample_transform_class, sample_transform_config = config[self.KEY_INIT_sample_transform_cls_cnfg]
        assert isinstance(sample_transform_config, dict)
        self.thread_transform = sample_transform_class(**sample_transform_config)
        assert isinstance(self.thread_transform, Transform)

    def transform(self, batch):
        return {self.PIPE_DEFAULT.KEY_OUT_batch: self.transform_body(batch)}

    def transform_body(self, batch, thread_transform_randomness_src: Transform=None):
        return [self.thread_transform(randomness_src=thread_transform_randomness_src, **x) for x in batch]


class _BatchPipeCycle:
    class _BatchPipeCycle(_PipeLight):
        def __init__(self, transform, global_batch_vars: set):
            """
            :param global_batch_vars: through-batch global vars
            """
            self.global_batch_vars = global_batch_vars
            assert isinstance(self.global_batch_vars, set)
            assert all([isinstance(x, str) for x in self.global_batch_vars])
            _PipeLight.__init__(self, transform)

        def __call__(self, **data):
            assert KEY_batch not in self.global_batch_vars
            result = set(self.global_batch_vars | {KEY_batch, }) - set(data.keys())
            assert result == set(), result
            data = _PipeLight.__call__(self, **data)  # TODO check transforms in batch (sample transforms) are using these reproduced keys only as IMMUT
            return data
    def __init__(self, global_batch_vars):
        self.global_batch_vars = global_batch_vars
    def __call__(self, transform):
        return self._BatchPipeCycle(transform=transform, global_batch_vars=self.global_batch_vars)
class BatchTransformInCycle(TransformNoDataIndependentRandomness, InitSampleTransform):
    """
    Processes batch in cycle:
    use common data for each sample from the same key-level as KEY_batch
    """
    PIPE_DEFAULT = None
    KEY_INIT_global_batch_vars = 'global_batch_vars'

    def _init_(self, **config):
        assert len(config[self.KEY_INIT_sample_transform_cls_cnfg]) == 2
        sample_transform_class, sample_transform_config = config[self.KEY_INIT_sample_transform_cls_cnfg]
        assert isinstance(sample_transform_config, dict), sample_transform_config
        self.thread_transform = sample_transform_class(**sample_transform_config)
        assert isinstance(self.thread_transform, Transform)
        self.PIPE_DEFAULT = _BatchPipeCycle(global_batch_vars=config[self.KEY_INIT_global_batch_vars])

    def transform(self, batch, **data):
        return self.transform_body(batch, **data)

    def transform_body(self, batch, thread_transform_randomness_src: Transform=None, **data):
        out_batch = []
        keys_through_cycle = self.PIPE_DEFAULT.global_batch_vars
        for x in batch:
            for k in keys_through_cycle:
                assert k not in x
                x[k] = data[k]
            out = self.thread_transform(randomness_src=thread_transform_randomness_src, **x)
            for k in keys_through_cycle:
                assert k in out
                data[k] = out[k]
                del out[k]
            out_batch.append(out)
        data[KEY_batch] = out_batch
        return data


class BatchTransformMultiThread(TransformNoDataIndependentRandomness, InitSampleTransform):
    """
    Processes batch with multithreading on cpu cores.
    """
    PIPE_DEFAULT = BatchPipe

    def _init_(self, **config):
        self.threads_batch_transforms = [BatchTransformSingleThread(**config) for _ in range(CPU_CORES)]
        self.threads_pool = ThreadPool(CPU_CORES)

    @staticmethod
    def transform_static(batch, threads_pool: ThreadPool, thread_function, threads_batch_transforms, thread_transform_randomness_src: Transform=None):  # TODO learn Optional[Callable] in Pool
        pool_size = threads_pool._processes
        batch_cpu = []
        bs = len(batch)
        n = bs // pool_size
        res = bs % pool_size
        for i in range(pool_size):  # the order of samples is saved here
            if res > 0:
                n_ = n + 1
                res -= 1
            else:
                n_ = n
            batch_cpu.append({BatchPipe.KEY_CALL_MUT_batch: batch[:n_]})
            batch[:n_] = []
        assert len(batch) == 0
        outs = threads_pool.starmap(thread_function, list(zip(threads_batch_transforms, batch_cpu, [thread_transform_randomness_src]*len(batch_cpu))))
        out_batch = []
        for out in outs:
            out_batch.extend(out)
        return out_batch

    def transform(self, batch):
        return {self.PIPE_DEFAULT.KEY_OUT_batch: self.transform_body(batch)}

    def transform_body(self, batch, thread_transform_randomness_src: Transform=None):
        return self.transform_static(batch, self.threads_pool, self._single_thread_batch_processing, self.threads_batch_transforms, thread_transform_randomness_src)

    @staticmethod
    def _single_thread_batch_processing(thread_batch_transform: BatchTransformSingleThread, batch, thread_transform_randomness_src: Transform=None):
        return thread_batch_transform.transform_body(thread_transform_randomness_src=thread_transform_randomness_src, **batch)


class Pipe_TestBatchTransformMultiVsSingleThread(PipeDefault):
    KEY_CALL_MUT_batch = BatchPipe.KEY_CALL_MUT_batch

    KEY_OUT_multithreading_time = 'multithreading_time'
    KEY_OUT_singlethreading_time = 'singlethreading_time'


class TestBatchTransformMultiVsSingleThread(TransformNoDataIndependentRandomness, InitSampleTransform):
    """
    Get times of work for multi and single threading BatchTransform to compare
    """
    PIPE_DEFAULT = Pipe_TestBatchTransformMultiVsSingleThread

    def _init_(self, **config):
        self.mthr_btf = BatchTransformMultiThread(**config)
        self.sthr_btf = BatchTransformSingleThread(**config)

    def transform(self, **data):
        print('{whtsgoingon}:'.format(whtsgoingon=type(self).__name__))

        m_time = get_time(self.mthr_btf, f_kwargs=deepcopy(data), n_avg=1, warmup=False)
        print('multi  thread time: {m_time:.02f}'.format(m_time=m_time))

        s_time = get_time(self.sthr_btf, f_kwargs=deepcopy(data), n_avg=1, warmup=False)
        print('single thread time: {s_time:.02f}'.format(s_time=s_time))

        return {
            self.PIPE_DEFAULT.KEY_OUT_multithreading_time: m_time,
            self.PIPE_DEFAULT.KEY_OUT_singlethreading_time: s_time,
        }


KEY_serie = 'serie'  # like one single series


class TimeSeriePipe(PipeDefault):
    KEY_CALL_MUT_serie = KEY_serie
    KEY_OUT_serie = KEY_serie


class SerieTransformSingleThread(TransformNoDataIndependentRandomness, InitSampleTransform):
    """
    Serie is sequence of samples.
    Sample is a dict.
    Generates the same randomness for all samples.
    Processes samples by using a single cpu-process.
    """
    PIPE_DEFAULT = TimeSeriePipe

    def _init_(self, **config):
        self.self = BatchTransformSingleThread(**config)
        self.thread_transform = self.self.thread_transform

    def transform(self, serie):
        self.thread_transform._setup_data_independent_randomness()
        return {self.PIPE_DEFAULT.KEY_OUT_serie: self.self.transform_body(serie, self.thread_transform)}


class SerieTransformMultiThread(TransformNoDataIndependentRandomness, InitSampleTransform):
    """
    Serie is sequence of samples.
    Sample is a dict.
    Generates the same randomness for all samples.
    Processes samples by using multithreading on cpu cores.
    """
    PIPE_DEFAULT = TimeSeriePipe
    def _init_(self, **config):
        self.self = BatchTransformMultiThread(**config)
        self.thread_transform = self.self.threads_batch_transforms[0].thread_transform

    def transform(self, serie):
        self.thread_transform._setup_data_independent_randomness()
        return {self.PIPE_DEFAULT.KEY_OUT_serie: self.self.transform_body(serie, self.thread_transform)}


class Pipe_TestSerieTransformMultiVsSingleThread(PipeDefault):
    KEY_CALL_MUT_serie = TimeSeriePipe.KEY_CALL_MUT_serie

    KEY_OUT_multithreading_time = Pipe_TestBatchTransformMultiVsSingleThread.KEY_OUT_multithreading_time
    KEY_OUT_singlethreading_time = Pipe_TestBatchTransformMultiVsSingleThread.KEY_OUT_singlethreading_time


class TestSerieTransformMultiVsSingleThread(TransformNoDataIndependentRandomness, InitSampleTransform):
    """
    Get times of work for multi and single threading SerieTransform to compare
    """
    PIPE_DEFAULT = Pipe_TestSerieTransformMultiVsSingleThread

    def _init_(self, **config):
        self.mthr_btf = SerieTransformMultiThread(**config)
        self.sthr_btf = SerieTransformSingleThread(**config)

    def transform(self, **data):
        print('{whtsgoingon}:'.format(whtsgoingon=type(self).__name__))

        m_time = get_time(self.mthr_btf, f_kwargs=deepcopy(data), n_avg=1, warmup=False)
        print('multi  thread time: {m_time:.02f}'.format(m_time=m_time))

        s_time = get_time(self.sthr_btf, f_kwargs=deepcopy(data), n_avg=1, warmup=False)
        print('single thread time: {s_time:.02f}'.format(s_time=s_time))

        return {
            self.PIPE_DEFAULT.KEY_OUT_multithreading_time: m_time,
            self.PIPE_DEFAULT.KEY_OUT_singlethreading_time: s_time,
        }


class Fitter(TransformNoDataIndependentRandomness):
    KEY_INIT_fittable_transform_cls = 'fittable_transform_cls'
    # KEY_INIT_fitting_transform_class_config = 'fitting_transform_class_config'
    KEY_INIT_name = 'name'
    KEY_INIT_save_dir = 'save_dir'

    KEY_CALL_actions = 'actions'

    DICT_ACTIONS = {'fit', 'save', 'load', 'transform'}

    def _init_(self, **config):
        self.fittable_transform_cls = config[self.KEY_INIT_fittable_transform_cls]
        # TODO assert isinstance(self.fittable_transform, Transform) not instance, just class
        # cls, cnfg = config[self.KEY_INIT_fitting_transform_class_config]
        # self.fitting_transform = cls(**cnfg)
        # assert isinstance(self.fitting_transform, Transform)

        self.name = config[self.KEY_INIT_name]
        assert isinstance(self.name, str)
        self.save_dir = config[self.KEY_INIT_save_dir]
        assert os.path.isdir(self.save_dir)
        self._save_file = os.path.join(self.save_dir, self.name+'.model')

    def transform(self, **data):
        actions = data[self.KEY_CALL_actions]
        assert all([x in self.DICT_ACTIONS for x in actions])
        for action in actions:  # TODO more elegant assertions
            if action == 'fit':
                # data.update({self.fittable_transform.KEY_FIT_fitting_transform: self.fitting_transform})
                cnfg = self.fittable_transform_cls.fit(**data)
                self.fittable_transform = self.fittable_transform_cls(**cnfg)
                assert isinstance(self.fittable_transform, Transform)
            elif action == 'save':
                assert isinstance(self.fittable_transform, Transform)
                pickle.dump(self.fittable_transform, open(self._save_file, 'wb'))
            elif action == 'load':
                self.fittable_transform = pickle.load(open(self._save_file, 'rb'))
                assert isinstance(self.fittable_transform, Transform)
            elif action == 'transform':
                assert isinstance(self.fittable_transform, Transform)
                data = self.fittable_transform.transform(**data)
            else:
                raise Exception
        return data


class FittableTransform:
    KEY_FIT_fitting_transform = 'fitting_transform'
    @classmethod
    @abstractmethod
    def fit(cls, **data):pass
class FittableTransformPipeline(TransformPipeline):  # TODO
    KEY_FIT_fittable_fitting_transforms_cls_cls_cnfg = 'fittable_fitting_transforms_cls_cls_cnfg'

    @classmethod
    def fit(cls, **data):
        ts = data[cls.KEY_FIT_fittable_fitting_transforms_cls_cls_cnfg]
        ops_fitting = []
        ops = []
        for t_ in ts:
            t, ft, ftcnfg = t_
            ops.append(t)
            op_fitting = ft(**ftcnfg)
            ops_fitting.append(op_fitting)

        out = []
        for t, ft in zip(ops, ops_fitting):
            data.update({t.KEY_FIT_fitting_transform: ft})
            cnfg = t.fit(**data)
            t__ = t(**cnfg)
            data = t__(**data)
            out.append([t, cnfg])
        return {
            cls.KEY_INIT_transforms: out
        }


# TODO cache all inner transforms as input parameter
# def is_data_independent_randomness(transform):  # TODO use in Cacher
#     try:
#         assert isinstance(transform, Transform)
#         r = transform.generate_data_independent_randomness()
#         assert isinstance(r, dict) and len(r)==0
#         r = transform.get_data_independent_randomness()
#         assert isinstance(r, dict) and len(r)==0
#     except Exception:
#         return False
#     return True
# TODO check all transform inside are TransformNoRandomness (either no randomness in data)
# TODO BatchTransforms are TransformNoDataIndependentRandomness but has randomness dependent on data
# TODO BatchTransforms are TransformOnlyDataDependentRandomness
# TODO can't pickle pyobdc.connection
# TODO some objects has no adequate str() representation (only name+address)
class Cacher(TransformNoDataIndependentRandomness, InitSampleTransform):
    PIPE_DEFAULT = _PipeLight

    KEY_INIT_cache_dir = 'cache_dir'
    KEY_INIT_name = 'name'
    KEY_INIT_use_cache = 'use_cache'

    KEY_CACHE_call_hash = 'call_hash'
    KEY_CACHE_out_hash = 'out_hash'
    KEY_CACHE_data_pickleable = 'data_pickleable'
    KEY_CACHE_pickleable_keys = 'pickleable_keys'
    KEY_CACHE_not_pickleable_keys = 'not_pickleable_keys'

    def _init_(self, **config):
        self.transform_ = init_transform(config[self.KEY_INIT_sample_transform_cls_cnfg])
        self.cache_dir = config[self.KEY_INIT_cache_dir]
        assert os.path.isdir(self.cache_dir)
        self.name = config[self.KEY_INIT_name]
        self.cache_file = os.path.join(self.cache_dir, self.name+'.cache')
        self.use_cache = config[self.KEY_INIT_use_cache]
        assert isinstance(self.use_cache, bool)

        self.keys_cache = _get_static_keys(self, 'KEY_CACHE_', _check_key)

    @staticmethod
    def get_dict_hash(d: dict):
        keys = sorted(d.keys())
        h = ''
        pickleable_keys = []
        not_pickleable_keys = []
        for k in keys:
            v = d[k]
            try:
                deepcopy(v)
                pickleable_keys.append(k)
                byts = (k + str(v)).encode('utf-8')
                h += get_md5_adler32(byts, True)
            except Exception:
                not_pickleable_keys.append(k)
        return h, pickleable_keys, not_pickleable_keys

    def transform(self, **data):
        call_hash, call_pickleable_keys, call_not_pickleable_keys = self.get_dict_hash(data)
        assert set(data.keys()) == set(call_pickleable_keys+call_not_pickleable_keys)

        if self.use_cache and os.path.isfile(self.cache_file):
            cache = pickle.load(open(self.cache_file, 'rb'))
            assert isinstance(cache, dict)
            assert set(cache.keys())==set(self.keys_cache)

            call_hash_loaded = cache[self.KEY_CACHE_call_hash]
            pickleable_keys_loaded = cache[self.KEY_CACHE_pickleable_keys]
            not_pickleable_keys_loaded = cache[self.KEY_CACHE_not_pickleable_keys]
            if call_hash == call_hash_loaded and \
                    set(data.keys())==set(pickleable_keys_loaded+not_pickleable_keys_loaded):
                data.update(cache[self.KEY_CACHE_data_pickleable])
                print('LOG: {}.{} loaded cache'.format(type(self).__name__, self.name))
                return data

        data = self.transform_(**data)
        out_hash, out_pickleable_keys, out_not_pickleable_keys = self.get_dict_hash(data)
        assert set(out_not_pickleable_keys)-set(call_not_pickleable_keys)==set()
        pickleable_data = {}
        for k in out_pickleable_keys:
            pickleable_data[k] = data[k]
        pickle.dump({
            self.KEY_CACHE_call_hash: call_hash,
            self.KEY_CACHE_out_hash: out_hash,
            self.KEY_CACHE_data_pickleable: pickleable_data,
            self.KEY_CACHE_pickleable_keys: call_pickleable_keys,
            self.KEY_CACHE_not_pickleable_keys: call_not_pickleable_keys,
        }, open(self.cache_file, 'wb'))
        print('LOG: {}.{} dumped cache'.format(type(self).__name__, self.name))
        return data
