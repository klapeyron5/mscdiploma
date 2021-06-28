from .base import *
import numpy as np
import pandas as pd


class SuperSimpleTransform(TransformNoDataIndependentRandomness):
    def transform(self, **data):
        n = 10**1
        for j in range(n):
            d = 4
            v = 9.675
            for i in range(10 ** d):
                v *= 1.4
            for i in range(10 ** d):
                v **= 0.4
            for i in range(10 ** d):
                v **= 2.3
            for i in range(10 ** d):
                v **= 0.4
            for i in range(10 ** d):
                v /= 2.4
        return data


class Pipe_Op(Pipe):
    KEY_CALL_MUT_v = 'v'
    KEY_OUT_v = KEY_CALL_MUT_v
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):pass


class Op0Transform(TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_Op
    def transform(self, **data):
        data['v'] += 1
        return data
class Op1Transform(TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_Op
    def transform(self, **data):
        data['v'] += 2
        return data


class _Flip:
    def __call__(self, arr):
        return np.flip(arr, 0)  # horizontal flip

class PipeArr(Pipe):
    KEY_CALL_MUT_arr = 'arr'
    KEY_OUT_arr = KEY_CALL_MUT_arr
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):pass


class FlipRandom(Transform, _Flip):
    PIPE_DEFAULT = PipeArr

    def _init_(self, **config):
        self.transform = self.transform

    def transform(self, arr):
        if self.do_flip:
            arr = _Flip.__call__(self, arr)
        return {self.PIPE_DEFAULT.KEY_OUT_arr: arr}

    def generate_data_independent_randomness(self) -> dict:
        return dict(do_flip=np.random.randint(2))

    def get_data_independent_randomness(self) -> dict:
        return dict(do_flip=self.do_flip)

    def set_data_independent_randomness(self, do_flip):
        self.do_flip = do_flip


def test_batch_multithreading():
    tp = TestBatchTransformMultiVsSingleThread(**{TestBatchTransformMultiVsSingleThread.KEY_INIT_sample_transform_cls_cnfg: [  # TODO key_init
                SuperSimpleTransform, {}
            ]})
    batch = [{}]*1000
    data = tp(**{TestBatchTransformMultiVsSingleThread.PIPE_DEFAULT.KEY_CALL_MUT_batch: batch})  # TODO
    c0 = data[TestBatchTransformMultiVsSingleThread.PIPE_DEFAULT.KEY_OUT_multithreading_time] < data[TestBatchTransformMultiVsSingleThread.PIPE_DEFAULT.KEY_OUT_singlethreading_time]
    c1 = data[TestBatchTransformMultiVsSingleThread.PIPE_DEFAULT.KEY_OUT_multithreading_time] / data[TestBatchTransformMultiVsSingleThread.PIPE_DEFAULT.KEY_OUT_singlethreading_time] < 0.7
    assert c0
    assert c1


def test_transform_pipeline():
    tp = TransformPipeline(**{
        TransformPipeline.KEY_INIT_transforms: [
            [Op0Transform, {}],
            [Op1Transform, {}],
        ]
    })
    v = 1
    data = tp(**{Op0Transform.PIPE_DEFAULT.KEY_CALL_MUT_v: v})
    assert data[Op0Transform.PIPE_DEFAULT.KEY_CALL_MUT_v] == v + 1 + 2


def test_batch_randomness():
    sample = np.array([0, 1])
    batch_size = 1000*100
    batch = [{'arr':sample}]*batch_size  # TODO with KEY_CALL
    data = {'batch': batch}

    def test(tp, data):
        data = tp(**data)
        batch = []
        for sample in data['batch']:
            batch.append(sample[FlipRandom.PIPE_DEFAULT.KEY_OUT_arr])
        l = np.array(batch)[:, 0]
        v, c = np.unique(l, return_counts=True)
        assert len(v) == 2
        assert abs(c[0]-c[1]) < batch_size*0.1, "Just try again firstly: {}".format(c)

    tp = BatchTransformMultiThread(**{
        BatchTransformMultiThread.KEY_INIT_sample_transform_cls_cnfg: (
            FlipRandom, {}
        )
    })
    test(tp, deepcopy(data))
    tp = BatchTransformSingleThread(**{
        BatchTransformSingleThread.KEY_INIT_sample_transform_cls_cnfg: (
            FlipRandom, {}
        )
    })
    np.random.seed(10)
    test(tp, deepcopy(data))


def test_serie_randomness():
    sample = np.array([0, 1])
    batch_size = 100
    batch = [{'arr': sample}] * batch_size  # TODO with KEY_CALL
    data = {'serie': batch}

    def test(tp, data, steps=1000):
        vals = []
        for i in range(steps):
            data_out = tp(**deepcopy(data))
            serie = []
            for sample in data_out['serie']:
                serie.append(sample[FlipRandom.PIPE_DEFAULT.KEY_OUT_arr])
            vs = set(np.array(serie)[:, 0])
            assert len(vs)==1
            vals.append(list(vs)[0])
        v, c = np.unique(vals, return_counts=True)
        assert set(v) == {0, 1}, "Broken series architecture"
        assert c[0]-c[1] < steps*0.1, "Just try again firstly: {}".format(c)

    tp = SerieTransformSingleThread(**{
        SerieTransformSingleThread.KEY_INIT_sample_transform_cls_cnfg: (
            FlipRandom, {}
        )
    })
    test(tp, deepcopy(data))
    tp = SerieTransformMultiThread(**{
        SerieTransformMultiThread.KEY_INIT_sample_transform_cls_cnfg: (
            FlipRandom, {}
        )
    })
    test(tp, deepcopy(data))


def test_serie_multithreading():
    tp = TestSerieTransformMultiVsSingleThread(**{TestSerieTransformMultiVsSingleThread.KEY_INIT_sample_transform_cls_cnfg: [
                SuperSimpleTransform, {}
            ]})
    batch = [{}]*1000
    data = tp(**{'serie': batch})
    c0 = data[TestSerieTransformMultiVsSingleThread.PIPE_DEFAULT.KEY_OUT_multithreading_time] <\
         data[TestSerieTransformMultiVsSingleThread.PIPE_DEFAULT.KEY_OUT_singlethreading_time]
    c1 = data[TestSerieTransformMultiVsSingleThread.PIPE_DEFAULT.KEY_OUT_multithreading_time] /\
         data[TestSerieTransformMultiVsSingleThread.PIPE_DEFAULT.KEY_OUT_singlethreading_time] < 0.7
    assert c0
    assert c1


def test_pipe():
    class P(Pipe):
        KEY_CALL_IMMUT_other_float = 'other_float'
        KEY_CALL_MUT_float = 'float'
        KEY_OUT_float = KEY_CALL_MUT_float
        def before_keys_assertions(self, **data):
            assert isinstance(data[self.KEY_CALL_MUT_float], float)
        def after_keys_assertions(self, **data):
            pass
    class T(TransformNoDataIndependentRandomness):
        PIPE_DEFAULT = P
        def transform(self, **data):
            float_immut = data[self.PIPE_DEFAULT.KEY_CALL_IMMUT_other_float]
            float_mut = data[self.PIPE_DEFAULT.KEY_CALL_MUT_float]
            data[self.PIPE_DEFAULT.KEY_CALL_IMMUT_other_float] += 1
            return {
                self.PIPE_DEFAULT.KEY_CALL_MUT_float: float_mut+2
            }

    it = 5.0
    tp = T(**{})
    out = tp(**{
        T.PIPE_DEFAULT.KEY_CALL_MUT_float: it,
        T.PIPE_DEFAULT.KEY_CALL_IMMUT_other_float: it,
    })
    assert out[T.PIPE_DEFAULT.KEY_OUT_float] == it+2
    assert out[T.PIPE_DEFAULT.KEY_CALL_IMMUT_other_float] == it

    key_hi = 'key_hi'
    out = tp(**{
        T.PIPE_DEFAULT.KEY_CALL_MUT_float: it,
        T.PIPE_DEFAULT.KEY_CALL_IMMUT_other_float: it,
        key_hi: "I'm the Key",
    })

    try:
        tp = T(**{})
        out = tp(**{
            T.PIPE_DEFAULT.KEY_CALL_MUT_float: it,
        })
    except Exception:
        pass
    else:
        raise Exception

    ###################
    class P(Pipe):
        KEY_CALL_IMMUT_df = 'df'
        def before_keys_assertions(self, **data):
            pass
        def after_keys_assertions(self, **data):
            pass
    class T(TransformNoDataIndependentRandomness):
        PIPE_DEFAULT = P
        def transform(self, df):
            df['new_col'] = np.zeros(shape=(len(df),), dtype=int)
            return {}
    df = pd.DataFrame({'a':[1,2,3,5]})
    assert set(df.columns) == {'a'}
    tp = T(**{})
    out = tp(**{
        T.PIPE_DEFAULT.KEY_CALL_IMMUT_df: df,
    })
    assert set(df.columns) == {'a'}
    assert set(out['df'].columns) == {'a'}

def test_cacher():
    import pandas as pd
    class P(Pipe):
        KEY_CALL_MUT_int = 'int'
        KEY_CALL_MUT_list = 'list'
        KEY_CALL_MUT_pddf = 'pddf'

        KEY_OUT_intx2 = 'intx2'
        KEY_OUT_lenlist = 'lenlist'
        KEY_OUT_dfshape = 'dfshape'

        def before_keys_assertions(self, **data):
            assert isinstance(data[self.KEY_CALL_MUT_int], int)
            assert isinstance(data[self.KEY_CALL_MUT_list], list)
            assert isinstance(data[self.KEY_CALL_MUT_pddf], pd.DataFrame)

        def after_keys_assertions(self, **data):
            assert isinstance(data[self.KEY_OUT_intx2], int)
            assert isinstance(data[self.KEY_OUT_lenlist], int)

            assert isinstance(data[self.KEY_OUT_dfshape], tuple)
            assert len(data[self.KEY_OUT_dfshape]) == 2
            assert all([isinstance(x, int) for x in data[self.KEY_OUT_dfshape]])

    class T(TransformNoDataIndependentRandomness):
        PIPE_DEFAULT = P

        def transform(self, **data):
            intx2 = 2*data[self.PIPE_DEFAULT.KEY_CALL_MUT_int]
            lenlist = len(data[self.PIPE_DEFAULT.KEY_CALL_MUT_list])
            dfshape = data[self.PIPE_DEFAULT.KEY_CALL_MUT_pddf].shape
            return {
                self.PIPE_DEFAULT.KEY_OUT_intx2: intx2,
                self.PIPE_DEFAULT.KEY_OUT_lenlist: lenlist,
                self.PIPE_DEFAULT.KEY_OUT_dfshape: dfshape,
            }

    it = 5
    lst = [1,2,3,5,'df',9.]
    pddf = pd.DataFrame({'A': [0,1,2,3,4], 'B': ['a',2,2,3,3]})
    tp = Cacher(**{
        Cacher.KEY_INIT_sample_transform_cls_cnfg: [T, {}],
        Cacher.KEY_INIT_name: 'test_cacher',
        Cacher.KEY_INIT_cache_dir: './',
        Cacher.KEY_INIT_use_cache: True,
    })
    for i in range(3):
        out = tp(**{
            T.PIPE_DEFAULT.KEY_CALL_MUT_int: it,
            T.PIPE_DEFAULT.KEY_CALL_MUT_list: lst,
            T.PIPE_DEFAULT.KEY_CALL_MUT_pddf: pddf,
        })
        assert out[T.PIPE_DEFAULT.KEY_OUT_intx2] == it*2
        assert out[T.PIPE_DEFAULT.KEY_OUT_lenlist] == len(lst)
        assert out[T.PIPE_DEFAULT.KEY_OUT_dfshape] == pddf.shape

    it = 5
    lst = [1,2,3,5,'df',9.]
    pddf = pd.DataFrame({'A': [0,1,2,3,4], 'B': ['a',2,2,3,3], 'Bb': ['a',2,2,3,3]})
    out = tp(**{
        T.PIPE_DEFAULT.KEY_CALL_MUT_int: it,
        T.PIPE_DEFAULT.KEY_CALL_MUT_list: lst,
        T.PIPE_DEFAULT.KEY_CALL_MUT_pddf: pddf,
    })
    assert out[T.PIPE_DEFAULT.KEY_OUT_intx2] == it * 2
    assert out[T.PIPE_DEFAULT.KEY_OUT_lenlist] == len(lst)
    assert out[T.PIPE_DEFAULT.KEY_OUT_dfshape] == pddf.shape


def test_renaming_pipe():
    class P0(Pipe):
        KEY_CALL_IMMUT_immut_k0 = 'immut_k0'
        KEY_CALL_MUT_mut_k0 = 'mut_k0'
        KEY_OUT_mut_k0 = 'mut_k0'

        KEY_CALL_IMMUT_immut_k1 = 'immut_k1'
        KEY_CALL_MUT_mut_k1 = 'mut_k1'
        KEY_OUT_mut_k1 = 'mut_k1'
        def before_keys_assertions(self, **data):pass
        def after_keys_assertions(self, **data):pass
    class T(TransformNoDataIndependentRandomness):
        PIPE_DEFAULT = P0
        def transform(self, **data):
            mut_k0 = 0
            mut_k1 = 0
            return {
                self.PIPE_DEFAULT.KEY_OUT_mut_k0: mut_k0,
                self.PIPE_DEFAULT.KEY_OUT_mut_k1: mut_k1,
            }
    fake_key = 'immut_k0_other_name'
    tp = T(**{
        T.KEY_INIT_pipes: [PipeInnerRename(dict_ext_int_keys={fake_key: T.PIPE_DEFAULT.KEY_CALL_IMMUT_immut_k0}),]
    })
    out = tp(**{
        # T.PIPE_DEFAULT.KEY_CALL_IMMUT_immut_k0: 10,
        fake_key: 10,
        T.PIPE_DEFAULT.KEY_CALL_MUT_mut_k0: 11,
        T.PIPE_DEFAULT.KEY_CALL_IMMUT_immut_k1: 12,
        T.PIPE_DEFAULT.KEY_CALL_MUT_mut_k1: 13,
    })
    assert out[fake_key] == 10
    assert out[T.PIPE_DEFAULT.KEY_CALL_MUT_mut_k0] == 0 == out[T.PIPE_DEFAULT.KEY_CALL_MUT_mut_k1]
    assert out[T.PIPE_DEFAULT.KEY_CALL_IMMUT_immut_k1] == 12


def test():
    test_batch_multithreading()
    test_transform_pipeline()
    test_batch_randomness()
    test_serie_randomness()
    test_serie_multithreading()
    test_pipe()
    test_cacher()
    test_renaming_pipe()


if __name__ == '__main__':
    test()
