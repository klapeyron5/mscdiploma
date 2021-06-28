from nirvana.utils.static_class_utils import static_init_class_decorator, get_class_attrs_names, get_class_attrs


class Initier:
    __INIT_ARGS_SETTERS = {}
    @classmethod
    def static_init(cls):
        init_keys_start = 'KEY_INIT_'
        init_keys_names = get_class_attrs_names(cls, init_keys_start)
        init_args_names = [x[len(init_keys_start):] for x in init_keys_names]
        for key_name, arg_name in zip(init_keys_names, init_args_names):
            setattr(cls, key_name, arg_name)
        setters_start = '_SET_INIT_'
        setters_names = [setters_start+x for x in init_args_names]
        setters = [getattr(cls, x) for x in setters_names]
        cls.__INIT_ARGS_SETTERS = dict(zip(init_args_names, setters))

    def __init__(self, **config):
        for arg, setter in self.__INIT_ARGS_SETTERS.items():
            setter(self, config[arg])


@static_init_class_decorator
class Preprocessor(Initier):
    KEY_INIT_scale_standardize = None
    KEY_INIT_mean_standardize = None
    KEY_INIT_drop_any_nan_trn_cols = None

    def __init__(self, **config):
        Initier.__init__(self, **config)

    def _SET_INIT_scale_standardize(self, x):
        assert isinstance(x, bool)
        self.scale_standardize = x

    def _SET_INIT_mean_standardize(self, x):
        assert isinstance(x, bool)
        self.mean_standardize = x

    def _SET_INIT_drop_any_nan_trn_cols(self, x):
        assert isinstance(x, bool)
        self.drop_any_nan_trn_cols = x

# p = Preprocessor(**{
#     Preprocessor.KEY_INIT_mean_standardize: False,
#     Preprocessor.KEY_INIT_scale_standardize: Preprocessor.KEY_INIT_scale_standardize.T1,
#     Preprocessor.KEY_INIT_drop_any_nan_trn_cols: False,
#     Preprocessor.KEY_INIT_cat_cols_preprocess: Preprocessor.VALUE_drop
# })
# print()