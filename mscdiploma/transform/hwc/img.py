from .. import base as tb
from . import KEY_hwc

import cv2

class Pipe_FileToHwc(tb.Pipe):
    KEY_CALL_IMMUT_file = 'file'
    KEY_OUT_hwc = KEY_hwc
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):
        img = data[self.KEY_OUT_hwc]
        assert len(img.shape)==3
class ReadImageCv2(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_FileToHwc
    # KEY_INIT_DTYPE = 'dtype'

    def transform(self, file):
        hwc = cv2.imread(file)#.astype(self.dtype)
        hwc = cv2.cvtColor(hwc, cv2.COLOR_BGR2RGB)
        return {
            self.PIPE_DEFAULT.KEY_OUT_hwc: hwc,
        }

    # def __set_dtype(self, dtype):
    #     self.dtype = np.dtype(dtype)

