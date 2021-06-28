from .transform import base as tb
import numpy as np
import face_alignment
from .transform import hwc


class Pipe_GetLms2d5n(tb.Pipe):
    KEY_CALL_IMMUT_hwc = hwc.KEY_hwc
    KEY_OUT_lms2d_5n = 'lms2d_5n'
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):
        lms2d_5n = data[self.KEY_OUT_lms2d_5n]
        if lms2d_5n is not None:
            assert lms2d_5n.shape == (5,2)
            assert lms2d_5n.dtype == int
class Get2dFanLms(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_GetLms2d5n
    def _init_(self, **config):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    def transform(self, **data):
        img = data[self.PIPE_DEFAULT.KEY_CALL_IMMUT_hwc]
        if img is None:
            lms5 = None
        else:
            if any([x < 32 for x in img.shape[:2]]):
                lms5 = None
            else:
                preds = self.fa.get_landmarks(img)
                if preds is None:
                    lms5 = None
                else:
                    Lm2D = preds[0]
                    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1

                    l_eye = np.mean(Lm2D[lm_idx[[1, 2]]], 0)
                    r_eye = np.mean(Lm2D[lm_idx[[3, 4]]], 0)
                    nose = Lm2D[lm_idx[0]]
                    l_mouth = Lm2D[lm_idx[5]]
                    r_mouth = Lm2D[lm_idx[6]]

                    lms5 = np.stack([l_eye, r_eye, nose, l_mouth, r_mouth], axis=0)
                    lms5 = np.round(lms5).astype(int)
        return {
            self.PIPE_DEFAULT.KEY_OUT_lms2d_5n: lms5,
        }
