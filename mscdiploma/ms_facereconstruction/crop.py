import tensorflow as tf
import numpy as np
from .resource_manager import ResourceManager


class Crop(tf.Module):
    def __init__(self, interpolation='bicubic', antialias=True):
        super(Crop, self).__init__()
        self._lm3d = self.load_lm3d()
        self._A = tf.Variable(tf.zeros([2 * 5, 8]))
        self._crop_target_size = tf.constant(224, dtype=tf.int32)

        self.interpolation = interpolation
        self.antialias = antialias

        self.__call__ = tf.function(
            self.__call__,
            input_signature=[
                tf.TensorSpec([None, None, None], tf.float32),
                tf.TensorSpec([5, 2], tf.float32),
            ])

    @staticmethod
    def load_lm3d():
        """
        Returns 5 standard 3D landmarks:
        [l_eye, r_eye, nose, l_mouth, r_mouth]
        l_eye = [x,y,z] - absolute coords on id_MU BFM_2009 yadira_mesh_cpp
        """
        Lm3D = ResourceManager.load_Ms_Deep3DFaceReconstruction_abs68Lms3D()
        # calculate 5 facial landmarks using 68 landmarks
        lm_idx = tf.constant([31, 37, 40, 43, 46, 49, 55]) - 1
        l_eye = tf.reduce_mean(tf.gather(Lm3D, tf.gather(lm_idx, [1, 2])), 0)
        r_eye = tf.reduce_mean(tf.gather(Lm3D, tf.gather(lm_idx, [3, 4])), 0)
        nose = tf.gather_nd(Lm3D, tf.gather(lm_idx, [0,]))
        l_mouth = tf.gather_nd(Lm3D, tf.gather(lm_idx, [5,]))
        r_mouth = tf.gather_nd(Lm3D, tf.gather(lm_idx, [6,]))
        Lm3D = tf.stack([l_eye, r_eye, nose, l_mouth, r_mouth], axis=0)
        return Lm3D

    def POS(self, xp, x):
        npts = tf.shape(xp)[-2]

        self._A[0:2 * npts - 1:2, 0:3].assign(x)
        self._A[0:2 * npts - 1:2, 3].assign(tf.ones(shape=[5, ], dtype=self._A.dtype))

        self._A[1:2 * npts:2, 4:7].assign(x)
        self._A[1:2 * npts:2, 7].assign(tf.ones(shape=[5, ], dtype=self._A.dtype))

        b = tf.reshape(xp, [2*npts, 1])

        k = tf.linalg.lstsq(self._A, b)

        R1 = k[0:3]
        R2 = k[4:7]
        sTx = k[3]
        sTy = k[7]
        s = (tf.linalg.norm(R1) + tf.linalg.norm(R2))/2
        t = tf.stack([sTx, sTy], axis=0)
        return t, s

    def process_img(self, img, lm, t, s):
        w0, h0, _ = tf.unstack(tf.cast(tf.shape(img), tf.float32))
        # w0 = tf.cast(w0, tf.float32)
        # h0 = tf.cast(h0, tf.float32)
        k = 102./s
        w = tf.cast(w0 * k, np.int32)
        h = tf.cast(h0 * k, np.int32)

        # TODO
        img = tf.image.resize(img, (w, h), self.interpolation, antialias=self.antialias)

        s0 = tf.cast(self._crop_target_size / 2, tf.float32)

        left = tf.cast(tf.cast(w/2, dtype=tf.float32) - s0 + (t[0][0] - w0 / 2) * k, tf.int32)
        right = left + self._crop_target_size
        up = tf.cast((tf.cast(h/2, dtype=tf.float32) - s0 + ((h0 / 2 - t[1][0]) * k)), tf.int32)
        below = up + self._crop_target_size

        # tf.print('tf crop: {}, {}, {}, {}'.format(left, up, right, below))
        img = img[up:below, left:right, :]
        img = tf.clip_by_value(img, 0, 255)
        # TODO
        # img = tf.round(img)
        # img = tf.cast(img, tf.uint8)
        # img = img[:,:,::-1] #RGBtoBGR making this inside RNet input

        lm = tf.stack([lm[:, 0] - t[0][0] + w0 / 2, lm[:, 1] - t[1][0] + h0 / 2], axis=1) * k
        lm = lm - \
             tf.reshape(
                 tf.stack([
                     tf.cast(w/2, dtype=tf.float32) - s0,
                     tf.cast(h/2, dtype=tf.float32) - s0
                 ]), [1, 2])
        return img, lm, k, tf.cast(left/h, np.float), tf.cast(up/w, np.float)

    def __call__(self, img, lms2d_n5):
        w0, h0, _ = tf.unstack(tf.shape(img))
        h0 = tf.cast(h0-1, tf.float32)

        # change from image plane coordinates to 3D space coordinates(X-Y plane)
        lms2d_n5 = tf.stack([lms2d_n5[:, 0], h0 - lms2d_n5[:, 1]], axis=1)

        t, s = self.POS(lms2d_n5, self._lm3d)
        img_new, lm_new, s_new, left, top = self.process_img(img, lms2d_n5, t, s)
        lm_new = tf.stack([lm_new[:, 0], 223 - lm_new[:, 1]], axis=1)
        # trans_params = tf.stack([w0, h0, 102.0 / s, t[0], t[1]])
        return img_new, lm_new, tf.stack([left, top]), s_new


def test():  # TODO delete
    crop = Crop()
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    def _get_lms2d_n5_from_txt(img_file):
        txt_file = os.path.splitext(img_file)[0]+'.txt'
        lms2d_n5 = np.loadtxt(txt_file)
        assert lms2d_n5.shape == (5, 2)
        return lms2d_n5

    def _read_img_pil(img_file):
        assert os.path.isfile(img_file)
        pil_img = Image.open(img_file)
        return pil_img
    img_file = os.path.join(os.path.dirname(__file__), 'resources/examples/000002.jpg')
    assert os.path.isfile(img_file)
    lms2d_n5 = _get_lms2d_n5_from_txt(img_file)
    img = _read_img_pil(img_file)
    img = np.array(img)
    plt.imshow(img); plt.show()
    img = img.astype(np.float32)
    cropped_img, cropped_lms2d_n5 = crop.__call__(img=img, lms2d_n5=lms2d_n5)
    cropped_img = cropped_img.numpy()
    cropped_img = cropped_img.astype(int)
    plt.imshow(cropped_img); plt.show()



    print()
