GPU = -1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

import tensorflow as tf
from mscdiploma.ms_crop import Crop
from mscdiploma.ms_rnet import RNet

class MS(tf.Module):
    def __init__(self):
        super(MS, self).__init__()
        self.crop = Crop()
        self.rnet = RNet()

        self.get_coeffs = tf.function(
            self.get_coeffs,
            input_signature=[
                tf.TensorSpec([None], tf.string, name='images_bytes'),
                tf.TensorSpec([1, 5, 2], tf.float32, name='facial_landmarks'),
            ])

    def get_coeffs(self, img_bytes, b_lms2d_n5):
        """
        Returns:
            crop [b, 224, 224, 3]
            coeffs [b, 257]
        """

        img = tf.io.decode_jpeg(img_bytes[0], channels=3)
        img = tf.cast(img, tf.float32)
        # img = bhwc[0]
        lms2d_n5 = b_lms2d_n5[0]

        img, lms2d_n5 = self.crop(img, lms2d_n5)
        coeffs = self.rnet(tf.expand_dims(img, axis=0))

        return {
            'crop': img,
            'coeffs': coeffs
        }
