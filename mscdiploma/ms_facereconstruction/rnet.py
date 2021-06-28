import tensorflow as tf
from .resource_manager import ResourceManager


class RNet(tf.Module):
    def __init__(self):
        super(RNet, self).__init__()
        model = tf.saved_model.load(ResourceManager._get_res_path(ResourceManager.RES_Ms_Deep3DFaceReconstruction_RNet))
        self.model = model.signatures["serving_default"]

        self.__call__ = tf.function(
            self.__call__,
            input_signature=[
                tf.TensorSpec([None, None, None, 3], tf.float32),
            ])

    def __call__(self, bhwc):
        """
        Args:
            bhwc: c==RGB
        Returns:
            coeffs
        """
        return self.model(bhwc[:, :, :, ::-1])["resnet/coeff"]
