import os
import numpy as np
import scipy.io as sio
from array import array
import tensorflow as tf

# define and check essential directories:
toolbox_pkg_name = 'threedsome'
toolbox_pkg_path = os.path.dirname(__file__)
errmsg = "{} doesn't seems like '{}' module path".format(toolbox_pkg_path, toolbox_pkg_name)
assert toolbox_pkg_path.endswith(toolbox_pkg_name), errmsg

resources_dir_name = 'resources'
resources_path = os.path.join(toolbox_pkg_path, resources_dir_name)
errmsg = "'{}' path {} isn't a directory".format(resources_dir_name, resources_path)
assert os.path.isdir(resources_path), errmsg


class ResourceManager:
    """RESOURCES"""
    RES_Ms_Deep3DFaceReconstruction_RNet = 'Ms_Deep3DFaceReconstruction/RNet_saved_model'

    """
    Absolute coords of 68 3D landmarks for id_MU of BFM_2009
    """
    RES_Ms_Deep3DFaceReconstruction_abs_68Lms3D = 'Ms_Deep3DFaceReconstruction/similarity_Lm3D_all.mat'

    @staticmethod
    def _get_res_path(res):
        res_path = os.path.join(resources_path, res)
        errmsg = "resource {} doesn't exist".format(res_path)
        assert os.path.isfile(res_path) or os.path.isdir(res_path), errmsg
        return res_path

    @staticmethod
    def load_Ms_Deep3DFaceReconstruction_RNet():
        res_path = ResourceManager._get_res_path(ResourceManager.RES_Ms_Deep3DFaceReconstruction_RNet)
        loaded_model = tf.saved_model.load(res_path)
        model = loaded_model.signatures["serving_default"]
        return model

    @staticmethod
    def load_Ms_Deep3DFaceReconstruction_abs68Lms3D():
        res_path = ResourceManager._get_res_path(ResourceManager.RES_Ms_Deep3DFaceReconstruction_abs_68Lms3D)
        Lm3D = sio.loadmat(res_path)
        Lm3D = Lm3D['lm']
        return Lm3D
