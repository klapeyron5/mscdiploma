import os
import numpy as np
import scipy.io as sio
from array import array
import tensorflow as tf

# define and check essential directories:
# toolbox_pkg_name = 'threedsome'
toolbox_pkg_path = os.path.dirname(__file__)
# errmsg = "{} doesn't seems like '{}' module path".format(toolbox_pkg_path, toolbox_pkg_name)
# assert toolbox_pkg_path.endswith(toolbox_pkg_name), errmsg

resources_dir_name = 'resources'
resources_path = os.path.join(toolbox_pkg_path, resources_dir_name)
errmsg = "'{}' path {} isn't a directory".format(resources_dir_name, resources_path)
assert os.path.isdir(resources_path), errmsg


class ResourceManager:
    """RESOURCES"""
    """
    Original BFM 2009 file
    """
    RES_BFM_2009 = "BFM_2009/01_MorphableModel.mat"

    """
    id+exp basis suitable for 300W_LP parameters
    seems like it is BFM_2009+Facewarehouse
    compiled in one .mat by https://github.com/YadiraF/face3d/tree/master/examples/Data/BFM
    """
    RES_Yafira_Face3D_BFM = 'Yadira_face3d/BFM.mat'

    """
    compiled by https://github.com/YadiraF/face3d/tree/master/examples/Data/BFM
    from "original": https://github.com/anilbas/3DMMasSTN/blob/master/util/BFM_UV.mat
    """
    RES_Yadira_Face3D_UV = 'Yadira_face3d/BFM_UV.mat'

    RES_Yadira_canonical_vertices = 'Yadira_face3d/canonical_vertices.npy'
    RES_Yadira_face_ind = 'Yadira_face3d/face_ind.txt'
    RES_Yadira_triangles = 'Yadira_face3d/triangles.txt'

    """
    "original" UV-coords

    Paper: 
    A. Bas, P. Huber, W.A.P. Smith, M. Awais and J. Kittler. "3D Morphable Models as Spatial Transformer Networks".
    In Proc. ICCV Workshop on Geometry Meets Deep Learning, pp. 904-912, 2017.

    Download:
    https://github.com/anilbas/3DMMasSTN/blob/master/util/BFM_UV.mat
    """
    RES_UV_coords = 'UV/UV.mat'

    """
    Expressions PC's transformed from Facewarehouse
    Used in Microsoft paper "Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set"

    Paper:
    CNN-based Real-time Dense Face Reconstruction with Inverse-rendered Photo-realistic Face Images 
    Yudong Guo, Juyong Zhang†, Jianfei Cai, Boyi Jiang and Jianmin Zheng

    Download:
    https://github.com/Juyong/3DFace
    """
    RES_Juyong_3DFace_exp_PC = 'Exp_Juyong_3DFace/Exp_Pca.bin'

    """
    Expressions EV's transformed from Facewarehouse

    Paper:
    Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set
    Y. Deng, J. Yang, S. Xu, D. Chen, Y. Jia, and X. Tong

    Download:
    https://github.com/microsoft/Deep3DFaceReconstruction
    """
    RES_Ms_Deep3DFaceReconstruction_exp_EV = 'Ms_Deep3DFaceReconstruction/std_exp.txt'

    """
    Absolute coords of 68 3D landmarks for id_MU of BFM_2009
    """
    RES_Ms_Deep3DFaceReconstruction_abs_68Lms3D = 'Ms_Deep3DFaceReconstruction/similarity_Lm3D_all.mat'

    RES_Ms_Deep3DFaceReconstruction_RNet = 'Ms_Deep3DFaceReconstruction/RNet_saved_model'

    RES_Ms_Deep3DFaceReconstruction_BFM_exp_idx = 'Ms_Deep3DFaceReconstruction/BFM_exp_idx.mat'

    RES_Ms_Deep3DFaceReconstruction_BFM_front_idx = 'Ms_Deep3DFaceReconstruction/BFM_front_idx.mat'

    RES_Ms_Deep3DFaceReconstruction_facemodel_info = 'Ms_Deep3DFaceReconstruction/facemodel_info.mat'

    RES_Ms_Deep3DFaceReconstruction_std_exp = 'Ms_Deep3DFaceReconstruction/std_exp.txt'

    @staticmethod
    def _get_res_path(res):
        res_path = os.path.join(resources_path, res)
        errmsg = "resource {} doesn't exist".format(res_path)
        assert os.path.isfile(res_path) or os.path.isdir(res_path), errmsg
        return res_path

    @staticmethod
    def load_BFM_2009():
        res_path = ResourceManager._get_res_path(ResourceManager.RES_BFM_2009)
        original_BFM = sio.loadmat(res_path)
        id_PC = original_BFM['shapePC']  # shape basis
        id_EV = original_BFM['shapeEV']  # corresponding eigen value
        id_MU = original_BFM['shapeMU']  # mean face
        tex_PC = original_BFM['texPC']  # texture basis
        tex_EV = original_BFM['texEV']  # eigen value
        tex_MU = original_BFM['texMU']  # mean texture
        tl = original_BFM['tl']  # triangles
        # remained keys: 'segbin', 'segMM', 'segMB'
        return id_MU, id_PC, id_EV,\
               tex_MU, tex_PC, tex_EV

    @staticmethod
    def load_ms_basis():
        shapeMU, shapePC, shapeEV, texMU, texPC, texEV = ResourceManager.load_BFM_2009()

        expPC = ResourceManager.load_Juyong_3DFace_exp_PC()
        expEV = ResourceManager.load_Ms_Deep3DFaceReconstruction_std_exp()

        # transfer BFM09 to our face model

        idBase = shapePC * np.reshape(shapeEV, [-1, 199])
        idBase = idBase / 1e5  # unify the scale to decimeter
        idBase = idBase[:, :80]  # use only first 80 basis

        exBase = expPC * np.reshape(expEV, [-1, 79])
        exBase = exBase / 1e5  # unify the scale to decimeter
        exBase = exBase[:, :64]  # use only first 64 basis

        texBase = texPC * np.reshape(texEV, [-1, 199])
        texBase = texBase[:, :80]  # use only first 80 basis

        uv_coords = ResourceManager.load_UV_original()
        assert len(uv_coords) == len(idBase)/3

        # our face model is cropped align face landmarks which contains only 35709 vertex.
        # original BFM09 contains 53490 vertex, and expression basis provided by JuYong contains 53215 vertex.
        # thus we select corresponding vertex to get our face model.

        index_exp = ResourceManager.load_Ms_Deep3DFaceReconstruction_BFM_front_idx()

        index_shape = ResourceManager.load_Ms_Deep3DFaceReconstruction_BFM_exp_idx()
        index_shape = index_shape[index_exp]

        uv_coords = uv_coords[index_shape[:, 0, 0], :]

        idBase = np.reshape(idBase, [-1, 3, 80])
        idBase = idBase[index_shape, :, :]
        idBase = np.reshape(idBase, [-1, 80])

        texBase = np.reshape(texBase, [-1, 3, 80])
        texBase = texBase[index_shape, :, :]
        texBase = np.reshape(texBase, [-1, 80])

        exBase = np.reshape(exBase, [-1, 3, 64])
        exBase = exBase[index_exp, :, :]
        exBase = np.reshape(exBase, [-1, 64])

        meanshape = np.reshape(shapeMU, [-1, 3]) / 1e5
        meanshape = meanshape[index_shape, :]
        meanshape = np.reshape(meanshape, [1, -1])

        meantex = np.reshape(texMU, [-1, 3])
        meantex = meantex[index_shape, :]
        meantex = np.reshape(meantex, [1, -1])

        # other info contains triangles, region used for computing photometric loss,
        # region used for skin texture regularization, and 68 landmarks index etc.
        frontmask2_idx, skinmask, keypoints, point_buf, tri, tri_mask2 = \
            ResourceManager.load_Ms_Deep3DFaceReconstruction_facemodel_info()

        # # save our face model
        # savemat('BFM/BFM_model_front.mat',
        #         {'meanshape': meanshape, 'meantex': meantex, 'idBase': idBase, 'exBase': exBase, 'texBase': texBase,
        #          'tri': tri, 'point_buf': point_buf, 'tri_mask2': tri_mask2 \
        #             , 'keypoints': keypoints, 'frontmask2_idx': frontmask2_idx, 'skinmask': skinmask})

        id_MU = meanshape
        id_PC = idBase
        tex_MU = meantex
        tex_PC = texBase
        exp_PC = exBase
        lms3d_n68_ids = np.squeeze(keypoints).astype(np.int32) - 1
        tri = (tri-1).astype(np.int32)
        point_buf = (point_buf-1).astype(np.int32)
        assert len(uv_coords) == len(id_MU[0])/3
        return id_MU, id_PC, tex_MU, tex_PC, exp_PC, lms3d_n68_ids, tri, point_buf, uv_coords

    @staticmethod
    def load_Yafira_Face3D_BFM(identity_n: int = 199, expression_n: int = 29):
        res_path = ResourceManager._get_res_path(ResourceManager.RES_Yafira_Face3D_BFM)

        assert isinstance(identity_n, int) and identity_n > 0
        assert isinstance(expression_n, int) and expression_n > 0

        C = sio.loadmat(res_path)
        model = C['model']
        model = model[0, 0]

        keys = np.dtype(model).fields.keys()

        # change dtype from double(np.float64) to np.float32,
        # since big matrix process (especially matrix dot) is too slow in python.
        id_MU = model['shapeMU'].astype(np.float32).T[0]  # TODO don't forget Yadira made shapeMU = shapeMU + expMU; debug this
        id_PC = model['shapePC'].astype(np.float32)[:, :identity_n]
        id_EV = model['shapeEV'].astype(np.float32).T[0][:identity_n]

        exp_MU = model['expMU'].astype(np.float32).T[0]
        id_MU += exp_MU
        exp_PC = model['expPC'].astype(np.float32)[:, :expression_n]
        exp_EV = model['expEV'].astype(np.float32).T[0][:expression_n]

        tex_MU = model['texMU'].astype(np.float32).T[0]
        tex_PC = model['texPC'].astype(np.float32)
        tex_EV = model['texEV'].astype(np.float32).T[0]

        mesh_ind_68lms = (np.squeeze(model['kpt_ind']) - 1).astype(np.int32)

        # matlab start with 1. change to 0 in python.
        tri = model['tri'].T.copy(order='C').astype(np.int32) - 1
        tri_mouth = model['tri_mouth'].T.copy(order='C').astype(np.int32) - 1

        # after-check
        n_id_para = id_PC.shape[-1]
        if identity_n > n_id_para:
            errmsg = "your identity_n is {}, but this MM support only {} identity PC's".format(identity_n, n_id_para)
            raise Exception(errmsg)
        n_exp_para = exp_PC.shape[-1]
        if expression_n > n_exp_para:
            errmsg = "your expression_n is {}, but this MM support only {} expression PC's".format(expression_n,
                                                                                                   n_exp_para)
            raise Exception(errmsg)

        return id_MU, id_PC, id_EV, \
               exp_PC, exp_EV, \
               tex_MU, tex_PC, tex_EV, \
               mesh_ind_68lms, \
               tri, tri_mouth

    @staticmethod
    def load_Yadira_Face3D_UV():
        res_path = ResourceManager._get_res_path(ResourceManager.RES_Yadira_Face3D_UV)

        C = sio.loadmat(res_path)
        uv_coords = C['UV'].copy(order='C')
        return uv_coords

    @staticmethod
    def load_UV_original():
        res_path = ResourceManager._get_res_path(ResourceManager.RES_UV_coords)

        C = sio.loadmat(res_path)
        uv_coords = C['UV'].copy(order='C')
        return uv_coords

    @staticmethod
    def load_Juyong_3DFace_exp_PC():
        res_path = ResourceManager._get_res_path(ResourceManager.RES_Juyong_3DFace_exp_PC)
        n_vertex = 53215
        Expbin = open(res_path, 'rb')
        exp_dim = array('i')
        exp_dim.fromfile(Expbin, 1)
        expMU = array('f')
        expPC = array('f')
        expMU.fromfile(Expbin, 3 * n_vertex)
        expPC.fromfile(Expbin, 3 * exp_dim[0] * n_vertex)

        expPC = np.array(expPC)
        expPC = np.reshape(expPC, [exp_dim[0], -1])
        expPC = np.transpose(expPC)

        return expPC

    @staticmethod
    def load_Ms_Deep3DFaceReconstruction_abs68Lms3D():
        res_path = ResourceManager._get_res_path(ResourceManager.RES_Ms_Deep3DFaceReconstruction_abs_68Lms3D)
        Lm3D = sio.loadmat(res_path)
        Lm3D = Lm3D['lm']
        return Lm3D

    @staticmethod
    def load_Ms_Deep3DFaceReconstruction_RNet():

        res_path = ResourceManager._get_res_path(ResourceManager.RES_Ms_Deep3DFaceReconstruction_RNet)
        loaded_model = tf.saved_model.load(res_path)
        model = loaded_model.signatures["serving_default"]

        return model

    @staticmethod
    def load_Ms_Deep3DFaceReconstruction_BFM_exp_idx():
        res_path = ResourceManager._get_res_path(ResourceManager.RES_Ms_Deep3DFaceReconstruction_BFM_exp_idx)
        index_shape = sio.loadmat(res_path)
        index_shape = index_shape['trimIndex'].astype(np.int32) - 1  # starts from 0 (to 53490)
        return index_shape

    @staticmethod
    def load_Ms_Deep3DFaceReconstruction_BFM_front_idx():
        res_path = ResourceManager._get_res_path(ResourceManager.RES_Ms_Deep3DFaceReconstruction_BFM_front_idx)
        index_exp = sio.loadmat(res_path)
        index_exp = index_exp['idx'].astype(np.int32) - 1  # starts from 0 (to 53215)
        return index_exp

    @staticmethod
    def load_Ms_Deep3DFaceReconstruction_facemodel_info():
        res_path = ResourceManager._get_res_path(ResourceManager.RES_Ms_Deep3DFaceReconstruction_facemodel_info)
        other_info = sio.loadmat(res_path)
        frontmask2_idx = other_info['frontmask2_idx']
        skinmask = other_info['skinmask']
        keypoints = other_info['keypoints']
        point_buf = other_info['point_buf']
        tri = other_info['tri']
        tri_mask2 = other_info['tri_mask2']
        return frontmask2_idx, skinmask, keypoints, point_buf, tri, tri_mask2

    @staticmethod
    def load_Ms_Deep3DFaceReconstruction_std_exp():
        res_path = ResourceManager._get_res_path(ResourceManager.RES_Ms_Deep3DFaceReconstruction_std_exp)
        expEV = np.loadtxt(res_path)
        return expEV
