import tensorflow as tf
from . import crop
from . import rnet
from . import coeffs_to_mesh

import os
RENDER_OP_LIBRARY = 'yadira_render_cpp.so'
lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
assert os.path.isdir(lib_dir), lib_dir
lib_file = os.path.join(lib_dir, RENDER_OP_LIBRARY)
assert os.path.isfile(lib_file),lib_file
render_ops = tf.load_op_library(lib_file)
assert hasattr(render_ops, 'render_colors')


class MS(tf.Module):
    def __init__(self):
        super(MS, self).__init__()
        self.crop = crop.Crop()
        self.rnet = rnet.RNet()
        self.coeffs_to_mesh = coeffs_to_mesh.CoeffsToMesh()

        self.render_colors = render_ops.render_colors

        self.get_coeffs = tf.function(
            self.get_coeffs,
            input_signature=[
                tf.TensorSpec([1, None, None, 3], tf.float32, name='images'),
                tf.TensorSpec([1, 5, 2], tf.float32, name='facial_landmarks'),
            ])

        self.get_mesh = tf.function(
            self.get_mesh,
            input_signature=[
                tf.TensorSpec([1, None, None, 3], tf.float32, name='images'),
                tf.TensorSpec([1, 5, 2], tf.float32, name='facial_landmarks'),
                tf.TensorSpec([], tf.bool),
            ])

        self.get_depthmap = tf.function(
            self.get_depthmap,
            input_signature=[
                tf.TensorSpec([1, None, None, 3], tf.float32, name='images'),
                tf.TensorSpec([1, 5, 2], tf.float32, name='facial_landmarks'),
                tf.TensorSpec([], tf.bool),
            ])

    def get_coeffs(self, b_img, b_lms2d_n5):
        """
        Returns:
            crop [b, 224, 224, 3]
            coeffs [b, 257]
        """
        img = b_img[0]
        lms2d_n5 = b_lms2d_n5[0]

        img, lms2d_n5, t, s = self.crop(img, lms2d_n5)
        coeffs = self.rnet(tf.expand_dims(img, axis=0))
        angles = coeffs[0, 224:227]  # pitch, yaw, roll
        return {
            'crop': img,
            't': t,
            's': s,
            'coeffs': coeffs,
            'angles': angles,
        }

    def get_mesh(self, b_img, b_lms2d_n5, frontalize):
        """
        Returns:
            crop [b, 224, 224, 3]
            coeffs [b, 257]
            mesh [b, N_vertices, 3]
        """
        out = self.get_coeffs(b_img, b_lms2d_n5)
        mesh = self.coeffs_to_mesh(out['coeffs'], frontalize)
        out.update({'mesh': mesh})
        return out

    def get_depthmap(self, img_bytes, b_lms2d_n5, frontalize):
        """
        Returns face info in source crop coordinates (either frontalize=True or not):
        Возвращает информацию о лице в координатах кропа (вне зависимости от frontalize=True или False)
            crop [b, 224, 224, 3]
            coeffs [b, 257]
            coeffs_251 [b, 251]
            mesh [b, N_vertices, 3]
            depthmap [b, 224, 224, 1]
        """
        out = self.get_mesh(img_bytes, b_lms2d_n5, frontalize)
        mouth = False

        def _get_depth_from_vertices_tf(vertices):
            z = vertices[:, 2:]
            z = z - tf.reduce_min(z)
            z = z / tf.reduce_max(z)
            return z

        z = _get_depth_from_vertices_tf(out['mesh'][0])
        depthmap = self.render_colors(out['mesh'][0], self.coeffs_to_mesh.get_triangles(mouth), z, 224, 224, 1)
        out.update({'depthmap': tf.expand_dims(depthmap, axis=0),})
        return out


def test():
    import os
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from copy import deepcopy
    def _get_lms2d_n5_from_txt(img_file):
        txt_file = os.path.splitext(img_file)[0]+'.txt'
        lms2d_n5 = np.loadtxt(txt_file)
        assert lms2d_n5.shape == (5, 2)
        return lms2d_n5
    def _read_img_pil(img_file):
        assert os.path.isfile(img_file)
        pil_img = Image.open(img_file)
        return pil_img

    def draw_vertices(mesh_vertices, image, vertice_color=[255, 255, 255], vertice_r=1, extra_scale=3):
        mesh_vertices *= extra_scale

        h, w, _ = image.shape
        w *= extra_scale
        h *= extra_scale
        image = cv2.resize(image, (w, h))

        def draw_r_1():
            image[y, x] = vertice_color
        def draw_r():
            cv2.circle(image, (x, y), vertice_r, vertice_color, thickness=-1)

        if vertice_r == 1:
            draw_func = draw_r_1
        else:
            draw_func = draw_r

        for vertex in mesh_vertices:
            x, y = [np.round(v).astype(int) for v in vertex[:2]]
            if 0 <= x < w and 0 <= y < h:
                draw_func()
        return image
    img_file = os.path.join(os.path.dirname(__file__), 'resources/examples/000002.jpg')
    img_file = os.path.normpath(os.path.join(os.path.dirname(__file__), '../working_dir/exmp1.jpg'))
    assert os.path.isfile(img_file)
    lms2d_n5 = _get_lms2d_n5_from_txt(img_file)

    img = _read_img_pil(img_file)
    img = np.array(img)
    plt.imshow(img); plt.show()
    img_vis = draw_vertices(deepcopy(lms2d_n5), deepcopy(img), vertice_color=[0,255,0],vertice_r=6)
    plt.imshow(img_vis); plt.show()
    img = img.astype(np.float32)

    ms = MS()
    img = np.expand_dims(img, axis=0)
    lms2d_n5 = np.expand_dims(lms2d_n5, axis=0)
    out = ms.get_depthmap(img, lms2d_n5, False)

    def _show_img(image, title='', title_color='lime', title_size=20, cmap=None):
        plt.figure(figsize=(10, 10))
        plt.suptitle(title, c=title_color, fontsize=title_size)
        plt.imshow(image, cmap=cmap)
        plt.show()

    plt.imshow(out['crop'].numpy().astype(int)); plt.show()
    mesh = out['mesh'].numpy()[0]
    vis = draw_vertices(mesh, out['crop'].numpy())
    plt.imshow(vis.astype(int)); plt.show()
    print()

from ..transform import base as tb
from ..transform import hwc
from ..fan2d import Pipe_GetLms2d5n
import numpy as np
class Pipe_MsFaceReconstruction(tb.Pipe):
    KEY_CALL_IMMUT_hwc = hwc.KEY_hwc
    KEY_CALL_IMMUT_lms2d_5n = Pipe_GetLms2d5n.KEY_OUT_lms2d_5n
    KEY_OUT_mesh = 'mesh'

    """(x,y)"""
    KEY_OUT_t = 't'

    KEY_OUT_s = 's'

    """
    pitch, yaw, roll
    """
    KEY_OUT_angles = 'angles'
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):
        mesh = data[self.KEY_OUT_mesh]
        if mesh is not None:
            assert len(mesh.shape)==2 and mesh.shape[-1]==3
            t = data[self.KEY_OUT_t]
            assert t.shape==(2,)
            s = data[self.KEY_OUT_s]
            assert s.shape==()
class MsFaceReconstruction(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_MsFaceReconstruction
    def _init_(self, **config):
        self.ms = MS()
    def transform(self, hwc, lms2d_5n):
        if lms2d_5n is not None:
            b_img = np.expand_dims(hwc, axis=0)
            b_lms2d_n5 = np.expand_dims(lms2d_5n, axis=0)
            out = self.ms.get_mesh(b_img, b_lms2d_n5, False)
            return {
                self.PIPE_DEFAULT.KEY_OUT_mesh: out['mesh'].numpy()[0],
                self.PIPE_DEFAULT.KEY_OUT_t: out['t'].numpy(),
                self.PIPE_DEFAULT.KEY_OUT_s: out['s'].numpy(),
                self.PIPE_DEFAULT.KEY_OUT_angles: out['angles'].numpy(),
            }
        else:
            return {
                self.PIPE_DEFAULT.KEY_OUT_mesh: None,
                self.PIPE_DEFAULT.KEY_OUT_t: None,
                self.PIPE_DEFAULT.KEY_OUT_s: None,
                self.PIPE_DEFAULT.KEY_OUT_angles: None,
            }
