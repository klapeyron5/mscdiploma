import tensorflow as tf
from .resource_manager import ResourceManager


class CoeffsToMesh(tf.Module):
    def __init__(self):
        super(CoeffsToMesh, self).__init__()

        id_MU, id_PC, \
        tex_MU, tex_PC, \
        exp_PC, \
        lms3d_n68_ids, \
        self.tri, \
        point_buf, uv_coords = ResourceManager.load_ms_basis()

        _,_,_, \
        _, _, \
        _, _, _, \
        _, \
        _, self.tri_mouth = ResourceManager.load_Yafira_Face3D_BFM()


        self.id_MU = tf.Variable(id_MU, dtype=tf.float32)
        self.id_PC = tf.Variable(id_PC, dtype=tf.float32)
        self.exp_PC = tf.Variable(exp_PC, dtype=tf.float32)

        self.__call__ = tf.function(
            self.__call__,
            input_signature=[
                tf.TensorSpec([None, 257], tf.float32),
                tf.TensorSpec([], tf.bool),
            ])
        self.get_triangles = tf.function(
            self.get_triangles,
            input_signature=[
                tf.TensorSpec([], tf.bool),
            ])

    @tf.function
    def __call__(self, coeffs, frontalize):
        id_coeff, ex_coeff, tex_coeff, angles, t, gamma = self.Split_coeff(coeffs)
        if frontalize:
            angles = tf.zeros_like(angles)
        R = self.Compute_rotation_matrix(angles)
        s = 1
        return self._get_projected_mesh(id_coeff, ex_coeff, R, t, s)


    @tf.function
    def get_triangles(self, mouth):
        if mouth:
            tris = tf.concat([self.tri, self.tri_mouth], axis=0)
        else:
            tris = self.tri
        return tris


    @staticmethod
    def Split_coeff(coeff):
        id_coeff = coeff[:, :80]  # identity
        ex_coeff = coeff[:, 80:144]  # expression
        tex_coeff = coeff[:, 144:224]  # texture
        angles = coeff[:, 224:227]  # euler angles for pose
        gamma = coeff[:, 227:254]  # lighting
        translation = coeff[:, 254:257]  # translation
        return id_coeff, ex_coeff, tex_coeff, angles, translation, gamma

    def Shape_formation_block(self, id_coeff, ex_coeff):
        face_shape = tf.einsum('ij,aj->ai', self.id_PC, id_coeff) + \
                     tf.einsum('ij,aj->ai', self.exp_PC, ex_coeff) + self.id_MU

        # reshape face shape to [batchsize,N,3]
        face_shape = tf.reshape(face_shape, [tf.shape(face_shape)[0], -1, 3])
        # re-centering the face shape with mean shape
        face_shape = face_shape - tf.reshape(tf.reduce_mean(tf.reshape(self.id_MU, [-1, 3]), 0), [1, 1, 3])

        return face_shape

    def _get_projected_mesh(self, id_coeffs, exp_coeffs, R, t, s):
        size = 224 * s
        mesh = self.Shape_formation_block(id_coeffs, exp_coeffs)
        projected_mesh, z_buffer = self.Projection_layer(mesh, R, t, center=size / 2, scale=s)
        projected_mesh = tf.stack([projected_mesh[:, :, 0], size - projected_mesh[:, :, 1]], axis=2)
        projected_mesh = tf.concat([projected_mesh, z_buffer * (-1)], axis=2)
        return projected_mesh

    def Projection_layer(self, face_shape, rotation, translation, focal=1015.0, center=112.0, scale=1):
        camera_pos = tf.reshape(tf.constant([0.0, 0.0, 10.0]), [1, 1, 3])  # camera position
        reverse_z = tf.reshape(tf.constant([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0]), [1, 3, 3])

        p_matrix = tf.concat([[focal], [0.0], [center], [0.0], [focal], [center], [0.0], [0.0], [1.0]],
                                  axis=0)  # projection matrix
        p_matrix = tf.reshape(p_matrix, [1, 3, 3])

        # calculate face position in camera space
        face_shape *= scale
        face_shape_r = tf.matmul(face_shape, rotation)  # the same as face_shape.dot(rotation)
        face_shape_t = face_shape_r + tf.reshape(translation, [1, 1, 3])
        face_shape_t = tf.matmul(face_shape_t, reverse_z) + camera_pos

        # calculate projection of face vertex using perspective projection
        aug_projection = tf.matmul(face_shape_t, tf.transpose(p_matrix, [0, 2, 1]))
        face_projection = aug_projection[:, :, 0:2] / tf.reshape(aug_projection[:, :, 2],
                                                                 [1, tf.shape(aug_projection)[1], 1])
        z_buffer = tf.reshape(aug_projection[:, :, 2], [1, -1, 1])

        return face_projection, z_buffer

    def Compute_rotation_matrix(self, angles):
        n_data = tf.shape(angles)[0]

        # compute rotation matrix for X-axis, Y-axis, Z-axis respectively
        rotation_X = tf.concat([tf.ones([n_data, 1]),
                                tf.zeros([n_data, 3]),
                                tf.reshape(tf.cos(angles[:, 0]), [n_data, 1]),
                                -tf.reshape(tf.sin(angles[:, 0]), [n_data, 1]),
                                tf.zeros([n_data, 1]),
                                tf.reshape(tf.sin(angles[:, 0]), [n_data, 1]),
                                tf.reshape(tf.cos(angles[:, 0]), [n_data, 1])],
                               axis=1
                               )

        rotation_Y = tf.concat([tf.reshape(tf.cos(angles[:, 1]), [n_data, 1]),
                                tf.zeros([n_data, 1]),
                                tf.reshape(tf.sin(angles[:, 1]), [n_data, 1]),
                                tf.zeros([n_data, 1]),
                                tf.ones([n_data, 1]),
                                tf.zeros([n_data, 1]),
                                -tf.reshape(tf.sin(angles[:, 1]), [n_data, 1]),
                                tf.zeros([n_data, 1]),
                                tf.reshape(tf.cos(angles[:, 1]), [n_data, 1])],
                               axis=1
                               )

        rotation_Z = tf.concat([tf.reshape(tf.cos(angles[:, 2]), [n_data, 1]),
                                -tf.reshape(tf.sin(angles[:, 2]), [n_data, 1]),
                                tf.zeros([n_data, 1]),
                                tf.reshape(tf.sin(angles[:, 2]), [n_data, 1]),
                                tf.reshape(tf.cos(angles[:, 2]), [n_data, 1]),
                                tf.zeros([n_data, 3]),
                                tf.ones([n_data, 1])],
                               axis=1
                               )

        rotation_X = tf.reshape(rotation_X, [n_data, 3, 3])
        rotation_Y = tf.reshape(rotation_Y, [n_data, 3, 3])
        rotation_Z = tf.reshape(rotation_Z, [n_data, 3, 3])

        # R = RzRyRx
        rotation = tf.matmul(tf.matmul(rotation_Z, rotation_Y), rotation_X)

        # because our face shape is N*3, so compute the transpose of R, so that rotation shapes can be calculated as face_shape*R
        rotation = tf.transpose(rotation, perm=[0, 2, 1])

        return rotation
