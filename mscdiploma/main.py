from ..transform import base as tb
from ..transform.hwc import img
from . import mappings, video_example, video, openpose, fan2d
from .ms_facereconstruction import main as ms_face
import numpy as np
import cv2
from copy import deepcopy
import os
import matplotlib.pyplot as plt


KEY_ImagePath = 'ImagePath'
KEY_img_frame_src = 'img_frame_src'
KEY_DetectedObjects = 'DetectedObjects'


class Pipe_AdjustTranslationForSrcImg(tb.Pipe):
    KEY_CALL_MUT_t = ms_face.MsFaceReconstruction.PIPE_DEFAULT.KEY_OUT_t
    KEY_CALL_IMMUT_head_bbox = openpose.GetBboxFromOpenposeLms.PIPE_DEFAULT.KEY_OUT_head_bbox
    KEY_CALL_IMMUT_head_crop = openpose.GetHeadCropFromHeadBbox.PIPE_DEFAULT.KEY_OUT_head_crop
    KEY_OUT_t = KEY_CALL_MUT_t
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):pass
class AdjustTranslationForSrcImg(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_AdjustTranslationForSrcImg
    def transform(self, t, head_bbox, head_crop):
        if t is not None:
            h,w,_ = head_crop.shape
            t[0] *= w
            t[1] *= h
            left, top = head_bbox[0], head_bbox[1]
            t[0] += left
            t[1] += top
        return {
            self.PIPE_DEFAULT.KEY_OUT_t: t,
        }

class Pipe_AdjustMeshForSrcImg(tb.Pipe):
    KEY_CALL_MUT_mesh = ms_face.MsFaceReconstruction.PIPE_DEFAULT.KEY_OUT_mesh
    KEY_CALL_IMMUT_s = ms_face.MsFaceReconstruction.PIPE_DEFAULT.KEY_OUT_s
    KEY_CALL_IMMUT_t = ms_face.MsFaceReconstruction.PIPE_DEFAULT.KEY_OUT_t
    KEY_OUT_mesh = KEY_CALL_MUT_mesh
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):pass
class AdjustMeshForSrcImg(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_AdjustMeshForSrcImg
    def transform(self, mesh, t, s):
        if t is not None:
            mesh *= 1/s
            t = np.append(t, [0])
            assert t.shape[-1]==mesh.shape[-1]
            mesh += t
        return {
            self.PIPE_DEFAULT.KEY_OUT_mesh: mesh,
        }


class Pipe_GetVisCanvas(tb.Pipe):
    KEY_CALL_IMMUT_img_frame_src = KEY_img_frame_src
    KEY_OUT_img_vis = 'img_vis'
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):pass
class GetVisCanvas(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_GetVisCanvas
    KEY_INIT_scale = 'scale'
    def _init_(self, **config):
        self.scale = config[self.KEY_INIT_scale]
        assert self.scale > 0
    def transform(self, **data):
        img = deepcopy(data[self.PIPE_DEFAULT.KEY_CALL_IMMUT_img_frame_src])
        h, w, _ = img.shape
        w *= self.scale
        h *= self.scale
        img = cv2.resize(img, (w, h))
        return {
            self.PIPE_DEFAULT.KEY_OUT_img_vis: img,
        }


class Pipe_VisualizeMesh(tb.Pipe):
    KEY_CALL_IMMUT_mesh = 'mesh'
    KEY_CALL_MUT_img_vis = Pipe_GetVisCanvas.KEY_OUT_img_vis
    KEY_OUT_img_vis = KEY_CALL_MUT_img_vis
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):pass
class VisualizeMesh(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_VisualizeMesh
    KEY_INIT_mesh_scale = 'mesh_scale'
    def _init_(self, **config):
        self.mesh_scale = config[self.KEY_INIT_mesh_scale]
        assert self.mesh_scale > 0
    def transform(self, **data):
        img = data[self.PIPE_DEFAULT.KEY_CALL_MUT_img_vis]
        mesh = data[self.PIPE_DEFAULT.KEY_CALL_IMMUT_mesh]
        if mesh is not None:
            mesh *= self.mesh_scale
            img = self.draw_vertices(mesh, img, extra_scale=1)
        return {
            self.PIPE_DEFAULT.KEY_OUT_img_vis: img,
        }

    @staticmethod
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


class Pipe_CreateVisDir(tb.Pipe):
    KEY_CALL_IMMUT_working_dir = mappings.GetWorkingDir.PIPE_DEFAULT.KEY_OUT_working_dir
    KEY_OUT_visualise_dir = 'visualise_dir'
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):pass
class CreateVisDir(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_CreateVisDir
    KEY_INIT_name = 'name'
    def _init_(self, **config):
        self.name = config[self.KEY_INIT_name]
        assert isinstance(self.name, str)
    def transform(self, working_dir):
        # TODO sync with Video2Frames
        vis_dir = os.path.join(*[working_dir, self.name+'_visualise_dir'])
        if os.path.isdir(vis_dir):
            for pdir, dirs, files in os.walk(vis_dir):
                assert len(dirs) == 0
                old_frames = [os.path.join(pdir, x) for x in os.listdir(vis_dir)]
                [os.remove(x) for x in old_frames]
            os.rmdir(vis_dir)
        assert not os.path.isdir(vis_dir), vis_dir
        pardir = os.path.dirname(vis_dir)
        assert os.path.isdir(pardir), pardir
        os.mkdir(vis_dir)
        return {
            self.PIPE_DEFAULT.KEY_OUT_visualise_dir: vis_dir,
        }


class Pipe_SaveVisCanvas(tb.Pipe):
    KEY_CALL_IMMUT_ImagePath = KEY_ImagePath
    KEY_CALL_IMMUT_img_vis = Pipe_GetVisCanvas.KEY_OUT_img_vis
    KEY_CALL_IMMUT_visualise_dir = CreateVisDir.PIPE_DEFAULT.KEY_OUT_visualise_dir
    def before_keys_assertions(self, **data):pass
    def after_keys_assertions(self, **data):pass
class SaveVisCanvas(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_SaveVisCanvas
    def _init_(self, **config):
        pass
    def transform(self, **data):
        visualise_dir = data[self.PIPE_DEFAULT.KEY_CALL_IMMUT_visualise_dir]
        ImagePath = data[self.PIPE_DEFAULT.KEY_CALL_IMMUT_ImagePath]
        img_vis = data[self.PIPE_DEFAULT.KEY_CALL_IMMUT_img_vis]
        save_file = os.path.join(visualise_dir, os.path.basename(ImagePath))
        assert not os.path.isfile(save_file), save_file
        # cv2.imwrite(save_file, img_vis)
        plt.imsave(save_file, img_vis)
        return {}


def tp_detect_frames(vis_scale=1, use_cache=False, cache_name=''):
    tp = [
        [tb.Cacher, {
            tb.Cacher.KEY_INIT_name: cache_name,
            tb.Cacher.KEY_INIT_cache_dir: '/tmp',
            tb.Cacher.KEY_INIT_use_cache: use_cache,
            tb.Cacher.KEY_INIT_sample_transform_cls_cnfg: [
                tb.TransformPipeline, {
                    tb.TransformPipeline.KEY_INIT_transforms: [
                        [openpose.DetectFrames, {}],
                        [tb.BatchTransformSingleThread, {
                            tb.BatchTransformSingleThread.KEY_INIT_pipes: [
                                openpose.Pipe_Rename_Frames_as_Batch,  # TODO
                                tb.PipeReproduceInBatch(
                                    keys_to_reproduce={CreateVisDir.PIPE_DEFAULT.KEY_OUT_visualise_dir, }),
                            ],
                            tb.BatchTransformSingleThread.KEY_INIT_sample_transform_cls_cnfg: [
                                tb.TransformPipeline, {
                                    tb.TransformPipeline.KEY_INIT_transforms: [
                                        [img.ReadImageCv2, {
                                            img.ReadImageCv2.KEY_INIT_pipes: [tb.PipeInnerRename(dict_ext_int_keys={
                                                KEY_ImagePath:
                                                    img.ReadImageCv2.PIPE_DEFAULT.KEY_CALL_IMMUT_file,
                                                KEY_img_frame_src:
                                                    img.ReadImageCv2.PIPE_DEFAULT.KEY_OUT_hwc,
                                            })],
                                        }],
                                        [GetVisCanvas, {
                                            GetVisCanvas.KEY_INIT_scale: vis_scale,
                                        }],

                                        [tb.BatchTransformInCycle, {
                                            tb.BatchTransformInCycle.KEY_INIT_global_batch_vars: {
                                                KEY_img_frame_src, GetVisCanvas.PIPE_DEFAULT.KEY_OUT_img_vis,},
                                            tb.BatchTransformInCycle.KEY_INIT_pipes: [
                                                tb.PipeInnerRename(
                                                    dict_ext_int_keys={KEY_DetectedObjects: tb.KEY_batch}),
                                            ],
                                            tb.BatchTransformInCycle.KEY_INIT_sample_transform_cls_cnfg: [
                                                tb.TransformPipeline, {
                                                    tb.TransformPipeline.KEY_INIT_transforms: [
                                                        [openpose.GetBboxFromOpenposeLms, {}],
                                                        [openpose.GetHeadCropFromHeadBbox, {
                                                            openpose.GetHeadCropFromHeadBbox.KEY_INIT_pipes: [
                                                                tb.PipeInnerRename(dict_ext_int_keys={
                                                                    KEY_img_frame_src:
                                                                        img.ReadImageCv2.PIPE_DEFAULT.KEY_OUT_hwc,
                                                                })],
                                                        }],
                                                        [fan2d.Get2dFanLms, {
                                                            fan2d.Get2dFanLms.KEY_INIT_pipes: [
                                                                tb.PipeInnerRename(dict_ext_int_keys={
                                                                    openpose.GetHeadCropFromHeadBbox.PIPE_DEFAULT.KEY_OUT_head_crop:
                                                                        fan2d.Get2dFanLms.PIPE_DEFAULT.KEY_CALL_IMMUT_hwc,
                                                                })],
                                                        }],
                                                        [ms_face.MsFaceReconstruction, {
                                                            ms_face.MsFaceReconstruction.KEY_INIT_pipes: [
                                                                tb.PipeInnerRename(dict_ext_int_keys={
                                                                    openpose.GetHeadCropFromHeadBbox.PIPE_DEFAULT.KEY_OUT_head_crop:
                                                                        fan2d.Get2dFanLms.PIPE_DEFAULT.KEY_CALL_IMMUT_hwc,
                                                                })
                                                            ],
                                                        }],
                                                        [AdjustTranslationForSrcImg, {}],
                                                        [AdjustMeshForSrcImg, {}],
                                                        [VisualizeMesh, {
                                                            VisualizeMesh.KEY_INIT_mesh_scale: vis_scale,
                                                            VisualizeMesh.KEY_INIT_pipes: [
                                                                tb.PipeInnerRename(dict_ext_int_keys={
                                                                    KEY_img_frame_src:
                                                                        img.ReadImageCv2.PIPE_DEFAULT.KEY_OUT_hwc,
                                                                })
                                                            ],
                                                        }],
                                                    ]
                                                }
                                            ],
                                        }],
                                        [SaveVisCanvas, {}],
                                        # [ClearData, {}],
                                    ]
                                }
                            ],
                        }],
                    ],
                }
            ],
        }],
    ]
    return tp


def demo_helvas():
    scale = 3
    tp = tb.TransformPipeline(**{
        tb.TransformPipeline.KEY_INIT_transforms: [
            [mappings.GetWorkingDir, {}],
            [video_example.GetDemoHelvas0Video, {}],
            [video.Video2Frames, {
                video.Video2Frames.KEY_INIT_sb_fps: 5,
            }],
            [CreateVisDir, {
                    CreateVisDir.KEY_INIT_name: 'demo_helvas',
            }],
        ]+tp_detect_frames(vis_scale=scale)
    })

    out = tp(**{})

    print("finished")
    return 0

def fit_angles_from_op(Bones, im):
    op_points = openpose.GetBboxFromOpenposeLms.get_points_from_bones(Bones)
    op_points = op_points[[
        0, 15, 16,
        17,
        # 18,
    ]].astype(float)
    m3d_points = np.array([
        [0, 0, 0],
        [-185, 170, -135],
        [185, 170, -135],
        [-185 * 2, 170, -3 * 135],
        # [185*2,170,-3*135],
    ], dtype=float)
    size = np.array([960, 1280])
    focal_length = size[1]
    center = size / 2
    camera_matrix = np.array([
        [focal_length, 0, center[1]],
        [0, focal_length, center[0]],
        [0, 0, 1],
    ], dtype=float)
    dist_coeffs = np.zeros((4, 1), dtype=float)  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(m3d_points, op_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_DLS)  # , useExtrinsicGuess=True)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)

    h, w, _ = im.shape
    w = int(round(w*1/3))
    h = int(round(h*1/3))
    im = cv2.resize(im, (w, h))
    for p in op_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
    p1 = (int(op_points[0][0]), int(op_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(im, p1, p2, (255, 0, 0), 2)
    plt.imshow(im); plt.show()
    print()


def tst():
    scale = 3

    tp = tb.Cacher(**{
        tb.Cacher.KEY_INIT_cache_dir: '/tmp',
        tb.Cacher.KEY_INIT_name: 'mscdiploma_tst',
        tb.Cacher.KEY_INIT_use_cache: True,
        tb.Cacher.KEY_INIT_sample_transform_cls_cnfg: [
            tb.TransformPipeline, {
            tb.TransformPipeline.KEY_INIT_transforms: [
                [mappings.GetWorkingDir, {}],
                [video_example.GetTstFrames, {}],
                [CreateVisDir, {
                    CreateVisDir.KEY_INIT_name: 'tst'
                }],
            ]+tp_detect_frames(vis_scale=scale)}
        ]
    })
    out = tp(**{})


    print()
    errors = []
    i = 0
    for a,b in zip(out['batch_frames'], out['openpose_detect_data']['FrameSequences'][0]['Frames']):
        head = a[video_example.GetTstFrames.PIPE_DEFAULT.lvl_KEY_OUT_batch_frames__head_angles_gt]
        cam = a[video_example.GetTstFrames.PIPE_DEFAULT.lvl_KEY_OUT_batch_frames__cam_angles_gt]
        a = cam-head

        im = b['img_vis']
        b = b['DetectedObjects']
        fit_angles_from_op(b[0]['Bones'], im)

        assert len(b)>0
        bangles = []
        for do in b:
            if do['angles'] is not None:
                bangles.append(do['angles'])
        if not len(bangles)==1:
            assert i==14
            b = bangles[1]
        else:
            b = bangles[0]
        b = np.array([np.rad2deg(x) for x in b])

        for x in [a,b]:
            assert len(x)==3

        if i>=10:
            print()
            errors.append(abs(-1*a[0]-(90+b[0])))
        else:
            errors.append(abs(a[1]-b[1]))
        i+=1
    print()
