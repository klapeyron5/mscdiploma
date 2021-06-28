from .transform import base as tb
from . import mappings
from .video import Pipe_Video2Frames
import os
import json
import numpy as np
import cv2


class Pipe_DetectFrames(tb.Pipe):
    KEY_CALL_IMMUT_batch_frames = Pipe_Video2Frames.KEY_OUT_batch_frames
    _KEY_CALL_IMMUT_batch_frames__file = Pipe_Video2Frames.lvl_KEY_OUT_batch_frames__file
    KEY_CALL_IMMUT_frames_dir = Pipe_Video2Frames.KEY_OUT_frames_dir
    KEY_CALL_IMMUT_working_dir = mappings.Pipe_GetWorkingDir.KEY_OUT_working_dir
    KEY_OUT_openpose_detect_json = 'openpose_detect_json'
    KEY_OUT_openpose_detect_data = 'openpose_detect_data'
    def before_keys_assertions(self, **data):
        pass
    def after_keys_assertions(self, **data):
        pass
class DetectFrames(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_DetectFrames
    def _init_(self, **config):
        pass
    def transform(self, frames_dir, working_dir, batch_frames):
        detect_json = os.path.join(working_dir, os.path.basename(frames_dir)+'-detect.json')
        if os.path.isfile(detect_json):
            data = json.load(open(detect_json))
        else:
            raise NotImplementedError(detect_json)

        a = data['FrameSequences'][0]['Frames']
        assert len(a) == len(batch_frames)
        for i in range(len(a)):
            f0 = os.path.basename(batch_frames[i]['file'])
            f1 = os.path.basename(a[i]['ImagePath'])
            try:
                assert f0 == f1, "{}; {}".format(batch_frames[i]['file'], a[i]['ImagePath'])
            except Exception:
                # TODO
                n0 = f0.split('_')[0]
                assert 0<=int(n0)<=9999
                n1 = os.path.splitext(f1)[0]
                assert int(n0)==int(n1)
            a[i]['ImagePath'] = batch_frames[i]['file']
        return {
            self.PIPE_DEFAULT.KEY_OUT_openpose_detect_json: detect_json,
            self.PIPE_DEFAULT.KEY_OUT_openpose_detect_data: data,
        }

class Pipe_Rename_Frames_as_Batch(tb._PipeLight):
    rename_KEY_CALL_MUT_openpose_detect_data = Pipe_DetectFrames.KEY_OUT_openpose_detect_data
    KEY_CALL_MUT_batch = tb.KEY_batch
    KEY_OUT_batch = tb.KEY_batch
    def before_keys_assertions(self, **data): pass
    def after_keys_assertions(self, **data): pass

    def __call__(self, **data):
        batch = data[self.rename_KEY_CALL_MUT_openpose_detect_data]['FrameSequences'][0]['Frames']
        del data[self.rename_KEY_CALL_MUT_openpose_detect_data]['FrameSequences'][0]['Frames']
        assert self.KEY_CALL_MUT_batch not in data
        data[self.KEY_CALL_MUT_batch] = batch
        data = tb._PipeLight.__call__(self, **data)
        data[self.rename_KEY_CALL_MUT_openpose_detect_data]['FrameSequences'][0]['Frames'] = data[self.KEY_OUT_batch]
        del data[self.KEY_OUT_batch]
        return data

# class Pipe_Rename_DetectedObjects_as_Batch(tb.Pipe):
#     rename_KEY_CALL_MUT_DetectedObjects = 'DetectedObjects'
#     reproduce_KEY_CALL_img_frame_src = 'img_frame_src'
#     KEY_CALL_MUT_batch = tb.KEY_batch
#     KEY_OUT_batch = tb.KEY_batch
#     def before_keys_assertions(self, **data): pass
#     def after_keys_assertions(self, **data): pass
#
#     def __call__(self, **data):
#         batch = data[self.rename_KEY_CALL_MUT_DetectedObjects]
#         del data[self.rename_KEY_CALL_MUT_DetectedObjects]
#         for s in batch:
#             s[self.reproduce_KEY_CALL_img_frame_src] = data[self.reproduce_KEY_CALL_img_frame_src]
#         assert self.KEY_CALL_MUT_batch not in data
#         data[self.KEY_CALL_MUT_batch] = batch
#         data = tb.Pipe.__call__(self, **data)
#         for s in data[self.KEY_OUT_batch]:
#             del s[self.reproduce_KEY_CALL_img_frame_src]
#         data[self.rename_KEY_CALL_MUT_DetectedObjects] = data[self.KEY_OUT_batch]
#         del data[self.KEY_OUT_batch]
#         return data


class Pipe_GetBboxFromOpenposeLms(tb.Pipe):
    KEY_CALL_IMMUT_Bones = 'Bones'
    KEY_OUT_head_bbox = 'head_bbox'
    def before_keys_assertions(self, **data): pass
    def after_keys_assertions(self, **data): pass
    def __call__(self, **data):
        data.setdefault(self.KEY_CALL_IMMUT_Bones, None)
        return tb.Pipe.__call__(self, **data)
class GetBboxFromOpenposeLms(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_GetBboxFromOpenposeLms
    def _init_(self, **config):
        self.ref_keyspoints = dict(
            nose = 0,
            leye = 15,
            reye = 16,
            lear = 17,
            rear = 18,
            chest = 1,
            lshoulder = 2,
            rshoulder = 5,
        )
    def transform(self, **data):
        Bones = data[self.PIPE_DEFAULT.KEY_CALL_IMMUT_Bones]
        if Bones is not None:
            points = self.get_points_from_bones(Bones)
            points = points[list(self.ref_keyspoints.values())]
            points = np.array(list(filter(lambda p: (p[0] >= 0) and (p[1] >= 0), points)))
            if len(points) == 0:
                head_bbox = None
            else:
                assert np.issubdtype(points.dtype, np.number)
                stds = 3 * np.array([np.std(points[:, 0]), 1.5 * np.std(points[:, 1])])
                if (any([x==0 for x in stds])) or (not 0.1 < stds[0] / stds[1] < 10.0):
                    head_bbox = None
                else:
                    center = np.array(np.mean(points, axis=0))
                    assert center.shape == stds.shape
                    head_bbox = [center[0] - stds[0], center[1] - stds[1], center[0] + stds[0], center[1] + stds[1]]
                    head_bbox = np.round(head_bbox).astype(int)
        else:
            head_bbox = None
        return {
            self.PIPE_DEFAULT.KEY_OUT_head_bbox: head_bbox,
        }
    @classmethod
    def get_points_from_bones(cls, Bones):
        points = []
        for x, y in zip(Bones[::2], Bones[1::2]):
            if x == -1 and y == -1:
                pass
            else:
                assert x >= 0 and y >= 0, "{}, {}".format(x, y)
            points.append([x, y])
        points = np.array(points)
        return points


from .transform import hwc
class Pipe_GetHeadCropFromHeadBbox(tb.Pipe):
    KEY_CALL_IMMUT_head_bbox = Pipe_GetBboxFromOpenposeLms.KEY_OUT_head_bbox
    KEY_CALL_IMMUT_hwc = hwc.KEY_hwc
    KEY_OUT_head_crop = 'head_crop'
    def before_keys_assertions(self, **data): pass
    def after_keys_assertions(self, **data): pass
class GetHeadCropFromHeadBbox(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_GetHeadCropFromHeadBbox
    def transform(self, **data):
        bbox = data[self.PIPE_DEFAULT.KEY_CALL_IMMUT_head_bbox]
        img = data[self.PIPE_DEFAULT.KEY_CALL_IMMUT_hwc]
        if bbox is not None:
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        else:
            crop = None
        return {
            self.PIPE_DEFAULT.KEY_OUT_head_crop: crop,
        }
