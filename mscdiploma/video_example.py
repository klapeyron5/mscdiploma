from ..transform import base as tb
from ..transform.file_system import KEY_file
from .video import KEY_video_file, KEY_frames_dir
import os
import numpy as np


class Pipe_GetTestVideo(tb.Pipe):
    KEY_CALL_IMMUT_working_dir = 'working_dir'
    KEY_OUT_video_file = KEY_video_file
    def before_keys_assertions(self, **data):
        pass
    def after_keys_assertions(self, **data):
        self.static_assertions__test_video(data[self.KEY_OUT_video_file])
    @classmethod
    def static_assertions__test_video(cls, test_video):
        assert os.path.isfile(test_video), test_video


class GetDemoHelvas0Video(tb.TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_GetTestVideo
    def _init_(self, **config):
        self.test_video_name = 'SVO-03-01.mov'
    def transform(self, working_dir):
        test_video = os.path.join(working_dir, self.test_video_name)
        return {
            self.PIPE_DEFAULT.KEY_OUT_video_file: test_video,
        }


class GetTstFrames(tb.TransformNoDataIndependentRandomness):
    class Pipe(tb.PipeDefault):
        KEY_CALL_IMMUT_working_dir = 'working_dir'
        KEY_OUT_batch_frames = tb.KEY_batch + '_frames'
        lvl_KEY_OUT_batch_frames__file = KEY_file
        lvl_KEY_OUT_batch_frames__head_angles_gt = 'head_angles_gt'
        lvl_KEY_OUT_batch_frames__cam_angles_gt = 'cam_angles_gt'
        KEY_OUT_frames_dir = KEY_frames_dir
    PIPE_DEFAULT = Pipe
    def transform(self, working_dir):
        tst_dir = os.path.join(working_dir, 'tst')
        assert os.path.isdir(tst_dir)
        for pdir, dirs, files in os.walk(tst_dir):
            assert len(dirs)==0
            assert all([x.endswith('.jpg') for x in files])
            frames = [os.path.join(pdir, x) for x in os.listdir(tst_dir)]
            frames.sort()
        def get_gt(file):
            name = os.path.splitext(os.path.basename(file))[0]
            if 'cam0' in name:
                cam_yaw = 100
            elif 'cam1' in name:
                cam_yaw = 80
            else: raise Exception
            cam_pitch = 0
            cam_roll = 0

            if 'yaw' in name:
                yaw = name.split('yaw')[-1]
                yaw = int(yaw)
                assert 0<=yaw<=180
                pitch = 0
                roll = 0
            elif 'pitch' in name:
                pitch = name.split('pitch')[-1]
                pitch = int(pitch)
                yaw = 113
                roll = 0
            else: raise Exception
            return np.array([pitch,yaw,roll]), np.array([cam_pitch, cam_yaw, cam_roll])
        batch_frames = []
        for x in frames:
            h,c = get_gt(x)
            batch_frames.append({
                self.PIPE_DEFAULT.lvl_KEY_OUT_batch_frames__file: x,
                self.PIPE_DEFAULT.lvl_KEY_OUT_batch_frames__head_angles_gt: h,
                self.PIPE_DEFAULT.lvl_KEY_OUT_batch_frames__cam_angles_gt: c,
            })
        return {
            self.PIPE_DEFAULT.KEY_OUT_batch_frames: batch_frames,
            self.PIPE_DEFAULT.KEY_OUT_frames_dir: tst_dir,
        }
