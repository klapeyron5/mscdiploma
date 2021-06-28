from nirvana.transform.base import TransformNoDataIndependentRandomness, Pipe, KEY_batch
from . import mappings
import os


KEY_video_file = 'video_file'
KEY_file = 'file'
KEY_frames_dir = 'frames_dir'


class Pipe_Video2Frames(Pipe):
    KEY_CALL_IMMUT_working_dir = mappings.Pipe_GetWorkingDir.KEY_OUT_working_dir
    KEY_CALL_IMMUT_video_file = KEY_video_file
    KEY_OUT_batch_frames = KEY_batch + '_frames'
    lvl_KEY_OUT_batch_frames__file = KEY_file
    KEY_OUT_frames_dir = KEY_frames_dir
    def before_keys_assertions(self, **data):
        pass
    def after_keys_assertions(self, **data):
        pass
class Video2Frames(TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_Video2Frames
    KEY_INIT_sb_fps = 'sb_fps'
    def _init_(self, **config):
        self.sb_fps = config[self.KEY_INIT_sb_fps]
        assert isinstance(self.sb_fps, int) and self.sb_fps>0
    def transform(self, working_dir, video_file):
        assert os.path.isfile(video_file), video_file
        sb_dname = os.path.splitext(os.path.basename(video_file))[0]#+'-sb'
        sb_dir = os.path.join(working_dir, sb_dname)  # TODO
        if os.path.isdir(sb_dir):
            for pdir, dirs, files in os.walk(sb_dir):
                assert len(dirs) == 0
                old_frames = [os.path.join(pdir, x) for x in os.listdir(sb_dir)]
                [os.remove(x) for x in old_frames]
            os.rmdir(sb_dir)
        assert not os.path.isdir(sb_dir), sb_dir
        pardir = os.path.dirname(sb_dir)
        assert os.path.isdir(pardir), pardir
        os.mkdir(sb_dir)
        out = self.ffmpeg_sb_from_video(video_file=video_file, sb_dir=sb_dir, sb_fps=self.sb_fps)
        assert out==0, "out: {}; video_file: {}; sb_dir:{}".format(out, video_file, sb_dir)
        for pdir, dirs, files in os.walk(sb_dir):
            assert len(dirs)==0
            frames = [os.path.join(pdir, x) for x in os.listdir(sb_dir)]
            frames.sort()
        batch_frames = [{self.PIPE_DEFAULT.lvl_KEY_OUT_batch_frames__file: x} for x in frames]
        return {
            self.PIPE_DEFAULT.KEY_OUT_batch_frames: batch_frames,
            self.PIPE_DEFAULT.KEY_OUT_frames_dir: sb_dir,
        }
    @classmethod
    def ffmpeg_sb_from_video(cls, video_file, sb_dir, sb_fps, sb_ext='.jpg', qscale=2):
        """
        Write storyboard (frames of video) from video by ffmpeg with request:
        ffmpeg -hide_banner -loglevel panic -i movie_path -r fps_out -qscale:v 2 files_dir/%04dextension
        :param video_file: path of video to storyboard
        :param sb_dir: folder to store frames of storyboard
        :param sb_fps: final storyboard fps
        :param sb_ext: extension of final frames of storyboard
        :param qscale: quality reduce (1-best, 31-worse)
        :return:
        """
        assert os.path.isfile(video_file)
        # assert rel_video_file.endswith(video_extensions)  # TODO
        assert os.path.isdir(sb_dir)
        assert sb_fps > 0
        assert sb_ext in {'.png', '.jpg'}
        assert isinstance(qscale, int)
        assert 1 <= qscale <= 31

        request = 'ffmpeg -hide_banner -loglevel panic ' \
                  '-i {video_file} -r {fps} -q:v {qscale} {sb_path}/%04d{ext}'\
            .format(video_file=video_file, fps=sb_fps, qscale=qscale, sb_path=sb_dir, ext=sb_ext)
        out = os.system(request)
        return out


if __name__ == '__main__':
    tp = Video2Frames(**{
        Video2Frames.KEY_INIT_sb_fps: 5,
    })
    # out = tp(**{})