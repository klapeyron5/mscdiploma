from nirvana.transform.base import TransformNoDataIndependentRandomness, Pipe
import os


class Pipe_GetWorkingDir(Pipe):
    KEY_OUT_working_dir = 'working_dir'
    def before_keys_assertions(self, **data):
        pass
    def after_keys_assertions(self, **data):
        assert len(data.keys())==1
        self.static_assertions__working_dir(data[self.KEY_OUT_working_dir])
    @classmethod
    def static_assertions__working_dir(cls, working_dir):
        assert os.path.isdir(working_dir), working_dir


class GetWorkingDir(TransformNoDataIndependentRandomness):
    PIPE_DEFAULT = Pipe_GetWorkingDir

    def _init_(self, **config):
        self.pardir = os.path.dirname(__file__)
        assert os.path.isdir(self.pardir), self.pardir

        self.working_dname = 'working_dir'

    def transform(self):
        working_dir = os.path.join(self.pardir, self.working_dname)

        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)
        return {
            self.PIPE_DEFAULT.KEY_OUT_working_dir: working_dir,
        }


if __name__ == '__main__':
    tp = GetWorkingDir(**{})
    out = tp(**{})
    print(out)
