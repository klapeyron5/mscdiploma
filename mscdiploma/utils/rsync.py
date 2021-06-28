import os
import tqdm


class RsyncException(Exception):
    """"""


class RsyncUtils:

    @staticmethod
    def make_rsync_file(data, rsync_file='./tmp_rsync_file'):
        """

        Args:
            data: frames_dir or videos
            rsync_file:

        Returns:

        """
        s = ''
        for x in data:
            s += x + '\n'
        with open(rsync_file, 'w') as f:
            f.write(s)
        return rsync_file

    @classmethod
    def download(cls, data, start='/', dst=None, usr=None, ip=None, aftercheck=False):
        rsync_file = cls.make_rsync_file(data)

        query = "rsync " \
                "-rtl " \
                "--files-from={rsync_file} " \
                "{usr}@{ip}:{start} " \
                "{dst}".format(
            rsync_file=rsync_file,
            usr=usr,
            ip=ip,
            start=start,
            dst=dst,
        )
        result = os.system(query)
        if result != 0:
            os.remove(rsync_file)
            raise RsyncException
        os.remove(rsync_file)

        if aftercheck:
            print('Checking downloaded files and dirs')
            for x in tqdm(data, total=len(data)):
                x = os.path.join(dst, x)
                assert os.path.isdir(x) or os.path.isfile(x), x
