from nirvana.utils.static_class_utils import static_init_class_decorator, get_class_attrs, StaticConstantsList
import os
import socket


class Server:
    HOSTNAME = None
    PUBLIC_ADDRESS = None

    __STATIC_INITS_STARTSWITH = 'static_init_'

    @classmethod
    def static_init(cls):
        """
        Also runs all static methods starting from "static_init_"
        """
        cls.__run_static_inits(cls)
        assert isinstance(cls.HOSTNAME, str)
        assert isinstance(cls.PUBLIC_ADDRESS, str) or cls.PUBLIC_ADDRESS is None

    @staticmethod
    def __run_static_inits(cls):
        statc_inits = get_class_attrs(cls, cls.__STATIC_INITS_STARTSWITH)
        for static_init in statc_inits:
            assert static_init.__self__ is cls
            assert static_init.__code__.co_argcount == 1
            static_init()


@static_init_class_decorator
class ServerKlapeyronserver(Server):
    HOSTNAME = 'klapeyronserver'


@static_init_class_decorator
class ServerTrain3(Server):
    HOSTNAME = 'train3.oz-services.ru'


@static_init_class_decorator
class ServerData(Server):
    HOSTNAME = 'data.oz-services.ru'
    PUBLIC_ADDRESS = HOSTNAME


@static_init_class_decorator
class Servers(StaticConstantsList):
    __DATA_STARTS_WITH = 'SERVER_'

    SERVER_TRAIN3 = ServerTrain3
    SERVER_DATA = ServerData
    SERVER_KLAPEYRONSERVER = ServerKlapeyronserver

    hostnames = []

    @classmethod
    def static_init(cls):
        cls.constants_list = get_class_attrs(cls, cls.__DATA_STARTS_WITH)
        cls.hostnames = [x.HOSTNAME for x in cls.constants_list]


@static_init_class_decorator
class CurrentServer(Server):
    @classmethod
    def static_init_current_hostname(cls):  # TODO
        cls.HOSTNAME = cls.get_current_hostname()

    @classmethod
    def get_current_hostname(cls):
        hostname = socket.gethostname()
        assert hostname in Servers.hostnames
        return hostname


class ServersDir:
    MAPPING = {
        Servers.SERVER_TRAIN3.HOSTNAME: None,
        Servers.SERVER_KLAPEYRONSERVER.HOSTNAME: None,
        Servers.SERVER_DATA.HOSTNAME: None,
    }

    DIR = None

    @classmethod
    def static_init(cls):
        cls.set_dir()

    @classmethod
    def set_dir(cls, current_server=CurrentServer):
        cls.DIR = cls.MAPPING[current_server.HOSTNAME]
        assert os.path.isdir(cls.DIR)


@static_init_class_decorator
class PycharmProjectDir(ServersDir):
    MAPPING = {
        Servers.SERVER_TRAIN3.HOSTNAME: '/home/nikitak',
        Servers.SERVER_KLAPEYRONSERVER.HOSTNAME: '/mnt/ssd120G_kingston',
        Servers.SERVER_DATA.HOSTNAME: '/tmp/pycharm_nikita',
    }


@static_init_class_decorator
class LogsDir(ServersDir):
    MAPPING = {
        Servers.SERVER_TRAIN3.HOSTNAME: '/home/nikitak/logs',
        Servers.SERVER_KLAPEYRONSERVER.HOSTNAME: '/mnt/ssd120G_kingston/logs',
    }


@static_init_class_decorator
class CacheDir(ServersDir):
    MAPPING = {
        Servers.SERVER_TRAIN3.HOSTNAME: '/home/nikitak/cache',
        Servers.SERVER_KLAPEYRONSERVER.HOSTNAME: '/home/klapeyron/cache',
    }


DRIVE_HDD = 'hdd'
DRIVE_SSD = 'ssd'


MAPPING_LABELER = {
    Servers.SERVER_KLAPEYRONSERVER.HOSTNAME: {
        DRIVE_HDD: '/mnt/hdd8T_toshiba/datasets/labeler/static',
        DRIVE_SSD: '/mnt/ssd1T_samsung/labeler/static',
    },
    Servers.SERVER_TRAIN3: {
        DRIVE_HDD: '/mnt/hdd3T/labeler/static',
    },
    Servers.SERVER_DATA: {
        DRIVE_HDD: '/opt/labeler/static',
    },
}


PYCHARM_PROJECTS_DIR = PycharmProjectDir.DIR
LOGS_DIR = LogsDir.DIR
CACHE_DIR = CacheDir.DIR
LABELER_HDD_DIR = MAPPING_LABELER[CurrentServer.HOSTNAME][DRIVE_HDD]
assert os.path.isdir(LABELER_HDD_DIR)
LABELER_SSD_DIR = MAPPING_LABELER[CurrentServer.HOSTNAME].get(DRIVE_SSD, None)


EXAMPLES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../examples'))
assert os.path.isdir(EXAMPLES_DIR)
EXAMPLE_ORIG_FRAMES_DIR_1 = os.path.normpath(EXAMPLES_DIR+os.sep+'close_1554444152977.mp4')
assert os.path.isdir(EXAMPLE_ORIG_FRAMES_DIR_1)
EXAMPLE_ZEROS_JPG = os.path.normpath(EXAMPLES_DIR+os.sep+'zeros.jpg')
assert os.path.isfile(EXAMPLE_ZEROS_JPG)
