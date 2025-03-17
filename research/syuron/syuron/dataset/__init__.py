from .dataset import *
from .mnist import *

tf.config.set_visible_devices([], device_type='GPU')  # jaxで使用するGPUをtfが専有するのを防ぐ
