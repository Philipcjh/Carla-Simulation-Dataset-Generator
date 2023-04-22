import sys
import yaml
import carla
import math
import numpy as np


# 从yaml文件中读取config
def yaml_to_config(file):
    try:
        with open(file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config
    except:
        return None

# 将config中的位姿转换为carla中的位姿
def config_transform_to_carla_transform(config_transform):
    carla_transform = carla.Transform(carla.Location(config_transform["location"][0],
                                                     config_transform["location"][1],
                                                     config_transform["location"][2]),
                                      carla.Rotation(config_transform["rotation"][0],
                                                     config_transform["rotation"][1],
                                                     config_transform["rotation"][2]))
    return carla_transform


# RGB相机内参矩阵
def camera_intrinsic(width, height):
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    f = width / (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    return k
