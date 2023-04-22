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
def set_camera_intrinsic(width, height):
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    f = width / (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    return k


# 根据预设距离对场景中的物体进行过滤
def filter_by_distance(data_dict, dis):
    environment_objects = data_dict["environment_objects"]
    actors = data_dict["actors"]
    for agent, _ in data_dict["agents_data"].items():
        data_dict["environment_objects"] = [obj for obj in environment_objects if
                                            get_distance(obj.transform.location, agent.get_location())
                                            < dis]
        data_dict["actors"] = [act for act in actors if
                               get_distance(act.get_location(), agent.get_location()) < dis]


# 计算carla中两个位置之间的欧式距离（xy平面上）
def get_distance(location1, location2):
    return math.sqrt(pow(location1.x - location2.x, 2) + pow(location1.y - location2.y, 2))
