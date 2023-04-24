import yaml
import carla
import math
import numpy as np


def yaml_to_config(file):
    """
        从yaml文件中读取config

        参数：
            file：文件路径

        返回：
            config：预设配置
    """
    try:
        with open(file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config
    except:
        return None


def config_transform_to_carla_transform(config_transform):
    """
        将config中的位姿转换为carla中的位姿

        参数：
            config_transform：config中的位姿

        返回：
            carla_transform：carla中的位姿
    """
    carla_transform = carla.Transform(carla.Location(config_transform["location"][0],
                                                     config_transform["location"][1],
                                                     config_transform["location"][2]),
                                      carla.Rotation(config_transform["rotation"][0],
                                                     config_transform["rotation"][1],
                                                     config_transform["rotation"][2]))
    return carla_transform


def set_camera_intrinsic(width, height):
    """
        设置相机内参矩阵

        参数：
            width：图像宽度(pixel)
            height：图像高度(pixel)

        返回：
            k：相机内参矩阵
    """
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    f = width / (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    return k


def object_filter_by_distance(data, distance):
    """
        根据预设距离对场景中的物体进行简单过滤

        参数：
            data：CARLA传感器相关数据（原始数据，内参，外参等）
            distance：预设的距离阈值

        返回：
            data：处理后的CARLA传感器相关数据
    """
    environment_objects = data["environment_objects"]
    actors = data["actors"]
    for agent, _ in data["agents_data"].items():
        data["environment_objects"] = [obj for obj in environment_objects if
                                       agent.get_location().distance(obj.transform.location) < distance]
        data["actors"] = [act for act in actors if
                          agent.get_location().distance(act.get_location()) < distance]
    return data


def raw_image_to_rgb_array(image):
    """
        将CARLA原始图像数据转换为RGB numpy数组

        参数：
            image：CARLA原始图像数据

        返回：
            array：RGB numpy数组
    """
    # 将CARLA原始数据转化为BGRA numpy数组
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))

    # 将BGRA numpy数组转化为RGB numpy数组
    # 向量只取BGR三个通道
    array = array[:, :, :3]
    # 倒序
    array = array[:, :, ::-1]
    return array


def depth_image_to_array(image):
    """
        将carla获取的raw depth_image转换成深度图
    """
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
    array = array[:, :, :3]  # Take only RGB
    array = array[:, :, ::-1]  # BGR
    array = array.astype(np.float32)  # 2ms
    gray_depth = ((array[:, :, 0] + array[:, :, 1] * 256.0 + array[:, :, 2] * 256.0 * 256.0) / (
            (256.0 * 256.0 * 256.0) - 1))  # 2.5ms
    gray_depth = 1000 * gray_depth
    return gray_depth
