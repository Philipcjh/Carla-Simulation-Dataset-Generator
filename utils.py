import sys
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
                                       get_distance(obj.transform.location, agent.get_location()) < distance]
        data["actors"] = [act for act in actors if
                          get_distance(act.get_location(), agent.get_location()) < distance]
    return data


def get_distance(location1, location2):
    """
        计算carla中两个位置之间的欧式距离（xy平面上）

        参数：
            location1：CARLA中的位置1
            location2：CARLA中的位置2

        返回：
            distance：两个位置之间的欧式距离
    """
    return math.sqrt(pow(location1.x - location2.x, 2) + pow(location1.y - location2.y, 2))


def spawn_datasets(data):
    """
        处理传感器原始数据，生成KITTI数据集(RGB图像，激光雷达点云，KITTI标签等)

        参数：
            data：CARLA传感器相关数据（原始数据，内参，外参等）

        返回：
            datasets：处理后的数据（RGB图像，激光雷达点云，KITTI标签等）
    """
    environment_objects = data["environment_objects"]
    environment_objects = [x for x in environment_objects if x.type == "vehicle"]
    agents_data = data["agents_data"]
    actors = data["actors"]
    actors = [x for x in actors if x.type_id.find("vehicle") != -1 or x.type_id.find("walker") != -1]
    for agent, dataDict in agents_data.items():
        intrinsic = dataDict["intrinsic"]
        extrinsic = dataDict["extrinsic"]
        sensors_data = dataDict["sensor_data"]
        sensor_location = dataDict["location"]
        kitti_datapoints = []
        carla_datapoints = []
        lidar_datapoints = []
        rgb_image = raw_image_to_rgb_array(sensors_data[0])
        image = rgb_image.copy()
        image_lidar = rgb_image.copy()
        depth_data = sensors_data[1]
        semantic_lidar = np.frombuffer(sensors_data[3].raw_data, dtype=np.dtype('f4,f4, f4, f4, i4, i4'))

        data["agents_data"][agent]["visible_environment_objects"] = []
        for obj in environment_objects:
            kitti_datapoint, carla_datapoint = is_visible_by_bbox(agent, obj, image, depth_data, intrinsic, extrinsic)
            # is_visible_by_lidar_bbox(agent, obj, image_lidar, semantic_lidar, intrinsic, extrinsic)
            if kitti_datapoint is not None:
                data["agents_data"][agent]["visible_environment_objects"].append(obj)
                kitti_datapoints.append(kitti_datapoint)
                carla_datapoints.append(carla_datapoint)

        data["agents_data"][agent]["visible_actors"] = []

        for act in actors:
            kitti_datapoint, carla_datapoint = is_visible_by_bbox(agent, act, image, depth_data, intrinsic, extrinsic)
            # is_visible_by_lidar_bbox(agent, act, image_lidar, semantic_lidar, intrinsic, extrinsic)
            if kitti_datapoint is not None:
                data["agents_data"][agent]["visible_actors"].append(act)
                kitti_datapoints.append(kitti_datapoint)
                carla_datapoints.append(carla_datapoint)

            lidar_datapoint = is_visible_in_lidar(agent, act, semantic_lidar, extrinsic, sensor_location)
            if lidar_datapoint is not None:
                print(lidar_datapoint)
                lidar_datapoints.append(lidar_datapoint)

        data["agents_data"][agent]["rgb_image"] = image
        data["agents_data"][agent]["lidar_image"] = image_lidar
        data["agents_data"][agent]["kitti_datapoints"] = kitti_datapoints
        data["agents_data"][agent]["carla_datapoints"] = carla_datapoints
        data["agents_data"][agent]["lidar_datapoints"] = lidar_datapoints
    return datasets


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
