import numpy as np
import carla
import math
from numpy.linalg import inv
from utils import raw_image_to_rgb_array, yaml_to_config, depth_image_to_array
from data_descripter import KittiDescriptor
from visual_utils import point_in_canvas, draw_2d_bounding_box, draw_3d_bounding_box

config = yaml_to_config("configs.yaml")
MAX_RENDER_DEPTH_IN_METERS = config["FILTER_CONFIG"]["MAX_RENDER_DEPTH_IN_METERS"]
MIN_VISIBLE_VERTICES_FOR_RENDER = config["FILTER_CONFIG"]["MIN_VISIBLE_VERTICES_FOR_RENDER"]
MAX_OUT_VERTICES_FOR_RENDER = config["FILTER_CONFIG"]["MAX_OUT_VERTICES_FOR_RENDER"]
WINDOW_WIDTH = config["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_x"]
WINDOW_HEIGHT = config["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_y"]


def spawn_dataset(data):
    """
        处理传感器原始数据，生成KITTI数据集(RGB图像，激光雷达点云，KITTI标签等)

        参数：
            data：CARLA传感器相关数据（原始数据，内参，外参等）

        返回：
            data：处理后的数据（RGB图像，激光雷达点云，KITTI标签等）
    """
    # 筛选环境中的车辆
    environment_objects = data["environment_objects"]
    environment_objects = [x for x in environment_objects if x.type == "vehicle"]

    # 筛选actors中的车辆与行人
    actors = data["actors"]
    actors = [x for x in actors if x.type_id.find("vehicle") != -1 or x.type_id.find("walker") != -1]

    agents_data = data["agents_data"]
    for agent, dataDict in agents_data.items():
        intrinsic = dataDict["intrinsic"]
        extrinsic = dataDict["extrinsic"]
        sensors_data = dataDict["sensor_data"]
        sensor_transform = dataDict["transform"]

        image_labels_kitti = []
        pc_labels_kitti = []

        rgb_image = raw_image_to_rgb_array(sensors_data[0])
        image = rgb_image.copy()
        # image_lidar = rgb_image.copy()
        depth_data = depth_image_to_array(sensors_data[1])
        # semantic_lidar = np.frombuffer(sensors_data[3].raw_data, dtype=np.dtype('f4,f4, f4, f4, i4, i4'))

        data["agents_data"][agent]["visible_environment_objects"] = []
        for obj in environment_objects:
            image_label_kitti = is_visible_in_camera(agent, obj, image, depth_data, intrinsic, extrinsic)
            if image_label_kitti is not None:
                data["agents_data"][agent]["visible_environment_objects"].append(obj)
                image_labels_kitti.append(image_label_kitti)

        data["agents_data"][agent]["visible_actors"] = []
        for act in actors:
            image_label_kitti = is_visible_in_camera(agent, act, image, depth_data, intrinsic, extrinsic)
            # is_visible_by_lidar_bbox(agent, act, image_lidar, semantic_lidar, intrinsic, extrinsic)
            if image_label_kitti is not None:
                data["agents_data"][agent]["visible_actors"].append(act)
                image_labels_kitti.append(image_label_kitti)

            # lidar_datapoint = is_visible_in_lidar(agent, act, semantic_lidar, extrinsic, sensor_transform)
            # if lidar_datapoint is not None:
            #     print(lidar_datapoint)
            #     pointcloud_labels_kitti.append(lidar_datapoint)

        data["agents_data"][agent]["rgb_image"] = rgb_image
        data["agents_data"][agent]["bbox_img"] = image
        data["agents_data"][agent]["image_labels_kitti"] = image_labels_kitti
        data["agents_data"][agent]["pc_labels_kitti"] = pc_labels_kitti
    return data


def is_visible_in_camera(agent, obj, rgb_image, depth_data, intrinsic, extrinsic):
    """
        筛选出在摄像头中可见的目标物并生成标签

        参数：
            agent：CARLA传感器相关数据（原始数据，内参，外参等）

        返回：
            dataset：处理后的数据（RGB图像，激光雷达点云，KITTI标签等）
    """
    obj_transform = obj.transform if isinstance(obj, carla.EnvironmentObject) else obj.get_transform()
    obj_bbox = obj.bounding_box
    if isinstance(obj, carla.EnvironmentObject):
        vertices_in_image = get_2d_vertices_in_image(intrinsic, extrinsic, obj_bbox, obj_transform, 0)
    else:
        vertices_in_image = get_2d_vertices_in_image(intrinsic, extrinsic, obj_bbox, obj_transform, 1)

    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(vertices_in_image, depth_data)
    if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and \
            num_vertices_outside_camera < MAX_OUT_VERTICES_FOR_RENDER:
        obj_tp = obj_type(obj)
        rotation_y = get_relative_rotation_yaw(agent.get_transform().rotation, obj_transform.rotation) % math.pi
        midpoint = midpoint_from_agent_location(obj_transform.location, extrinsic)
        bbox_2d = calc_projected_2d_bbox(vertices_in_image)
        ext = obj.bounding_box.extent
        truncated = num_vertices_outside_camera / 8
        if num_visible_vertices >= 6:
            occluded = 0
        elif num_visible_vertices >= 4:
            occluded = 1
        else:
            occluded = 2

        # draw bounding box
        # draw_3d_bounding_box(rgb_image, vertices_pos2d)
        bbox_2d = draw_2d_bounding_box(rgb_image, bbox_2d)

        kitti_label = KittiDescriptor()
        kitti_label.set_truncated(truncated)
        kitti_label.set_occlusion(occluded)
        kitti_label.set_bbox(bbox_2d)
        kitti_label.set_3d_object_dimensions(ext)
        kitti_label.set_type(obj_tp)
        kitti_label.set_3d_object_location(midpoint)
        kitti_label.set_rotation_y(rotation_y)

        return kitti_label
    return None


def get_2d_vertices_in_image(intrinsic_mat, extrinsic_mat, obj_bbox, obj_transform, obj_tp):
    """
        将物体3d bounding box的顶点坐标投影到像素坐标系上

        参数：
            agent：CARLA传感器相关数据（原始数据，内参，外参等）

        返回：
            vertices_in_image：8个顶点在像素坐标系下的坐标
    """
    bbox = extension_to_vertices(obj_bbox)
    # actors
    if obj_tp == 1:
        bbox_transform = carla.Transform(obj_bbox.location, obj_bbox.rotation)
        bbox = transform_points(bbox_transform, bbox)
    # 环境物体
    else:
        box_location = carla.Location(obj_bbox.location.x - obj_transform.location.x,
                                      obj_bbox.location.y - obj_transform.location.y,
                                      obj_bbox.location.z - obj_transform.location.z)
        box_rotation = obj_bbox.rotation
        bbox_transform = carla.Transform(box_location, box_rotation)
        bbox = transform_points(bbox_transform, bbox)
    # 获取bbox在世界坐标系下的点的坐标
    bbox = transform_points(obj_transform, bbox)
    # 将世界坐标系下的bbox八个顶点转换到二维图片中
    vertices_in_image = vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat)
    return vertices_in_image


def extension_to_vertices(obj_bbox):
    """
        从CARLA物体的bounding box得到车辆3D bounding box的八个顶点坐标（以车辆中心为原点的坐标系）

        参数：
            obj_bbox：CARLA物体的bounding box

        返回：
            vertices：车辆3D bounding box的八个顶点坐标(以车辆中心为原点的坐标系）[3 × 8 矩阵]
    """
    ext = obj_bbox.extent
    # 8 × 3
    vertices = np.array([
        [ext.x, ext.y, ext.z],  # Top left front
        [- ext.x, ext.y, ext.z],  # Top left back
        [ext.x, - ext.y, ext.z],  # Top right front
        [- ext.x, - ext.y, ext.z],  # Top right back
        [ext.x, ext.y, - ext.z],  # Bottom left front
        [- ext.x, ext.y, - ext.z],  # Bottom left back
        [ext.x, - ext.y, - ext.z],  # Bottom right front
        [- ext.x, - ext.y, - ext.z]  # Bottom right back
    ])
    # vertices = vertices.transpose()
    return vertices


def calculate_occlusion_stats(vertices_pos2d, depth_image):
    """ 作用：筛选bbox八个顶点中实际可见的点 """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    for y_2d, x_2d, vertex_depth in vertices_pos2d:
        # 点在可见范围中，并且没有超出图片范围
        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas((y_2d, x_2d)):
            is_occluded = point_is_occluded(
                (y_2d, x_2d), vertex_depth, depth_image)
            if not is_occluded:
                num_visible_vertices += 1
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def point_is_occluded(point, vertex_depth, depth_image):
    y, x = map(int, point)
    from itertools import product
    neighbours = product((1, -1), repeat=2)
    is_occluded = []
    for dy, dx in neighbours:
        if point_in_canvas((dy + y, dx + x)):
            # 判断点到图像的距离是否大于深对应深度图像的深度值
            if depth_image[y + dy, x + dx] < vertex_depth:
                is_occluded.append(True)
            else:
                is_occluded.append(False)
    # 当四个邻居点都大于深度图像值时，点被遮挡。返回true
    return all(is_occluded)


def midpoint_from_agent_location(location, extrinsic_mat):
    """ 将agent在世界坐标系中的中心点转换到相机坐标系下 """
    midpoint_vector = np.array([
        [location.x],  # [[X,
        [location.y],  # Y,
        [location.z],  # Z,
        [1.0]  # 1.0]]
    ])
    transformed_3d_midpoint = proj_to_camera(midpoint_vector, extrinsic_mat)
    return transformed_3d_midpoint


def proj_to_camera(pos_vector, extrinsic_mat):
    """ 作用：将点的world坐标转换到相机坐标系中 """
    # inv求逆矩阵
    transformed_3d_pos = np.dot(inv(extrinsic_mat), pos_vector)
    return transformed_3d_pos


def transform_points(transform, points):
    """ 作用：将三维点坐标转换到指定坐标系下 """
    # 转置
    points = points.transpose()
    # [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]  (4,n)
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # transform.get_matrix() 获取当前坐标系向相对坐标系的旋转矩阵
    points = np.mat(transform.get_matrix()) * points
    # 返回前三行
    return points[0:3].transpose()


def calc_projected_2d_bbox(vertices_pos2d):
    """ 根据八个顶点的图片坐标，计算二维bbox的左上和右下的坐标值 """
    legal_pos2d = list(filter(lambda x: x is not None, vertices_pos2d))
    y_coords, x_coords = [int(x[0][0]) for x in legal_pos2d], [
        int(x[1][0]) for x in legal_pos2d]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return [min_x, min_y, max_x, max_y]


def vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat):
    """将bbox在世界坐标系中的点投影到该相机获取二维图片的坐标和点的深度"""
    vertices_pos2d = []
    for vertex in bbox:
        # 获取点在world坐标系中的向量
        pos_vector = vertex_to_world_vector(vertex)
        # 将点的world坐标转换到相机坐标系中
        transformed_3d_pos = proj_to_camera(pos_vector, extrinsic_mat)
        # 将点的相机坐标转换为二维图片的坐标
        pos2d = proj_to_2d(transformed_3d_pos, intrinsic_mat)
        # 点实际的深度
        vertex_depth = pos2d[2]
        # 点在图片中的坐标
        x_2d, y_2d = pos2d[0], pos2d[1]
        vertices_pos2d.append((y_2d, x_2d, vertex_depth))
    return vertices_pos2d


def vertex_to_world_vector(vertex):
    """ 以carla世界向量（X，Y，Z，1）返回顶点的坐标 （4,1）"""
    return np.array([
        [vertex[0, 0]],  # [[X,
        [vertex[0, 1]],  # Y,
        [vertex[0, 2]],  # Z,
        [1.0]  # 1.0]]
    ])


def proj_to_camera(vector, extrinsic_mat):
    """ 作用：将点的world坐标转换到相机坐标系中 """
    # inv求逆矩阵
    transformed_3d_pos = np.dot(inv(extrinsic_mat), vector)
    return transformed_3d_pos


def proj_to_2d(camera_pos_vector, intrinsic_mat):
    """将相机坐标系下的点的3d坐标投影到图片上"""
    cords_x_y_z = camera_pos_vector[:3, :]
    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    pos2d = np.dot(intrinsic_mat, cords_y_minus_z_x)
    # normalize the 2D points
    if abs(pos2d[2]) < 1e-6:
        pos2d = np.array([0, 0, 0])
    else:
        pos2d = np.array([pos2d[0] / pos2d[2], pos2d[1] / pos2d[2], pos2d[2]])
    return pos2d


def get_relative_rotation_yaw(agent_rotation, obj_rotation):
    """
        得到agent和物体在yaw的相对角度

        参数：
            obj：CARLA物体

        返回：
            obj.type：CARLA物体种类
    """
    rot_agent = agent_rotation.yaw
    rot_car = obj_rotation.yaw
    return math.radians(rot_agent - rot_car)


def obj_type(obj):
    """
        得到CARLA物体种类，对行人和汽车的种类重命名

        参数：
            obj：CARLA物体

        返回：
            obj.type：CARLA物体种类
    """
    if isinstance(obj, carla.EnvironmentObject):
        return obj.type
    else:
        if obj.type_id.find('walker') is not -1:
            return 'Pedestrian'
        if obj.type_id.find('vehicle') is not -1:
            return 'Car'
        return None
