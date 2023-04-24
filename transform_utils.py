import numpy as np
from numpy.linalg import inv


def midpoint_from_world_to_camera(location, extrinsic_mat):
    """
        将agent在世界坐标系中的中心点转换到相机坐标系下

        参数：
            location：agent在世界坐标系中的中心点「carla.location」
            extrinsic_mat：相机的外参矩阵

        返回：
            midpoint_camera：相机坐标系下agent的中心点
    """
    midpoint_world = np.array([
        [location.x],  # [[X,
        [location.y],  # Y,
        [location.z],  # Z,
        [1.0]  # 1.0]]
    ])
    midpoint_camera = transform_from_world_to_camera(midpoint_world, extrinsic_mat)
    return midpoint_camera


def transform_points(transform, points):
    """
        将任意齐次坐标点转换到指定三维坐标系下

        参数：
            transform：CARLA格式的位姿「carla.transform」
            points： 需要进行变换的点 [4 × n 矩阵]

        返回：
            points_transformed：指定坐标系下点的三维齐次坐标[4 × n 矩阵]
                                [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]
    """
    # 转换成齐次坐标
    points_transformed = np.mat(transform.get_matrix()) * points
    return points_transformed


def get_2d_bbox_in_pixel(vertices_2d):
    """
        根据八个顶点的像素坐标，计算二维bounding box的左上和右下的坐标值

        参数：
            vertices_2d：八个顶点的像素坐标

        返回：
            [min_x, min_y, max_x, max_y]：二维bounding box的左上和右下的坐标值
    """
    legal_pos2d = list(filter(lambda x: x is not None, vertices_2d))
    y_vec, x_vec = [int(x[0][0]) for x in legal_pos2d], [
        int(x[1][0]) for x in legal_pos2d]
    min_x, max_x = min(x_vec), max(x_vec)
    min_y, max_y = min(y_vec), max(y_vec)
    return [min_x, min_y, max_x, max_y]


def transform_from_world_to_camera(points_world, extrinsic_mat):
    """
        将世界坐标转下的点转换至相机坐标系中

        参数：
            points_world：世界坐标系下的3d齐次坐标
            extrinsic_mat：相机外参矩阵

        返回：
            vector3d：相机坐标系下的点的3d齐次坐标
    """
    points_camera = np.dot(inv(extrinsic_mat), points_world)
    return points_camera


def transform_from_camera_to_pixel(points_camera, intrinsic_mat):
    """
        将相机坐标系下的点的3d坐标转换至像素坐标系

        参数：
            points_camera：相机坐标系下的点的3d坐标
            intrinsic_mat：相机内参矩阵

        返回：
            points_pixel：像素坐标与该点的深度
    """
    points_in_cords_x_y_z = points_camera[:3, :]
    points_in_cords_y_minus_z_x = np.concatenate([points_in_cords_x_y_z[1, :], -points_in_cords_x_y_z[2, :],
                                                  points_in_cords_x_y_z[0, :]])
    points_pixel = np.dot(intrinsic_mat, points_in_cords_y_minus_z_x)

    # 归一化
    if abs(points_pixel[2]) < 1e-6:
        points_pixel = np.array([0, 0, 0])
    else:
        points_pixel = np.array([points_pixel[0] / points_pixel[2], points_pixel[1] / points_pixel[2], points_pixel[2]])
    return points_pixel


def transform_from_world_to_pixel(points, intrinsic_mat, extrinsic_mat):
    """
        将在世界坐标系中的点(齐次坐标)转换到像素坐标系并求出对应点的坐标和深度

        参数：
            points：世界坐标系中的点坐标矩阵
            intrinsic_mat：相机内参矩阵
            extrinsic_mat：相机外参矩阵

        返回：
            points_2d：像素坐标系下点的坐标和对应深度
    """
    points_2d = []

    for i in range(points.shape[1]):
        # 获取点在世界坐标系中的向量
        point_world = points[:, i].reshape(-1, 1)
        # 将点从世界坐标系转换到相机坐标系中
        point_camera = transform_from_world_to_camera(point_world, extrinsic_mat)
        # 将点从相机坐标系转换到像素坐标系中
        point_pixel = transform_from_camera_to_pixel(point_camera, intrinsic_mat)
        # 点在像素坐标系中的坐标（含深度）
        points_2d.append((point_pixel[1], point_pixel[0], point_pixel[2]))
    return points_2d


def get_vertices_pixel(intrinsic_mat, extrinsic_mat, obj_bbox, obj_transform, obj_tp):
    """
        将物体3d bounding box的顶点坐标投影到像素坐标系上

        参数：
            intrinsic_mat：相机内参矩阵
            extrinsic_mat：相机外参矩阵
            obj_bbox：物体bounding box
            obj_transform：物体在CARLA中的位姿
            obj_tp：物体的种类

        返回：
            vertices_pixel：3d bounding box的8个顶点在像素坐标系下的坐标
    """
    vertices = extension_to_vertices(obj_bbox)

    # actors
    if obj_tp == 1:
        bbox_transform = carla.Transform(obj_bbox.location, obj_bbox.rotation)
        vertices_local = transform_points(bbox_transform, vertices)
    # 环境物体
    else:
        box_location = carla.Location(obj_bbox.location.x - obj_transform.location.x,
                                      obj_bbox.location.y - obj_transform.location.y,
                                      obj_bbox.location.z - obj_transform.location.z)
        bbox_transform = carla.Transform(box_location, obj_bbox.rotation)
        vertices_local = transform_points(bbox_transform, vertices)

    # 获取3d bounding box在世界坐标系下八个顶点的坐标
    vertices_world = transform_points(obj_transform, vertices_local)
    # 将世界坐标系下的bbox八个顶点转换到像素坐标系中
    vertices_pixel = transform_from_world_to_pixel(vertices_world, intrinsic_mat, extrinsic_mat)
    return vertices_pixel


def extension_to_vertices(obj_bbox):
    """
        从CARLA物体的bounding box得到车辆3D bounding box的八个顶点坐标（以车辆中心为原点的坐标系）

        参数：
            obj_bbox：CARLA物体的bounding box

        返回：
            vertices：车辆3D bounding box的八个顶点齐次坐标(以车辆中心为原点的坐标系）[4 × 8 矩阵]
    """
    ext = obj_bbox.extent
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
    vertices = vertices.transpose()
    vertices = np.append(vertices, np.ones((1, vertices.shape[1])), axis=0)
    return vertices
