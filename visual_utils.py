from utils import yaml_to_config

config = yaml_to_config("configs.yaml")
WINDOW_WIDTH = config["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_x"]
WINDOW_HEIGHT = config["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_y"]


def rectify_2d_bounding_box(bbox_2d):
    """
        对2d bounding box进行校正，将超出图片范围外的部分

            参数：
                bbox_2d：2d bounding box的左上和右下的像素坐标

            返回：
                bbox_2d_rectified：校正后的2d bounding box的左上和右下的像素坐标
    """
    min_x, min_y, max_x, max_y = bbox_2d
    if point_in_canvas((min_y, min_x)) or point_in_canvas((max_y, max_x)):
        min_x_rectified, min_y_rectified = set_point_in_canvas((min_x, min_y))
        max_x_rectified, max_y_rectified = set_point_in_canvas((max_x, max_y))
        bbox_2d_rectified = [min_x_rectified, min_y_rectified, max_x_rectified, max_y_rectified]
        return bbox_2d_rectified
    else:
        return None


def draw_2d_bounding_box(image, bbox_2d):
    """
        在图像中绘制2d bounding box

            参数：
                image：RGB图像
                bbox_2d：2d bounding box的左上和右下的像素坐标

    """
    min_x, min_y, max_x, max_y = bbox_2d

    # 将2d bounding box的四条边设置为红色
    for y in range(min_y, max_y):
        image[y, min_x] = (255, 0, 0)
        image[y, max_x] = (255, 0, 0)

    for x in range(min_x, max_x):
        image[max_y, int(x)] = (255, 0, 0)
        image[min_y, int(x)] = (255, 0, 0)

    # 将2d bounding box的四个顶点设置为绿色
    image[min_y, min_x] = (0, 255, 0)
    image[min_y, max_x] = (0, 255, 0)
    image[max_y, min_x] = (0, 255, 0)
    image[max_y, max_x] = (0, 255, 0)


def draw_3d_bounding_box(image, vertices_pos2d):
    # Shows which verticies that are connected so that we can draw lines between them
    # The key of the dictionary is the index in the bbox array, and the corresponding value is a list of indices
    # referring to the same array.
    vertex_graph = {0: [1, 2, 4],
                    1: [0, 3, 5],
                    2: [0, 3, 6],
                    3: [1, 2, 7],
                    4: [0, 5, 6],
                    5: [1, 4, 7],
                    6: [2, 4, 7]}
    # Note that this can be sped up by not drawing duplicate lines
    for vertex_idx in vertex_graph:
        neighbour_index = vertex_graph[vertex_idx]
        from_pos2d = vertices_pos2d[vertex_idx]
        for neighbour_idx in neighbour_index:
            to_pos2d = vertices_pos2d[neighbour_idx]
            if from_pos2d is None or to_pos2d is None:
                continue
            y1, x1 = from_pos2d[0], from_pos2d[1]
            y2, x2 = to_pos2d[0], to_pos2d[1]
            # Only stop drawing lines if both are outside
            if not point_in_canvas((y1, x1)) and not point_in_canvas((y2, x2)):
                continue

            for x, y in get_line(x1, y1, x2, y2):
                if point_in_canvas((y, x)):
                    image[int(y), int(x)] = (0, 0, 255)

            if point_in_canvas((y1, x1)):
                image[int(y1), int(x1)] = (255, 0, 255)
            if point_in_canvas((y2, x2)):
                image[int(y2), int(x2)] = (255, 0, 255)


def get_line(x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # print("Calculating line from {},{} to {},{}".format(x1,y1,x2,y2))
    points = []
    is_steep = abs(y2 - y1) > abs(x2 - x1)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    delta_x = x2 - x1
    delta_y = abs(y2 - y1)
    error = int(delta_x / 2)
    y = y1
    if y1 < y2:
        y_step = 1
    else:
        y_step = -1
    for x in range(x1, x2 + 1):
        if is_steep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= delta_y
        if error < 0:
            y += y_step
            error += delta_x
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


def set_point_in_canvas(point):
    x, y = point[0], point[1]

    if x < 0:
        x = 0
    elif x >= WINDOW_WIDTH:
        x = WINDOW_WIDTH - 1

    if y < 0:
        y = 0
    elif y >= WINDOW_HEIGHT:
        y = WINDOW_HEIGHT - 1

    return x, y


def point_in_canvas(pos):
    """
    如果点在图片内，返回true
    """
    if (pos[0] >= 0) and (pos[0] < WINDOW_HEIGHT) and (pos[1] >= 0) and (pos[1] < WINDOW_WIDTH):
        return True
    return False