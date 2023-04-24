from export_utils import *
from utils import config_transform_to_carla_transform


class DatasetSave:
    def __init__(self, config):
        self.config = config
        self.OUTPUT_FOLDER = None
        self.LIDAR_PATH = None
        self.KITTI_LABEL_PATH = None
        self.IMAGE_PATH = None
        self.DEPTH_IMAGE_PATH = None
        self.BBOX_IMAGE_PATH = None
        self.CALIBRATION_PATH = None
        self.LIDAR_LABEL_PATH = None
        self._generate_path(self.config["SAVE_CONFIG"]["ROOT_PATH"])
        self.captured_frame_no = self._current_captured_frame_num()

    def _generate_path(self, root_path):
        """ 生成数据存储的路径"""
        PHASE = "training"
        self.OUTPUT_FOLDER = os.path.join(root_path, PHASE)
        folders = ['calib', 'image', 'image_label', 'bbox_img', 'velodyne', 'lidar_label']

        for folder in folders:
            directory = os.path.join(self.OUTPUT_FOLDER, folder)
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.LIDAR_PATH = os.path.join(self.OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
        self.KITTI_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_label/{0:06}.txt')
        self.IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, 'image/{0:06}.png')
        self.BBOX_IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, 'bbox_img/{0:06}.png')
        self.CALIBRATION_PATH = os.path.join(self.OUTPUT_FOLDER, 'calib/{0:06}.txt')
        self.LIDAR_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'lidar_label/{0:06}.txt')

    def _current_captured_frame_num(self):
        """获取文件夹中存在的数据量"""
        label_path = os.path.join(self.OUTPUT_FOLDER, 'image_label/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        print("当前存在{}个数据".format(num_existing_data_files))
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(
                self.OUTPUT_FOLDER))
        if answer.upper() == "O":
            logging.info(
                "Resetting frame number to 0 and overwriting existing")
            return 0
        logging.info("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files

    def save_training_files(self, data):
        lidar_filename = self.LIDAR_PATH.format(self.captured_frame_no)
        kitti_label_filename = self.KITTI_LABEL_PATH.format(self.captured_frame_no)
        img_filename = self.IMAGE_PATH.format(self.captured_frame_no)
        bbox_img_filename = self.BBOX_IMAGE_PATH.format(self.captured_frame_no)
        calib_filename = self.CALIBRATION_PATH.format(self.captured_frame_no)
        lidar_label_filename = self.LIDAR_LABEL_PATH.format(self.captured_frame_no)

        for agent, dt in data["agents_data"].items():
            extrinsic = dt["extrinsic"]

            camera_transform = config_transform_to_carla_transform(self.config["SENSOR_CONFIG"]["RGB"]["TRANSFORM"])
            lidar_transform = config_transform_to_carla_transform(self.config["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"])

            save_ref_files(self.OUTPUT_FOLDER, self.captured_frame_no)
            save_image_data(img_filename, dt["sensor_data"][0])
            save_bbox_image_data(bbox_img_filename, dt["bbox_img"])
            save_label_data(kitti_label_filename, dt["image_labels_kitti"])
            save_calibration_matrices([camera_transform, lidar_transform], calib_filename, dt["intrinsic"])
            save_lidar_data(lidar_filename, dt["sensor_data"][2], extrinsic)
            save_label_data(lidar_label_filename, dt["pc_labels_kitti"])
        self.captured_frame_no += 1
