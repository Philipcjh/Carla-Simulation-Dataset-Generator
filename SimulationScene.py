import carla
import random
import logging
import queue
import numpy as np
from utils.utils import config_transform_to_carla_transform, set_camera_intrinsic, object_filter_by_distance
from utils.label import spawn_dataset


class SimulationScene:
    def __init__(self, config):
        """
            初始化

            参数：
                config：预设配置
        """
        self.config = config
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        # 设置Carla地图
        self.world = None
        self.traffic_manager = self.client.get_trafficmanager()
        self.init_settings = None
        self.frame = None
        self.actors = {"non_agents": [], "walkers": [], "agents": [], "sensors": {}}
        self.data = {"sensor_data": {}, "environment_data": None}  # 记录每一帧的数据
        self.vehicle = None

    def set_map(self):
        """
            设置场景地图
        """
        self.world = self.client.load_world('Town02')

    def set_weather(self):
        """
            设置场景天气
        """
        weather = carla.WeatherParameters(cloudiness=0.0, precipitation=0.0, sun_altitude_angle=50.0)
        self.world.set_weather(weather)

    def set_synchrony(self):
        """
            开启同步模式
        """
        self.init_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        # 固定时间步长 (0.05s, 20fps)
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def spawn_actors(self):
        """
            在场景中生成actors(车辆与行人)
        """
        # 生成车辆
        num_of_vehicles = self.config["CARLA_CONFIG"]["NUM_OF_VEHICLES"]
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if num_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
            num_of_vehicles = num_of_vehicles
        elif num_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, num_of_vehicles, number_of_spawn_points)
            num_of_vehicles = number_of_spawn_points

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= num_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform))

            for response in self.client.apply_batch_sync(batch):
                if response.error:
                    continue
                else:
                    self.actors["non_agents"].append(response.actor_id)

        # 生成行人
        num_of_walkers = self.config["CARLA_CONFIG"]["NUM_OF_WALKERS"]
        blueprintsWalkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        spawn_points = []
        for i in range(num_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                continue
            else:
                self.actors["walkers"].append(response.actor_id)
        print("spawn {} vehicles and {} walkers".format(len(self.actors["non_agents"]),
                                                        len(self.actors["walkers"])))
        self.world.tick()

    def set_actors_route(self):
        """
            设置actors自动运动
        """
        # 设置车辆Autopilot
        self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        self.traffic_manager.set_synchronous_mode(True)
        vehicle_actors = self.world.get_actors(self.actors["non_agents"])
        for vehicle in vehicle_actors:
            vehicle.set_autopilot(True, self.traffic_manager.get_port())

        # 设置行人随机运动
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        batch = []
        for i in range(len(self.actors["walkers"])):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(),
                                                  self.actors["walkers"][i]))
        controllers_id = []
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                pass
            else:
                controllers_id.append(response.actor_id)
        self.world.set_pedestrians_cross_factor(0.2)

        for walker_id in controllers_id:
            # 开始步行
            self.world.get_actor(walker_id).start()
            # 设置步行到随机目标
            destination = self.world.get_random_location_from_navigation()
            self.world.get_actor(walker_id).go_to_location(destination)
            # 行人最大速度
            self.world.get_actor(walker_id).set_max_speed(10)

    def spawn_agent(self):
        """
            生成agent（用于放置传感器的车辆与传感器）
        """
        vehicle_bp = random.choice(self.world.get_blueprint_library().filter(self.config["AGENT_CONFIG"]["BLUEPRINT"]))
        config_transform = self.config["AGENT_CONFIG"]["TRANSFORM"]
        carla_transform = config_transform_to_carla_transform(config_transform)
        # transform = random.choice(self.world.get_map().get_spawn_points())
        agent = self.world.spawn_actor(vehicle_bp, carla_transform)
        agent.set_autopilot(False, self.traffic_manager.get_port())
        self.actors["agents"].append(agent)

        # 生成config中预设的传感器
        self.actors["sensors"][agent] = []
        for sensor, config in self.config["SENSOR_CONFIG"].items():
            sensor_bp = self.world.get_blueprint_library().find(config["BLUEPRINT"])
            for attr, val in config["ATTRIBUTE"].items():
                sensor_bp.set_attribute(attr, str(val))
            config_transform = config["TRANSFORM"]
            carla_transform = config_transform_to_carla_transform(config_transform)
            sensor = self.world.spawn_actor(sensor_bp, carla_transform, attach_to=agent)
            self.actors["sensors"][agent].append(sensor)
        self.world.tick()

    def set_spectator(self):
        """
            设置观察视角(与RGB相机一致)
        """
        spectator = self.world.get_spectator()

        # agent(放置传感器的车辆)位姿「相对世界坐标系」
        agent_transform_config = self.config["AGENT_CONFIG"]["TRANSFORM"]
        agent_transform = config_transform_to_carla_transform(agent_transform_config)

        # RGB相机位姿「相对agent坐标系」
        rgb_transform_config = self.config["SENSOR_CONFIG"]["RGB"]["TRANSFORM"]
        rgb_transform = config_transform_to_carla_transform(rgb_transform_config)

        # spectator位姿「相对世界坐标系」
        spectator_location = agent_transform.location + rgb_transform.location
        spectator_rotation = carla.Rotation(yaw=agent_transform.rotation.yaw + rgb_transform.rotation.yaw,
                                            pitch=agent_transform.rotation.pitch + rgb_transform.rotation.pitch,
                                            roll=agent_transform.rotation.roll + rgb_transform.rotation.roll)
        spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

    def set_recover(self):
        """
            数据采集结束后，恢复默认设置
        """
        self.world.apply_settings(self.init_settings)
        self.traffic_manager.set_synchronous_mode(False)
        batch = []
        for actor_id in self.actors["non_agents"]:
            batch.append(carla.command.DestroyActor(actor_id))
        for walker_id in self.actors["walkers"]:
            batch.append(carla.command.DestroyActor(walker_id))
        for agent in self.actors["agents"]:
            for sensor in self.actors["sensors"][agent]:
                sensor.destroy()
            agent.destroy()
        self.client.apply_batch_sync(batch)

    def listen_sensor_data(self):
        """
            监听传感器信息
        """
        for agent, sensors in self.actors["sensors"].items():
            self.data["sensor_data"][agent] = []
            for sensor in sensors:
                q = queue.Queue()
                self.data["sensor_data"][agent].append(q)
                sensor.listen(q.put)

    def retrieve_data(self, q):
        """
            检查并获取传感器数据

            参数：
                q: CARLA原始数据

            返回：
                data：检查后的数据
        """
        while True:
            data = q.get()
            # 检查传感器数据与场景是否处于同一帧
            if data.frame == self.frame:
                return data

    def record_tick(self):
        """
            记录帧

            返回：
                data：CARLA传感器相关数据（原始数据，内参，外参等）
        """
        data = {"environment_objects": None, "actors": None, "agents_data": {}}
        self.frame = self.world.tick()

        data["environment_objects"] = self.world.get_environment_objects(carla.CityObjectLabel.Any)
        data["actors"] = self.world.get_actors()

        # 生成RGB图像的分辨率
        image_width = self.config["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"]
        image_height = self.config["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"]

        for agent, dataQue in self.data["sensor_data"].items():
            original_data = [self.retrieve_data(q) for q in dataQue]
            assert all(x.frame == self.frame for x in original_data)
            data["agents_data"][agent] = {}
            data["agents_data"][agent]["sensor_data"] = original_data
            # 设置传感器内参（仅相机有内参）
            data["agents_data"][agent]["intrinsic"] = set_camera_intrinsic(image_width, image_height)
            # 设置传感器外参
            data["agents_data"][agent]["extrinsic"] = np.mat(
                self.actors["sensors"][agent][0].get_transform().get_matrix())
            # 设置传感器的carla位姿
            data["agents_data"][agent]["transform"] = self.actors["sensors"][agent][0].get_transform()
            # 设置传感器的种类
            data["agents_data"][agent]["type"] = agent

        # 根据预设距离对场景中的物体进行过滤
        data = object_filter_by_distance(data, self.config["FILTER_CONFIG"]["PRELIMINARY_FILTER_DISTANCE"])

        dataset = spawn_dataset(data)

        return dataset
