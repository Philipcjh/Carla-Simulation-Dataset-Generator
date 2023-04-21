import carla
import random
import logging


class SimulationScene:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        # 设置Carla地图
        self.world = self.client.load_world('Town02')
        self.traffic_manager = self.client.get_trafficmanager()
        self.init_settings = None
        self.frame = None
        self.actors = {"non_agents": [], "walkers": [], "agents": [], "sensors": {}}
        self.data = {"sensor_data": {}, "environment_data": None}  # 记录每一帧的数据
        self.vehicle = None

    # 开启同步模式
    def set_synchrony(self):
        self.init_settings = self.world.get_settings()
        settings = self.world.get_settings()
        weather = carla.WeatherParameters(cloudiness=0.0, precipitation=0.0, sun_altitude_angle=50.0)
        self.world.set_weather(weather)
        settings.synchronous_mode = True
        # 固定时间步长 (0.05s, 20fps)
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    # 在场景中生成actors(车辆与行人)
    def spawn_actors(self):
        # 生成车辆
        num_of_vehicles = self.cfg["CARLA_CONFIG"]["NUM_OF_VEHICLES"]
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
        num_of_walkers = self.cfg["CARLA_CONFIG"]["NUM_OF_WALKERS"]
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

    # 设置actors自动运动
    def set_actors_route(self):
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

    # 恢复默认设置
    def set_recover(self):
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