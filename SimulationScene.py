import carla


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
