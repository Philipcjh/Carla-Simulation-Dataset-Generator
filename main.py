from utils import yaml_to_config
from SimulationScene import SimulationScene
from DatasetSave import DatasetSave

def main():
    config = yaml_to_config("configs.yaml")
    scene = SimulationScene(config)
    dataset_save = DatasetSave(config)
    try:
        # 开启同步模式
        scene.set_synchrony()
        # 在场景中生成actors(车辆与行人)
        scene.spawn_actors()
        # 设置actors自动运动
        scene.set_actors_route()
        # 生成agent（用于放置传感器的车辆与传感器）
        scene.spawn_agent()
        # 设置观察视角(与RGB相机一致)
        scene.set_spectator()
        # 监听传感器信息
        scene.listen_sensor_data()

        # 帧数
        frame = 0
        # 记录步长
        STEP = config["SAVE_CONFIG"]["STEP"]

        while True:
            if frame % STEP == 0:
                # 记录帧
                dataset = scene.record_tick()
                dataset_save.save_training_files(dataset)

            else:
                # 运行帧
                scene.world.tick()

            frame += 1
    finally:
        # 恢复默认设置
        scene.set_recover()


if __name__ == '__main__':
    main()
