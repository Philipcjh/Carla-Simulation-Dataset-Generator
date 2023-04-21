from utils import yaml_to_config
from SimulationScene import SimulationScene


def main():
    config = yaml_to_config("configs.yaml")
    scene = SimulationScene(config)
    try:
        scene.set_synchrony()
        scene.spawn_actors()
        scene.set_actors_route()
        scene.spawn_agent()
        scene.set_spectator()
        scene.world.tick()
    finally:
        scene.set_recover()


if __name__ == '__main__':
    main()
