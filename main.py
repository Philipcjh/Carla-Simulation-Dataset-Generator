from config import config_from_yaml
from SimulationScene import SimulationScene


def main():
    config = config_from_yaml("configs.yaml")
    scene = SimulationScene(config)


if __name__ == '__main__':
    main()
