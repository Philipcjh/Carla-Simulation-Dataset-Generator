import sys
import yaml
import carla


def yaml_to_config(file):
    try:
        with open(file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config
    except:
        return None


def config_transform_to_carla_transform(trans_config):
    carla_transform = carla.Transform(carla.Location(trans_config["location"][0],
                                               trans_config["location"][1],
                                               trans_config["location"][2]),
                                carla.Rotation(trans_config["rotation"][0],
                                               trans_config["rotation"][1],
                                               trans_config["rotation"][2]))
    return carla_transform
