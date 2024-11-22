import yaml

def load_config(config_path="config.yaml"):
    """
    Loads the configuration from the given YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Returns:
        dict: Loaded configuration settings.
    """
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config
