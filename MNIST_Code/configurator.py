import os
import sys

from omegaconf import OmegaConf
from ast import literal_eval


def resolve_config_from_cli():
    # Initialize an empty config or a base config file
    config = OmegaConf.create()

    # Iterate through the arguments passed via sys.argv
    for arg in sys.argv[1:]:
        if not arg.startswith('--'):
            # If it's not starting with '--', assume it's a YAML file
            if os.path.isfile(arg):
                # Load the YAML file and merge it with the current config
                yaml_config = OmegaConf.load(arg)
                config = OmegaConf.merge(config, yaml_config)
            else:
                print(f"Warning: File '{arg}' not found!")
        else:
            # If it starts with '--' and contains '=', split it into key and value
            assert arg.startswith('--') and '=' in arg
            key, val = arg[2:].split('=', 1)
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            # Update the configuration with this key-value pair
            OmegaConf.update(config, key, attempt, merge=True)
    
    return config
