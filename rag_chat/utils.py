import yaml



INDEX_YAML_FILE = "conf/index_conf.yaml"
INFERENCE_YAML_FILE = "conf/inference_conf.yaml"

def read_index_conf():
    with open(INDEX_YAML_FILE, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data

def read_inference_conf():
    with open(INFERENCE_YAML_FILE, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data



if __name__ == "__main__":
    config = read_index_conf()
    print(config)