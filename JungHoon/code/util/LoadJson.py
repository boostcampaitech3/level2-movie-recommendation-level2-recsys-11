# %%
import json


# %%
def dump(args, path_json:str):
    with open(path_json, 'r') as f:
        new_dict = json.load(f)
    args.__dict__.update(new_dict)
    return args


# %%
if __name__ == '__main__':
    class Dummy:
        pass
    
    args = Dummy()
    args = dump(args, "/opt/ml/baseline/code/config.json")
    print(args.__dict__)
    
# %%
