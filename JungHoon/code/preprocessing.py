# %%
import pandas as pd
from util import LoadJson


# %%
class Dummy:
    pass
    
args = Dummy()
LoadJson.dump(args, "/opt/ml/baseline/code/config.json")

# %%
def main(attr:str="genre"):
    assert attr in ('genre', 'director', 'writer', 'year')
    
    attr_df = pd.read_csv(f"/opt/ml/input/data/train/{attr}s.tsv", sep="\t")
    array, index = pd.factorize(attr_df[attr])
    print("array.shape : ", array.shape)
    print("index :\n", index)
    attr_df[attr] = array
    print(args.data_name)
    attr_df.groupby("item")[attr].apply(list).to_json(
        f"/opt/ml/input/data/train/{args.data_name}_item2attributes.json"
    )


# %%
if __name__ == "__main__":
    keys = ('genre', 'director', 'writer', 'year')
    main(keys[0])

# %%
