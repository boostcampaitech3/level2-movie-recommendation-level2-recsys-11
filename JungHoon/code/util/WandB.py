# %%
import wandb

import copy


# %%
PJT = "MovieRec"
ENTITY="iksadnorth"

# %%
def init(**kwargs):
    wandb.init(project=PJT, entity=ENTITY, **kwargs)

def config(args, attr_ignore:list=None):
    wandb_config = copy.deepcopy(args.__dict__)
    for attr in attr_ignore:
        wandb_config.pop(attr)
    wandb.config.update(wandb_config)