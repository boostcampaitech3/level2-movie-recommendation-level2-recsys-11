# %%
import random
from pathlib import Path
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


# %%
data_dir = '/opt/ml/input/data/train/'
N = 50
PDF = norm()
    
base_path = Path(data_dir)
user_item_time = pd.read_csv(base_path / 'train_ratings.csv')
cnt_series = user_item_time["item"].value_counts()[:N]

weight = [PDF.pdf(i * 4 / N) for i in range(N)]
topN = cnt_series.index.to_list()

# %%
def neg_sample(item_set):
    item = random.choices(topN, weights=weight)[0]
    while item in item_set:
        item = random.choices(topN, weights=weight)[0]
    return item

def extract_samples(sequence, length):
    not_watched = [i for i in topN if not i in sequence]
    l = len(not_watched)
    assert not_watched, f"""해당 사용자가 Top {N} 안의 모든 영화를 소비함. 
    {__name__}의 필드 N 값을 증가시키기를 권장함."""
    return random.choices(not_watched, weights=weight[:l], k=length)

# %%
if __name__ == '__main__':
    # 실제 뽑힌 값들의 분포
    df = pd.DataFrame(
        extract_samples([47, 50], 100000)
        )
    plt.plot(df.value_counts().tolist())
    
# %%
if __name__ == '__main__':
    # 실제 뽑힌 값들과 TopN과의 차이 확인
    compare = zip(topN, df.value_counts().index.tolist()).__iter__()
    print('topN\tpred\tcorrect?')
    print('=' * 30)
    for _ in range(20):
        item_topN, pred = compare.__next__()
        print(f'{item_topN}\t{pred[0]}\t{item_topN==pred[0]}')
    
# %%
