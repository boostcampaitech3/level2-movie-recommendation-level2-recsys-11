# Enhancing VAEs for Collaborative Filtering: Flexible Priors & Gating Mechanisms
This is the source code used for experiments for the paper published in RecSys '19:  
"Enhancing VAEs for Collaborative Filtering: Flexible Priors & Gating Mechanisms"    
(arxiv preprint: https://arxiv.org/abs/1911.00936, ACM DL: https://dl.acm.org/citation.cfm?id=3347015)

An example of training a hierarchical VampPrior VAE for Collaborative Filtering on the Netflix dataset is as follows:
`python experiment.py  --dataset_name="netflix" --max_beta=0.3 --model_name="hvamp" --gated --input_type="binary" --z1_size=200 --z2_size=200 --hidden_size=600 --num_layers=2 --note="Netflix(H+Vamp+Gate)"`

### Requirements
Requirements are listed in `requirements.txt`

### Datasets
Datasets should be downloaded and preprocessed according to instructions in `./datasets/`

### Acknowledgements
Many of our code is reformulated based on https://github.com/dawenl/vae_cf and https://github.com/jmtomczak/vae_vampprior




# data

python experiment.py  --inference --epochs=3 --max_beta=0.4 --gated --input_type=“binary” --z1_size=200 --z2_size=200 --hidden_size=600 --num_layers=2 --note=“1”

### 수정사항
1. experiment에 inference 추가 
2. 콘솔에 --inference 넣고 돌리면 submission.csv 까지 나오게 만들었
3. 근데 submission.csv index 제대로 안맞춰서 나올수도 있어서 jupyter같은걸로 좀 손봐야함
4. 그리고 inference만 따로 돌게는 못했음 ㅋ
5. 위에 # data에 있는거 처럼 하면 돌아가긴함