# 실행 경로 조정
cd /opt/ml/baseline/code/script

# 사전 훈련 시작
python ../run_pretrain.py --data_dir /opt/ml/input/data/train/

# 훈련 시작
python ../run_train.py --using_pretrain --data_dir /opt/ml/input/data/train/

# inference 시작
python ../inference.py --data_dir /opt/ml/input/data/train/