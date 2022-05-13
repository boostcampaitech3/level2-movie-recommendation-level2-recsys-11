# 스크립트 파일 권한 755 부여
chmod 755 run.sh

# background 실행
nohup /opt/ml/baseline/code/script/run.sh > bg_log.out $