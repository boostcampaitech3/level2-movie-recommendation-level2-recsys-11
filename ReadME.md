# Movie Recommendation

## 프로젝트 개요

<p align="center"><img src="https://user-images.githubusercontent.com/66674140/173006936-cf3a0078-e489-45c4-bbc1-a565d28dd078.PNG"></p>


이 프로젝트는 timestamp를 고려한 사용자의 순차적인 이력을 고려하고 implicit feedback을 고려한다는 점에서, 1-5점 평점 (explicit feedback) 기반의 행렬을 사용한 협업 필터링 문제와 차별화됩니다. 

implicit feedback 기반의 sequential recommendation 시나리오를 바탕으로 사용자의 time-ordered sequence에서 일부 item이 누락된 (dropout)된 상황을 상정합니다. 

뿐 만 아니라 여러가지 아이템 (영화)과 관련된 content (side-information)가 존재합니다.

원본 데이터가 있다면 특정 시점 이후의 데이터 (sequential)와 특정 시점 이전의 일부 데이터(static) 데이터를 임의로 추출하여, 정답 (ground-truth) 데이터로 사용하고 있습니다. 

즉, 데이터 셋의 구성은 기존에 존재하던 sequential recommendation 시나리오에서 의도적으로 데이터를 누락 시킨 상황입니다.


## 평가 방법
변형된 Recall@K을 사용하고 그것은 다음과 같습니다.
<p align="center"><img src="https://user-images.githubusercontent.com/66674140/173007196-33557772-fb72-48fe-b923-babd14c3ed9f.PNG"></p>

## Members

|                                                  [김연요](https://github.com/arkdusdyk)                                                   |                                                                          [김진우](https://github.com/Jinu-uu)                                                                           |                                                 [박정훈](https://github.com/iksadNorth)                                                  |                                                                        [이호진](https://github.com/ili0820)                                                                         |                                                                         [최준혁](https://github.com/JHchoiii)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/69205130?s=400&u=a14d779da6a9023a45e60e44072436d356a9461c&v=4)](https://github.com/arkdusdyk) | [![Avatar](https://avatars.githubusercontent.com/u/82719310?v=4)](https://github.com/Jinu-uu) | [![Avatar](https://avatars.githubusercontent.com/u/66674140?v=4)](https://github.com/iksadNorth) | [![Avatar](https://avatars.githubusercontent.com/u/65278309?v=4)](https://github.com/ili0820) | [![Avatar](https://avatars.githubusercontent.com/u/99862931?v=4)](https://github.com/JHchoiii) |


## 실험 기록
<p align="center"><img src="https://user-images.githubusercontent.com/66674140/173007753-582b1a82-b508-4eae-af64-7c00041b4211.PNG"></p>


## Modeling
### S3Rec (0.0884) 

기존에 주어진 baseline 모델은 모든 side-information을 고려하지 않고 ‘장르’ 속성만을 활용했습니다. 

장르 뿐만아니라 연도, 감독등도 활용 해보았지만 각 속성 사이의 고차원 interaction을 고려하지 못하여 오히려 Score가 낮아진 것이라 판단했습니다.

### Bert4Rec (0.0138)

베이스라인 코드와 유사하게 static 보다는 sequential 문제에 특화된 모델이었고, 역시나 성능이 그렇게 좋진 않았습니다.

### Mult-VAE (0.1289)

베이스 라인 모델인 S3Rec 보다 훨씬 좋은 성능을 보였고, sequential 보다는 static을 고려한 모델이 더 이 과제에 적합한 듯한 모습을 보여 주었고 이 때문에, VAE based 모델을 더 찾아 보게 되었습니다.

### RecVAE (0.1464)

전처리 방식과 모델의 pred가 MultiVAE와 동일하여 적용하기 쉬웠습니다. 

그에 비해 성능은 뛰어나 리더보드 3위 까지 상승하는 모습을 보여주었습니다.

### VASP (0.0633)

FLVAE와 Neural EASE을 앙상블한 모델. 

MovieLens20M 에서 VAE 와 유사한 방식이라고 생각되어 시도해보았다. 리더보드 제출결과가 제공되었던 baseline code 인 S3REC보다 좋지 않아, 진행을 중단했습니다.

### EVCF (0.0884)

Multi VAE에서 추가된 기능 : Hierarchical VAE, Gating Mechanism

VAE 모델과 유사한 방식이지만 다른 추가적인 부분과 MovieLens20M에서의 성능 향상의 근거로 시도해보았고 결과적으로 리더보드에서 결과가 좋지 않아 중단하였습니다.

### EASE (0.1415)

어렵지 않은 전처리와 정말 얕은 모델이기 때문에 속도가 정말 빨랐습니다. 

성능 또한 RECVAE만큼은 아니어도 비교적 좋게 나왔습니다.


## Ensemble
### SOFT VOTING

- 개요

    각 모델이 “모든 유저의 모든 아이템에 대한 Rate Prediction” 결과를 가중치 합산한 결과물을 토대로 TOP10을 뽑는 아이디어.

- 구현 아이디어

    1. 각 모델을 저마다의 방식으로 학습을 시킨 후, User-Item Score Table을 생성하게 하는 ensemble.py를 만들었습니다.

    2. 각 모델에 대해 정규화를 진행한 후, 해당 Table을 Elementwise 가중치합을 하고 그 결과물의 Top10을 뽑아 제출ㅎ.

- 결과

    각각을 적용했을 때보다 훨씬 성능이 저하됨을 관찰할 수 있었습니다. 

    각 모델의 잘못된 판단이 오히려 상대의 훌륭한 판단에 간섭을 일으켜서 성능을 저하시킨 것으로 판단됩니다.

### HARD VOTING

위에서 언급한 모델들(S3REC,BERT4REC,Multi VAE,RECVAE,VASP,EVCF,EASE)과
BPR, BiVAE,LightGBM등의 결과중 가장 많이 나온 10개의 영화를 추출했습니다. 

성능에 대한 가중치를 주지 않아 성능이 좋지 않은 모델처럼 결과가 나오게 되었습니다.


## 하이퍼파라미터 튜닝

- WANDB sweep 사용

- 가장 성능이 좋았던 RECVAE 모델을 기준으로 아래의 parameter들을 변경하며 가장 좋은 성능을 보여주는 최적의 파라미터를 찾아서 사용 하였습니다.

- 결과 : 이 모델을 사용해서 0.1480의 점수 달성

- 상세 설정(파라미터 변경 횟수, 등)이 어려워 완벽하게 원하는 대로 파라미터를 탐색해 볼 수 없었습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/66674140/173012083-b12e7b2c-9db8-4464-a713-ffd3283bac5e.PNG"></p>
