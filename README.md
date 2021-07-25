# Project

```bash
├── DATA
│   ├── Final_DATA
│   │   ├── task02_test
│   │   └── task02_train
│   ├── polygon_iou.json
│   └── data.csv
├── config
│   └── config.yaml
├── model
│   ├── submission_model_weight
│   └── preprocessing
│       └── preprocessing_model_weight
├── modules
│   ├── dataset.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── scheduler.py
│   ├── solver.py
│   ├── transform.py
│   └── utils.py
├── output
│   ├── log
│   ├── runs
│   │   └── output_model_weight
│   └── result.json
├── README.md
├── predict.py
└── train.py
```
## config
모델 학습을 위한 설정
## data
task02_train : train data <br>
task02_test : test data  <br>
data.csv : 원본 데이터를 전처리한 데이터 csv <br>
polygon_iou.json : 전처리한 데이터의 iou 값 파일
## model
submission_model_weight : 제출용 model weight 파일 <br>
preprocessing_model_weight : 전처리를 위한 model weight 파일 <br>
## modules
#### dataset.py
dataset
#### transform.py
augmentation
#### utils.py
yaml 파일 로드 <br>
logger <br>
seed 설정 <br>
#### preprocessing.py
데이터셋 전처리
#### models.py
Encoder, Decoder
#### scheduler.py
scheduler
#### solver.py
전처리와 학습에 필요한 함수
## output
log : train log <br>
runs : 학습 결과 model weight <br>
result.json : predict 결과 파일 <br>

<br>

## Code 실행방법

### 1. preprocess.py 실행
```console
python preprocess.py
```
### 2. train.py 실행
```console
python train.py
```

### 3. predict.py 실행
```console
python predict.py
```
