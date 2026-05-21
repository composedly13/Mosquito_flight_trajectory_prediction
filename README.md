# Mosquito Flight Trajectory Prediction

월간 데이콘 **모기 비행 궤적 예측 AI 경진대회** 풀이 레포지토리입니다.

> 대회 링크: https://dacon.io/competitions/official/236716/overview/description

---

## 대회 개요

LiDAR 센서로 관측된 모기의 3차원 궤적 데이터를 바탕으로, 시스템 처리 지연(80ms) 이후의 모기 위치를 예측합니다.

- **입력**: -400ms ~ 0ms 구간의 11개 시점 좌표 (40ms 간격)
- **출력**: +80ms 시점의 3차원 좌표 (x, y, z)
- **평가 지표**: R-Hit@1cm

```python
def r_hit(pred, true):
    R_HIT = 0.01
    distance = np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1)
    return np.mean(distance <= R_HIT)
```

---

## 데이터셋

### 좌표계

LiDAR **sensor-local** 3차원 좌표계 기준입니다 (방 기준 절대 좌표 아님).

| 축 | 방향 | 단위 |
|---|---|---|
| x | forward | m |
| y | left | m |
| z | up | m |

### 구성

| 파일/폴더 | 설명 | 샘플 수 |
|---|---|---|
| `data/train/` | 학습용 샘플 CSV | 10,000개 |
| `data/test/` | 평가용 샘플 CSV | 10,000개 |
| `data/train_labels.csv` | 학습 정답 레이블 | 10,000행 |
| `data/sample_submission.csv` | 제출 양식 | - |

### 샘플 CSV 구조 (`train/`, `test/`)

각 파일은 모기 1개체의 관측 시계열입니다.

```
timestep_ms,x,y,z
-400,3.58,-0.08,2.53
-360,3.57,-0.10,2.51
...
0,3.51,-0.19,2.39
```

- 행 수: 11개 (고정)
- `timestep_ms`: -400 ~ 0 (40ms 간격)

### 레이블 CSV 구조 (`train_labels.csv`)

```
id,x,y,z
TRAIN_00001,3.49,-0.21,2.35
...
```

- `id`: 해당 train 샘플 파일명 (확장자 제외)
- x, y, z: 마지막 관측(0ms) 기준 **+80ms** 시점의 실제 좌표

### 환경 다양성

학습/평가 데이터는 **서로 다른 장소 및 환경**을 기반으로 구성됩니다.

- 실내, 복도, 창고, 반실외, 야외 등
- 속도, 가속도, scene ID 등 보조 정보는 **제공되지 않음**
- 오직 과거 좌표 변화만으로 미래 위치를 예측해야 함

---

## 프로젝트 구조

```
.
├── data/                        # 대회 제공 데이터 (레포 미포함)
│   ├── train/
│   │   ├── TRAIN_00001.csv
│   │   └── ...                  # 총 10,000개
│   ├── test/
│   │   ├── TEST_00001.csv
│   │   └── ...                  # 총 10,000개
│   ├── train_labels.csv
│   └── sample_submission.csv
├── experiments/
│   └── configs/
│       └── baseline.yaml
├── src/
│   ├── data/
│   │   ├── loader.py            # 데이터 로딩
│   │   └── features.py          # 피처 엔지니어링
│   ├── models/
│   │   ├── physics.py           # 물리 기반 모델
│   │   └── ml.py                # 머신러닝 모델
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── submissions/                 # 제출 파일 (레포 미포함)
├── README.md
└── requirements.txt
```

---

## 환경 설정

```bash
pip install -r requirements.txt
```

---

## 브랜치 구조

| 브랜치 | 용도 |
|---|---|
| `main` | 문서, 프로젝트 구조 |
| `dev` | 모델 개발 및 실험 |
