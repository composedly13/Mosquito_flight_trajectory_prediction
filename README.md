# Mosquito Flight Trajectory Prediction

월간 데이콘 **모기 비행 궤적 예측 AI 경진대회** 풀이 레포지토리입니다.

> 대회 링크: https://dacon.io/competitions/official/236716/overview/description

---

## 대회 개요

LiDAR 센서로 관측된 모기의 3차원 궤적 데이터를 바탕으로, 시스템 처리 지연(80ms) 이후의 모기 위치를 예측합니다.

- **입력**: -400ms ~ 0ms 구간의 11개 시점 좌표 (40ms 간격)
- **출력**: +80ms 시점의 3차원 좌표 (x, y, z)
- **평가 지표**: R-Hit@1cm — 예측과 실제 거리가 1cm 이내인 비율

```python
def r_hit(pred, true):
    return np.mean(np.linalg.norm(pred - true, axis=-1) <= 0.01)
```

---

## 데이터셋

### 좌표계

LiDAR **sensor-local** 3차원 좌표계 기준

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
| `data/train_labels.csv` | 학습 정답 레이블 (id, x, y, z) | 10,000행 |
| `data/sample_submission.csv` | 제출 양식 | - |

각 샘플 CSV: `timestep_ms, x, y, z` (11행, -400ms ~ 0ms)

---

## 데이터 분석 (EDA)

train 10,000개 샘플 전수 분석 결과입니다.

### 물리 베이스라인 적중률

| 방법 | R-Hit@1cm |
|---|---|
| 선형 외삽 `x(0) + 2v` | 57.88% |
| 가속도 보정 `x(0) + 2v + 0.6a` | **60.08%** |
| SG 스무딩 후 예측 | 51.32% (더 나쁨) |

β=0.6이 최적이며, 전처리/스무딩은 성능 저하 확인.

### 궤적 운동 특성

| 지표 | 평균 | 중간값 | p95 |
|---|---|---|---|
| 속도 (m/40ms) | 0.0256 | 0.0234 | 0.0538 |
| 가속도 크기 (m/40ms²) | 0.00552 | 0.00341 | 0.01825 |
| 궤적 직선성 (1.0=완전 직선) | 0.8945 | 0.9664 | - |

### 케이스 분류

| 케이스 | 수 | 비율 |
|---|---|---|
| 둘 다 HIT | 5,466 | 54.7% |
| 선형만 HIT | 322 | 3.2% |
| 가속도만 HIT | 542 | 5.4% |
| 둘 다 MISS | 3,670 | **36.7%** |

MISS 케이스는 급격한 방향 전환이 원인 — 속도(+20%), 가속도(+63%) 높음.

---

## 모델 구조

### 전체 파이프라인

```
입력: 11 시점 3D 좌표 (N, 11, 3)
        │
        ├── [Candidate Generation]
        │     Frenet 프레임 기반 35개 후보 생성 → (N, 35, 3)
        │
        ├── [Feature Extraction]
        │     seq_feat:  (N, 11, 9)  — 속도/가속도/곡률/jerk 등
        │     cand_feat: (N, 35, 10) — 후보별 Frenet 투영 피처
        │
        ├── [CandidateSelector — Transformer]
        │     TransformerEncoder (seq_feat)        → 시퀀스 컨텍스트
        │     Cross-Attention (candidates ← seq)  → 후보별 score
        │     → logits (N, 35)
        │     → Top-3 가중 평균 예측
        │
        └── [BoundaryMLP]
              오차 0.5~2.5cm 구간에 대해 잔차 보정 (최대 6mm)
              → 최종 예측 (N, 3)
```

### CandidateSelector 상세

| 컴포넌트 | 구성 |
|---|---|
| seq_proj | Linear(9 → 128) |
| PositionalEncoding | Embedding(11, 128) |
| TransformerEncoder | d_model=128, nhead=4, layers=3, norm_first=True |
| cand_proj | Linear(10 → 128) |
| cross_attn | MultiheadAttention(candidates → sequence) |
| head | Linear(128×2+10 → 128) → GELU → Linear(128 → 35) |

### 손실 함수

```
L = L_softCE + 0.25 × L_pairwise

L_softCE   : soft label cross-entropy (거리 기반 타겟 분포, temp=0.005)
L_pairwise : good candidate > bad candidate 랭킹 손실 (margin=0.12)
```

### 후보 생성 (Candidates)

Frenet 프레임으로 가속도를 분해하여 35개 물리적으로 타당한 후보 생성:

```
pred = p0 + d1 × t × v + par × t² × acc_par + perp × t² × acc_perp + jerk × t² × jerk
```

| 계열 | 수 | 설명 |
|---|---|---|
| Base | 1 | 순수 선형 외삽 |
| Acceleration | 4 | 가속도 보정 |
| Frenet | 6 | 접선/법선 방향 분리 |
| Turn | 8 | 방향 전환 케이스 |
| Jerk | 4 | 순간 가속도 변화 |
| Latency | 12 | 시스템 지연 보정 (0.85~1.15×) |

### 학습 전략

| 항목 | 설정 |
|---|---|
| K-Fold | 5-Fold (MD5 해시 기반 안정적 분할) |
| 데이터 증강 | SO3 랜덤 3D 회전 — 배치 단위로 GPU에서 수행 |
| 옵티마이저 | AdamW (lr=3e-4, weight_decay=1e-4) |
| 스케줄러 | CosineAnnealingLR |
| Early Stopping | patience=30 |
| Boundary MLP | OOF 예측으로만 학습 (데이터 누수 방지) |

### 원본(PB 0.6822) 대비 개선점

| 항목 | 원본 | 개선 |
|---|---|---|
| 셀렉터 | Attn-GRU | **Transformer + Cross-Attention** |
| 후보 수 | 28개 | **35개** |
| 학습 | 단일 모델 | **5-Fold 앙상블** |
| 증강 | 없음 | **SO3 3D 회전** |
| 손실 | CE | **Soft-label CE + Pairwise ranking** |
| Boundary | 전체 데이터 | **OOF만 사용** |

---

## 프로젝트 구조

```
.
├── data/                        # 대회 데이터 (레포 미포함, .gitignore)
│   ├── train/                   # 학습 샘플 CSV (10,000개)
│   ├── test/                    # 평가 샘플 CSV (10,000개)
│   ├── train_labels.csv
│   └── sample_submission.csv
├── dev/                         # 개발 작업 (dev 브랜치 전용)
│   ├── eda/                     # 탐색적 데이터 분석
│   │   ├── beta_search.py       # β 최적값 탐색 → β=0.6 확인
│   │   ├── noise_analysis.py    # SG 스무딩 효과 분석
│   │   └── weighted_velocity.py # 가중 속도 방식 비교
│   └── experiments/             # 모델 코드
│       ├── config.py            # 경로 및 하이퍼파라미터
│       ├── candidates.py        # Frenet 기반 후보 생성 + 피처 추출
│       ├── dataset.py           # MosquitoDataset (raw coords 반환) + augment_batch_gpu
│       ├── model.py             # CandidateSelector (Transformer)
│       ├── boundary.py          # BoundaryMLP (잔차 보정)
│       ├── train.py             # K-Fold 학습 루프
│       ├── predict.py           # 앙상블 추론 + 제출 파일 생성
│       └── outputs/             # 저장된 모델 가중치 (레포 미포함)
├── README.md
└── requirements.txt
```

---

## 환경 설정

```bash
conda create -n mosquito python=3.11 -y
conda activate mosquito
# CUDA 12.8 (RTX 5080 / Blackwell 이상)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas tqdm scikit-learn matplotlib
```

> CUDA 12.4 환경(Ampere/Ada 이하)은 `cu124`로 변경

Windows에서 OMP 중복 경고 발생 시:
```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

---

## 실행 방법

```bash
# 데이터 준비: data/ 폴더에 대회 데이터 압축 해제

# 학습 (5-Fold + Boundary MLP)
cd D:\Mosquito
python dev/experiments/train.py

# 추론 및 제출 파일 생성
python dev/experiments/predict.py
# → dev/experiments/outputs/submission.csv
```

---

## 브랜치 구조

| 브랜치 | 용도 |
|---|---|
| `main` | 문서 (README, requirements) |
| `dev` | 모델 개발 및 실험 전체 |

---

## 성능 기록

| 모델 | CV R-Hit | LB R-Hit |
|---|---|---|
| 물리 베이스라인 (β=0.6) | 60.08% | - |
| PB 참고 솔루션 | - | 68.22% |
| CandidateSelector (ours) | 진행 중 | - |

---

## 개발 로그

### 2026-05-21

**[환경 구성]**
- Anaconda `mosquito` 환경 생성 (Python 3.11)
- GPU: RTX 5080 (Blackwell, CUDA 13.2 드라이버) 확인
- PyTorch 2.11.0+cu128 설치 — Blackwell 아키텍처(sm_100) 대응 CUDA 12.8 빌드
- 패키지: numpy, pandas, tqdm, matplotlib, scikit-learn

**[GPU 연산 최적화]**
- 문제: `MosquitoDataset.__getitem__`에서 샘플마다 numpy feature 계산 → CPU 병목
  - 배치 256 기준 feature 계산: 140ms (샘플 1개씩 256회 순차 호출)
- 해결: feature 계산 전체를 GPU 배치 연산으로 이전
  - `make_candidates_gpu`, `make_seq_features_gpu`, `make_cand_features_gpu` 추가 (`candidates.py`)
  - `augment_batch_gpu` 추가 — SO3 회전을 배치 단위로 GPU에서 수행 (`dataset.py`)
  - `MosquitoDataset.__getitem__`은 raw coords만 반환, feature는 학습 루프에서 GPU 계산
  - `train.py`, `predict.py` 루프 업데이트
- 결과: feature 계산 140ms → 5.6ms (배치 256 기준, **25x 향상**)
- 수치 동일성: numpy 대비 최대 오차 7.6e-06 (float32 정밀도 수준)

### 2026-05-22

**[analyze.py 결과 — 현재 모델 진단]**
- Oracle R-Hit (35 candidates 상한선): **72.18%**
- Selector efficiency: **89.1%** (OOF 64.28% / Oracle 72.18%)
- Physics blend: α=1.0(모델 단독)이 최고 → physics blend 불필요
- Top-k: Top-3 최적 (Top-1: 0.6358 / Top-3: 0.6428 / Top-5: 0.6423)
- Boundary MLP: **-7.22%** (0.6428 → 0.5706) → 심각한 성능 저하, 즉시 제거

**[Boundary MLP 원인 분석 및 수정]**
- 문제: BOUNDARY_LO=0.5cm, BOUNDARY_HI=2.5cm → 55.6% 샘플을 보정 시도
  - 0.5~1.0cm 구간: 이미 hit인 샘플을 miss로 바꿀 위험
  - 1.0~2.5cm 구간: 최대 6mm 보정으로는 1cm 이하 달성 불가
- 수정: 범위를 BOUNDARY_LO=0.9cm, BOUNDARY_HI=1.3cm으로 축소 (재학습 필요)
- predict.py: boundary 없는 버전을 submission.csv 기본으로, boundary 버전은 submission_boundary.csv 별도 저장

**[후보군 확장 35 → 50개]**
- Oracle 72.18% 기준, 70% 목표 달성에는 후보군 확장이 필수
  - 기존 |perp| 최대 0.20 → 급격한 방향 전환 케이스 미커버
  - 기존 |jerk| 최대 0.15 → 강한 순간 가속도 미커버
- 추가: turn 계열 6개 (|perp| 0.30~0.60), jerk 계열 4개 (|jerk| 0.30~0.50), turn+jerk 복합 5개
- 재학습 후 oracle 상한선 상승 예상 (74~76% 목표)

**[TTA 구현 (Test-Time Augmentation)]**
- augment_batch_gpu_with_R 추가 (dataset.py) — rotation matrix R 반환
- analyze.py에 TTA×8 OOF 평가 섹션 추가 — 재학습 없이 selector efficiency 개선 측정
- 기대 효과: +0.5~1.5% (훈련 시 SO3 증강을 사용했으므로 회전 불변성 있음)

**[analyze.py 재실행 — 50개 후보 진단 결과]**

| 항목 | 35 candidates | 50 candidates |
|---|---|---|
| Oracle R-Hit | 72.18% | **74.89%** (+2.71pp) |
| Selector efficiency | 89.1% | 85.3%* |
| oracle↔selector 갭 | 7.9pp | 11.0pp* |
| Top-k 최적 | 3 | 5 |
| TTA×8 효과 | - | 없음 (±0.00%) |
| Best physics blend | α=1.0 | α=0.7 (+0.0001pp, 사실상 없음) |

\* 기존 모델이 35개 후보로 학습되어 새 15개 후보 평가 불안정 → 재학습 후 회복 예상

**[TTA 효과 없는 이유]**
- SO3 훈련 증강으로 모델이 회전 불변에 가까움 → rotate→predict→unrotate ≈ predict
- TTA는 회전 불변이 불완전할 때 효과 있음 → 현재 구조에서는 의미 없음

**[현 시점 목표 계산]**
- Oracle 74.89%, 재학습 후 selector efficiency 89% 회복 시: 74.89% × 89% ≈ **66.7%**
- selector efficiency 93% 달성 시: 74.89% × 93% ≈ **69.6%**
- 70% 달성 조건: efficiency ≥ 93.5% (74.89% × 93.5% = 70.0%)
- 다음 실험 방향: 50개 후보로 재학습 → CV 확인 → selector 개선 (증강/손실/구조)
