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
        │     Frenet 프레임 기반 50개 후보 생성 → (N, 50, 3)
        │
        ├── [Feature Extraction]
        │     seq_feat:  (N, 11, 11) — 속도/가속도/곡률/jerk/jerk_abs/acc_cos
        │     cand_feat: (N, 50, 10) — 후보별 Frenet 투영 피처
        │
        └── [CandidateSelector — Transformer]
              TransformerEncoder (seq_feat)        → 시퀀스 컨텍스트
              Cross-Attention (candidates ← seq)  → 후보별 score
              → logits (N, 50)
              → Top-10 가중 평균 예측 → 최종 예측 (N, 3)
```

### CandidateSelector 상세

| 컴포넌트 | 구성 |
|---|---|
| seq_proj | Linear(11 → 128) |
| PositionalEncoding | Embedding(11, 128) |
| TransformerEncoder | d_model=128, nhead=4, layers=3, norm_first=True |
| cand_proj | Linear(10 → 128) |
| cross_attn | MultiheadAttention(candidates → sequence) |
| head | Linear(128×2+10 → 128) → GELU → Linear(128 → 1) per candidate |

### 손실 함수

```
L = L_softCE + 0.25 × L_pairwise + 0.05 × L_listMLE

L_softCE   : soft label cross-entropy (거리 기반 타겟 분포, temp=0.005)
L_pairwise : good candidate > bad candidate 랭킹 손실 (margin=0.12)
L_listMLE  : -log P(oracle ranked #1) — oracle 후보 직접 최적화
```

### 후보 생성 (Candidates)

Frenet 프레임으로 가속도와 jerk를 분해하여 50개 물리적으로 타당한 후보 생성:

```
pred = p0 + d1·t·v + par·t²·acc_par + perp·t²·acc_perp + jerk·t²·jerk_vec
```

| 계열 | 수 | 설명 |
|---|---|---|
| Base | 1 | 순수 선형 외삽 |
| Acceleration | 4 | 가속도 보정 |
| Frenet | 11 | 접선/법선 방향 세밀 분해 |
| Turn | 13 | 방향 전환 (|perp| 0.20~0.60) + 복합 |
| Jerk | 6 | 순간 가속도 변화 (|jerk| 0.08~0.50) |
| Latency | 15 | 시스템 지연 보정 (0.85~1.15×) + turn 복합 |

### 학습 전략

| 항목 | 설정 |
|---|---|
| K-Fold | 5-Fold (MD5 해시 기반 안정적 분할) |
| 데이터 증강 | **yaw + speed-scale** (z축 회전 + p0 기준 이동량 스케일, z=UP 보존) |
| 옵티마이저 | AdamW (lr=3e-4, weight_decay=1e-4) |
| 스케줄러 | CosineAnnealingLR (T_max=EPOCHS, epoch 단위 step) |
| Early Stopping | patience=40 |
| Boundary MLP | **완전 제거** (OOF −7.9pp, 오차 벡터 방향 학습 불가) |

### 원본(PB 0.6822) 대비 개선점

| 항목 | 원본 | 현재 |
|---|---|---|
| 셀렉터 | Attn-GRU | **Transformer + Cross-Attention** |
| 후보 수 | 28개 | **50개** |
| Seq features | — | **11개** (jerk_abs, acc_cos 추가) |
| 학습 | 단일 모델 | **5-Fold 앙상블, multi-seed 지원** |
| 증강 | 없음 | **yaw + speed-scale** (z=UP 보존 + 속도 스케일 일반화) |
| 손실 | CE | **Soft-CE + Pairwise + ListMLE** |
| Top-k 예측 | Top-1 (argmax) | **Top-10 가중 평균** |
| Boundary | 전체 데이터 적용 | **완전 제거** (−7.9pp 확인) |

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
│       ├── dataset.py           # MosquitoDataset + augment_batch_gpu / augment_speed_scale_gpu
│       ├── model.py             # CandidateSelector (Transformer)
│       ├── boundary.py          # BoundaryMLP (잔차 보정)
│       ├── train.py             # K-Fold 학습 루프
│       ├── predict.py           # 앙상블 추론 + 제출 파일 생성
│       └── outputs/             # 저장된 모델 가중치 (레포 미포함)
│           ├── seed42/          #   seed별 서브디렉토리 (train.py --seed 42)
│           ├── seed123/
│           └── seed777/
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

# ── 단일 seed 학습 (기본: seed=42) ──────────────────────────────────────────
cd D:\Mosquito
python dev/experiments/train.py              # seed=42 (config.SEED 기본값)
python dev/experiments/train.py --seed 123   # 다른 seed
# → dev/experiments/outputs/seed{N}/selector_fold{0..4}.pt

# ── 진단 (OOF 분석 — 학습 후 실행) ──────────────────────────────────────────
python dev/experiments/analyze.py            # seed=42 기준
python dev/experiments/analyze.py --seed 123

# ── 추론 및 제출 파일 생성 ────────────────────────────────────────────────────
python dev/experiments/predict.py            # seed=42 단일
python dev/experiments/predict.py --seed 123
python dev/experiments/predict.py --seeds 42 123 777  # 3-seed ensemble (15 models)
# → dev/experiments/outputs/submission.csv

# ── 3-seed 앙상블 전체 학습 ──────────────────────────────────────────────────
python dev/experiments/train.py --seed 42
python dev/experiments/train.py --seed 123
python dev/experiments/train.py --seed 777
python dev/experiments/predict.py --seeds 42 123 777
```

---

## 현재 파이프라인 (Current Default)

| 항목 | 현재 설정 |
|---|---|
| Candidates | **50개** (Frenet + turn + jerk — 60-cand 실험 후 efficiency 역행으로 복귀) |
| Augmentation | **yaw-only** (speed-scale 0.85~1.15 실험 결과 효과 없음 → 복귀) |
| Selector | Transformer + Cross-Attention (d_model=128, 3 layers) |
| Seq features | **11개** (jerk_abs, acc_cos — SEQ_DIM 9→11) |
| Loss | **CE + PW×0.25 + LML×0.05** (그리드 탐색 A~D 완료, 0.05 최적 확정) |
| Prediction | **Top-10** weighted average, temp=1.0 (config.TOPK로 통일) |
| Boundary MLP | **완전 제거** (OOF -7.93pp, 구조적 한계 확인) |
| TTA | **완전 제거** (효과 없음, yaw 불변 모델) |
| CV R-Hit | **64.89%** (yaw-only, LML=0.05 — current best) |
| Oracle ceiling | **74.89%** (50-cand 기준) |
| Selector efficiency | **86.3%** (목표: 93.5% → 70% 달성 조건) |
| Multi-seed | train/analyze/predict 모두 `--seed` / `--seeds` CLI 지원 |

---

## 브랜치 구조

| 브랜치 | 용도 |
|---|---|
| `main` | 문서 (README, requirements) |
| `dev` | 모델 개발 및 실험 전체 |

---

## 성능 기록

| 모델 | N-cand | CV R-Hit | Oracle | Efficiency | OOF | 비고 |
|---|---:|---:|---:|---:|---:|---|
| Linear extrapolation | - | 57.88% | - | - | - | p₀ + 2v |
| Acceleration β=0.6 | - | 60.08% | - | - | - | 물리 베이스라인 최고 |
| PB 참고 솔루션 | - | - | - | - | - | LB 68.22% |
| 35-cand selector (SO3, Top-3) | 35 | 64.28% | 72.18% | 89.1% | 64.28% | 베이스라인 |
| 35-cand + Boundary | 35 | - | - | - | 57.06% | ❌ -7.22pp |
| 50-cand (SO3) | 50 | 63.65% | 74.89% | 85.0% | - | |
| 50-cand (yaw) | 50 | 64.13% | 74.89% | 85.6% | - | SO3→yaw +4.8pp |
| 50-cand (yaw), SOFT_TEMP=0.003 | 50 | 63.44% | 74.89% | 84.7% | - | ❌ -0.57pp |
| 50-cand (yaw), PW=0.5 | 50 | 62.15% | 74.89% | - | - | ❌ fold5 붕괴 |
| 50-cand (yaw), SEQ11+LML=0.5 | 50 | 64.61% | 74.89% | 86.3% | 64.61% | oracle rank 22→7.8 |
| 60-cand (yaw), SEQ11+LML=0.5 | 60 | 63.75% | **78.31%** | 81.4% | 63.75% | ❌ efficiency 역행 |
| **실험 A**: 50-cand, LML=0.0 | 50 | **64.66%** | 74.89% | **86.3%** | **64.66%** | clean baseline ★ |
| 실험 B: LML=0.05 | 50 | **64.89%** | 74.89% | **86.6%** | **64.89%** | oracle rank mean 13.1 — **현재 best** |
| 실험 C: LML=0.10 | 50 | 64.80% | 74.89% | 86.5% | 64.80% | oracle rank mean 10.0 |
| 실험 D: LML=0.20 | 50 | 64.65% | 74.89% | 86.3% | 64.65% | ❌ std↑ fold5↓, 불안정 |
| 실험 E: yaw+speed-scale 0.85~1.15 | 50 | 64.65% | 74.89% | 86.3% | 64.65% | ❌ -0.24pp, 효과 없음 → yaw 복귀 |
| 실험 F: speed-scale 0.80~1.20 | 50 | — | — | — | — | ❌ E 실패로 건너뜀 |
| **실험 G**: Smart 50-cand (후보 교체) | 50\* | — | — | — | — | 진행 예정 |

### Selector Error Decomposition

| 실험 | Oracle rank mean | Oracle in Top-5 | A그룹 | B그룹 | C그룹 |
|---|---:|---:|---:|---:|---:|
| 50-cand, SEQ9, CE+PW | 22.7 | 26.4% | 18.5% | 56.4% | 25.1% |
| 50-cand, SEQ11, CE+PW+LML=0.5 | **7.8** | — | **34.7%** | 40.2% | 25.1% |
| 60-cand, SEQ11, CE+PW+LML=0.5 | 12.0 | — | 27.1% | 51.2% | 21.7% |
| **실험 A**: 50-cand, LML=0.0 | 21.9 | 28.3% | 20.0% | **54.9%** | 25.1% |
| 실험 B: LML=0.05 | **13.1** | **47.1%** | **33.4%** | 41.5% | 25.1% |
| 실험 C: LML=0.10 | **10.0** | **51.8%** | **36.0%** | 38.9% | 25.1% |
| 실험 D: LML=0.20 | — | — | — | — | — | ❌ 불안정 |
| 실험 E: yaw+speed-scale 0.85~1.15 | 13.0 | 46.7% | 33.2% | 41.7% | 25.1% | ❌ 변화 없음 |

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

**[50개 후보 재학습 결과]**

| | 35 candidates | 50 candidates |
|---|---|---|
| CV mean R-Hit | **0.6429** | 0.6410 (-0.19pp) |
| Oracle | 72.18% | **74.89%** (+2.71pp) |
| Selector efficiency | **89.1%** | 85.6% (-3.5pp) |
| Boundary samples | 55.6% | **16.6%** (범위 수정 효과 확인) |
| Boundary 적용 후 OOF | 0.5706 | 0.5665 (여전히 심각) |

**결론**: oracle은 상승했으나 selector efficiency 하락이 상쇄 → 순 -0.19pp
- 원인: d_model=128, 3-layer Transformer의 용량이 50개 후보 선별에 부족
- 35개 → 50개로 후보 늘어나면서 "비슷해 보이는 나쁜 후보"도 늘어남 → 더 강한 판별력 필요
- Boundary MLP: 범위 수정(55.6%→16.6%) 후에도 -7.45% → 피처/구조 자체가 한계

**Boundary MLP 평가 종료**
- 피처 12개 / hidden 64 구조로는 올바른 보정 방향 학습 불가
- OOF 정확한 보정 방향 학습 어려움 (오차 벡터 방향이 랜덤에 가까움)
- 이후 실험에서 제외, predict.py 기본값 유지 (boundary 미적용)

**[다음 실험: 모델 용량 확대]**
- 목표: selector efficiency 85.6% → 93%+ (50개 후보 환경에서)
- 변경: d_model 128→256, num_layers 3→4
- 기대: 74.89% × 93% ≈ 69.6% → 70% 달성 근접
- 추가 고려: patience 30→50 (더 충분한 학습 기회)

---

### 2026-05-22 (2)

**[코드 품질 개선 — 5개 이슈 검토 및 수정]**

**이슈 1: scheduler.step() 위치 → 이미 올바름 (수정 불필요)**
- train.py line 83: indent 8칸 (epoch loop 안, batch loop 밖)
- `for batch in train_loader:` (line 61, indent 8칸), `optimizer.step()` (line 81, indent 12칸)
- `scheduler.step()` (line 83, indent 8칸) → epoch당 1회 호출, CosineAnnealingLR T_max=EPOCHS와 정상 동기

**이슈 2: SO3 증강 → yaw-only 증강으로 변경**
- x=forward, y=left, **z=UP** 좌표계: z가 중력 반대 방향 의미 보유
- SO3 전체 회전은 z=UP 구조를 파괴 → 물리적으로 불가능한 뒤집힌 궤적 생성 가능
- dataset.py: `augment_batch_gpu_yaw()`, `augment_batch_gpu_yaw_with_R()` 추가
- config.py: `AUG_MODE = 'yaw'` (옵션: 'so3' | 'yaw' | 'none')
- train.py: AUG_MODE 설정값에 따라 증강 방식 선택

**이슈 3: Top-k 비교에 top-all 추가**
- analyze.py: k=[1, 2, 3, 5, N_CANDIDATES(50)] 비교 → weighted average vs argmax 효과 측정
- Top-k가 top-1보다 높으면 가중 평균이 유효, 낮으면 단일 후보 선택이 유리

**이슈 4: Oracle 상세 리포트 추가**
- analyze.py: `candidate_oracle_report()` 함수 추가
- 출력: oracle R-Hit, best dist 평균, 백분위수(p50/p75/p90/p95), top-10 oracle 후보 인덱스
- 어떤 후보 파라미터가 실제 정답과 가장 가까운지 확인 가능

**이슈 5: Boundary MLP 이중 저장 → 이미 구현됨**
- predict.py: `submission.csv` (boundary 미적용, 기본), `submission_boundary.csv` (비교용)
- 이미 이전 세션에서 구현 완료

**[모델 설정 변경 (config.py)]**

| 설정 | 이전 | 현재 |
|---|---|---|
| D_MODEL | 128 | **256** |
| NHEAD | 4 | **8** |
| NUM_LAYERS | 3 | **4** |
| PATIENCE | 30 | **50** |
| AUG_MODE | SO3 (하드코딩) | **'yaw'** (설정 가능) |

- 파라미터 수: ~0.8M → ~3.2M (4배 증가)
- yaw 증강으로 z=UP 좌표 보존
- patience 50으로 대형 모델 충분한 학습 기회 부여

**[다음 실험: 재학습 후 확인 사항]**
1. CV R-Hit ≥ 0.67 목표 (현 0.6410 대비 +3pp)
2. selector efficiency ≥ 90% 목표 (현 85.6% 대비)
3. analyze.py로 top-1 vs top-k 비교 — 가중 평균 유효성 확인
4. oracle report로 어떤 후보 파라미터가 주도적인지 확인

**[50개 후보 학습 결과 — d_model=128, SO3 aug]**

| | d_model=128, SO3 (35) | d_model=128, SO3 (50) |
|---|---:|---:|
| Fold 1 | - | 0.6396 |
| Fold 2 | - | 0.6356 |
| Fold 3 | - | 0.6419 |
| Fold 4 | - | 0.6391 |
| Fold 5 | - | 0.6265 |
| CV mean | **0.6429** | 0.6365 |
| Selector efficiency | 89.1% | 85.0% |

**결론**: 후보 35→50 확장 후 CV -6.4pp 하락 (0.6429→0.6365)
- Fold 5 = 0.6265 이상치 — 특정 fold 데이터에 취약할 수 있음
- selector efficiency 89.1%→85.0%: 후보 50개 환경에서 판별력 부족
- 다음: yaw augmentation 변수 분리 (d128+yaw vs d128+SO3 비교)

**[변수 분리 실험 결과 — d_model=128, yaw aug, patience=40]**

| | SO3 (35-cand) | SO3 (50-cand) | yaw (50-cand) |
|---|---:|---:|---:|
| Fold 1 | - | 0.6396 | **0.6515** |
| Fold 2 | - | 0.6356 | 0.6385 |
| Fold 3 | - | 0.6419 | 0.6393 |
| Fold 4 | - | 0.6391 | 0.6406 |
| Fold 5 | - | 0.6265 | 0.6365 |
| CV mean | **0.6429** | 0.6365 | **0.6413** (+4.8pp) |
| Selector efficiency | 89.1% | 85.0% | 85.6% |

**결론: yaw augmentation 유효 확인 (+4.8pp)**
- SO3는 z=UP 구조를 파괴하여 물리적으로 불가능한 궤적(뒤집힌 비행)을 학습에 주입
- yaw-only는 수평면 회전만 적용 → 센서 시야각 변화 반영, z=UP 보존
- 35-cand 베이스라인(0.6429) 대비 -1.6pp: 후보 확장 비용 대부분 회복

**이상 패턴**
- Fold 1: 0.6515 — 다른 fold 대비 +1pp 이상 높음, 데이터 분포 차이 가능성
- Fold 5: 여전히 최저 (0.6365) — 일관된 약 fold, 어려운 케이스 집중됐을 가능성

**다음 단계**
1. analyze.py 실행 → oracle top-k in, selector error decomposition
2. soft-label temperature 재탐색 (0.003 / 0.005 / 0.007 / 0.010)
3. pairwise loss weight 재탐색 (0.0 / 0.1 / 0.25 / 0.5)
4. top-k 재탐색 (1 / 3 / 5 / 7) — analyze.py가 자동 출력

---

### 2026-05-22 (3)

**[실험 로드맵 확정 및 analyze.py 진단 강화]**

**방향 검증 (이전 실험 결과 기반)**
- Boundary MLP 제거: ✅ (0.6428 → 0.5706, 포폴용 "실패한 후처리 분석 후 제거" 사례)
- 후보 확장 35→50: ✅ oracle +2.71pp (72.18%→74.89%), 단 selector 재학습 필요
- TTA 제거: ✅ (SO3 훈련으로 회전 불변, 추론 시간만 낭비)

**실험 우선순위**

| 순위 | 내용 | 현재 상태 |
|---|---|---|
| 1 | 50개 후보 완전 재학습 (d_model=256, yaw aug) | 진행 중 |
| 2 | soft-label temperature 재탐색 (0.003/0.005/0.007/0.010) | 대기 |
| 3 | pairwise loss weight 재탐색 (0.0/0.1/0.25/0.5) | 대기 |
| 4 | Top-k 재탐색 (1/3/5/7) — 재학습 후 analyze.py 자동 확인 | 대기 |
| 5 | selector error decomposition 분석 | analyze.py에 구현 완료 |

**현실적 목표**

| 단계 | 기대 CV |
|---|---|
| 50개 후보 재학습 성공 | 66~67% |
| loss/temp/top-k 튜닝 | 67~68.5% |
| selector error 분석 후 개선 | ~69% |
| efficiency ≥ 93.5% 달성 시 | 70% (74.89% × 93.5%) |

**analyze.py 섹션 4 추가: SELECTOR ERROR DECOMPOSITION**
- `oracle_selector_decomposition()` 함수 구현
- 3개 그룹으로 분류:
  - A (oracle hit ∩ top-5 포함): 가중 평균이 성능을 깎는지 확인 (Top-1 vs Top-5 비교)
  - B (oracle hit ∩ top-5 미포함): selector 학습/ranking 문제
  - C (oracle miss): 후보군 한계 → 후보 확장으로만 해결 가능
- Oracle candidate rank 분포 (mean/median/p75/p90) 출력
- 해석 가이드 자동 출력 (B 비중, Top-1 vs Top-5 우세 여부)

---

### 2026-05-22 (4)

**[50개 후보 재학습 최종 진단 — d128+yaw 기준]**

| | 35 candidates | 50 candidates |
|---|---:|---:|
| Oracle R-Hit | 72.18% | **74.89%** (+2.71pp) |
| OOF R-Hit | **64.28%** | 64.01% (-0.27pp) |
| Selector efficiency | **89.1%** | 85.5% (-3.6pp) |
| Boundary 적용 후 | 57.06% | 56.08% |

**결론: 50개 후보 확장은 실패에 가까움**
- Oracle은 +2.71pp 상승했으나, selector efficiency 89.1% → 85.5%로 하락이 상쇄
- 원인: 후보가 늘어날수록 "비슷해 보이는 나쁜 후보"도 증가 → classification 난이도 상승
- d128 Transformer 용량으로는 50개 후보 환경에서 89%+ efficiency 달성 불가
- d256으로 증가 시 오히려 CV 하락 (0.6413 → 0.6365): 10,000 샘플에서 과적합

**Boundary MLP 완전 기각 및 제거**
- 0.9~1.3cm 범위(17.1% 샘플)로 좁혔음에도 OOF -7.93pp (0.6401 → 0.5608)
- 오차 벡터 방향이 랜덤에 가까워 학습 자체가 불가능한 구조적 한계
- `train.py`에서 Boundary 학습/저장 블록 완전 삭제
- `analyze.py`에서 Boundary 섹션 완전 삭제

**[코드 리팩터링]**

| 파일 | 변경 내용 |
|---|---|
| `config.py` | `SOFT_TEMP=0.005`, `PAIRWISE_WEIGHT=0.25` 파라미터 추출 |
| `model.py` | `soft_labels()` temperature → config `SOFT_TEMP` 사용 |
| `train.py` | `PAIRWISE_WEIGHT` config 연동, Boundary 블록 삭제 |
| `analyze.py` | oracle 후보 기여도 테이블, Top-k 확장(7·10), 70% 달성 조건 자동 출력, Boundary 섹션 삭제 |

---

### 2026-05-22 (5)

**[analyze.py 전체 진단 — 50개 후보 현재 모델]**

**Oracle 후보 기여도 Top 15**

| idx | name | count | % |
|---:|---|---:|---:|
| 43 | jerk_xl_pos | 1212 | 12.1% |
| 21 | latency_s085 | 1184 | 11.8% |
| 44 | jerk_xl_neg | 1069 | 10.7% |
| 40 | turn_n060 | 852 | 8.5% |
| 39 | turn_p060 | 688 | 6.9% |
| 24 | latency_l115 | 621 | 6.2% |
| 4 | acc_2d1_060 | 534 | 5.3% |
| 49 | turn_fast_n030 | 393 | 3.9% |
| 38 | turn_n045 | 340 | 3.4% |
| 25 | latency_l110_turn | 270 | 2.7% |
| 26 | latency_s090_turn | 256 | 2.6% |
| 0 | p0_2d1 | 222 | 2.2% |
| 23 | latency_l108 | 216 | 2.2% |
| 17 | frenet_fast_p120_n020 | 163 | 1.6% |
| 22 | latency_s092 | 156 | 1.6% |

- 미사용 후보 0개 — **50개 전부 최소 1번은 oracle-best** → 후보 자르기 불가
- `jerk_xl_pos` + `jerk_xl_neg` 단 2개가 전체 22.8% 담당 — 50개 확장의 핵심 기여

**Top-k 비교 (OOF)**

| k | R-Hit |
|---:|---:|
| 1 | 0.6287 |
| 2 | 0.6367 |
| 3 | 0.6401 |
| 5 | 0.6399 |
| 7 | 0.6425 |
| **10** | **0.6431** |
| all(50) | 0.6276 |

→ **Top-10이 최적.** k가 클수록 좋아지는 것 자체가 oracle rank가 높다는 방증.

**Selector Error Decomposition**

| 그룹 | 샘플 | 비율 | 의미 |
|---|---:|---:|---|
| A: oracle ∩ top-5 | 1,847 | 18.5% | selector 성공 |
| **B: oracle ∩ top-5 밖** | **5,642** | **56.4%** | ← 핵심 실패 |
| C: oracle 자체 없음 | 2,511 | 25.1% | 후보군 한계 |

```
Oracle candidate rank:  mean=22.7  median=24  p75=39  p90=46
Oracle in Top-1: 11.4%  Top-3: 20.5%  Top-5: 26.4%  Top-7: 30.7%

랜덤 기댓값 (50개 균등): Top-1=2.0%  Top-5=10.0%
현재: Top-5=26.4% → 랜덤 대비 2.6배 — 여전히 심각하게 낮음
```

**결론: 후보 수가 아닌 랭킹 학습이 문제**
- oracle이 존재하는 7,489샘플 중 **75.3%에서 selector가 top-5 밖으로 밀어냄**
- `jerk_xl` 계열처럼 극단 파라미터 후보를 모델이 "비정상"으로 판단하는 것으로 추정
- SOFT_TEMP=0.005(기존)가 너무 넓어 여러 후보에 확률 질량 분산 → oracle 후보에 집중 못 함
- 후보 pruning 시 oracle 직접 하락 → 확장한 50개 유지하면서 selector 개선이 올바른 방향

**[실험: SOFT_TEMP 0.005 → 0.003]**

```
가설: sharper soft label → oracle 후보에 집중된 gradient → oracle rank 개선
```

- 기존 `SOFT_TEMP=0.005`: dist=0.5cm 후보와 dist=1.5cm 후보 간 label 차이 작음
- `SOFT_TEMP=0.003`: 거리 차이가 label에 더 가파르게 반영 → 모델이 정답을 더 명확히 구분

변경: `config.py` `SOFT_TEMP = 0.003`

**결과: 실패 (-0.57pp)**

| | SOFT_TEMP=0.005 | SOFT_TEMP=0.003 |
|---|---:|---:|
| CV mean | **0.6401** | 0.6344 |
| OOF | **0.6401** | 0.6345 |
| Efficiency | **85.5%** | 84.7% |

원인: 50개 후보 환경에서 1cm 이내 유효 후보 다수 공존 → one-hot에 가까운 label이 gradient 신호를 희소하게 만들어 학습 불안정. 기존 0.005가 최적.

**[다음 실험: PAIRWISE_WEIGHT 0.25 → 0.5]**

- temp 방향 막힘 → ranking loss 강화로 oracle rank 22.7 직접 공략
- `SOFT_TEMP=0.005` 복귀, `PAIRWISE_WEIGHT=0.5`
- 가설: pairwise weight 2배 → good/bad 마진 압력 증가 → oracle 후보 상위 랭킹 개선
- 성공 기준: analyze.py oracle rank mean < 20

---

### 2026-05-22 (6)

**[PAIRWISE_WEIGHT=0.5 실패 — CV 붕괴]**

| | PAIRWISE_WEIGHT=0.25 | PAIRWISE_WEIGHT=0.5 |
|---|---:|---:|
| Fold 1 | ~0.64 | 0.6397 |
| Fold 2 | ~0.64 | 0.6397 |
| Fold 3 | ~0.64 | 0.6397 |
| Fold 4 | ~0.64 | 0.6397 |
| Fold 5 | ~0.64 | 0.5974 |
| CV mean | **0.6401** | 0.6215 (−1.86pp) |

- Fold 5 = 0.5974 이상치 — pairwise loss 과도 시 특정 fold 데이터에서 gradient 충돌
- pairwise ranking 신호가 CE loss를 압도 → 전체 확률 질량이 극단으로 쏠림
- 결론: **PAIRWISE_WEIGHT=0.25 유지**

**[seq feature 확장 — SEQ_DIM 9→11]**

기존 9개 피처에 2개 추가:

| 피처 | 설명 | 기대 효과 |
|---|---|---|
| `jerk_abs` | 절대 jerk 크기 (acc 변화율) | jerk_xl 계열 후보 필요 케이스 감지 |
| `acc_cos` | 연속 acc 벡터 간 코사인 유사도 | 급격한 방향 전환 감지 |

- `candidates.py`: `make_seq_features`, `make_seq_features_gpu` t≥3 분기에 추가
- `model.py`: `SEQ_DIM 9→11`, `seq_proj: Linear(9→128) → Linear(11→128)`
- 피처 추가 비용 미미 (연산량 동일), 모델 파라미터 수 변화 없음

**[ListMLE loss 추가]**

oracle 후보를 직접 top-1로 올리는 ranking loss 추가:

```python
def listmle_loss(logits, cands, true):
    dist = torch.norm(cands - true.unsqueeze(1), dim=-1)
    oracle_idx = dist.argmin(dim=-1)
    return -F.log_softmax(logits, dim=-1).gather(1, oracle_idx.unsqueeze(1)).mean()

loss = loss_ce + 0.25 × loss_pair + 0.5 × loss_lml
```

- `LISTMLE_WEIGHT = 0.5` (config.py에 추가)
- 가설: oracle 후보 log-probability 직접 최대화 → oracle rank 개선

**[재학습 결과 — d128 + yaw + SEQ11 + ListMLE (현재 최고)]**

| | 이전 (SEQ9, 손실CE+PW) | 현재 (SEQ11, CE+PW+ListMLE) |
|---|---:|---:|
| CV mean R-Hit | 0.6401 | **0.6461** (+6.0pp) |
| OOF R-Hit | 0.6401 | **0.6461** |
| Oracle rank mean | 22.7 | 7.8 (−14.9) |
| Oracle rank median | 24 | 4 (−20) |
| Selector efficiency | 85.5% | **86.3%** (+0.8pp) |

- Oracle rank가 극적으로 개선 (mean 22.7→7.8, median 24→4)
- jerk_abs/acc_cos 피처 추가 + ListMLE 복합 효과
- Top-10 OOF 기준 **0.6461 = 현 최고** (이전 0.6431)

**Selector Error Decomposition (재학습 후)**

| 그룹 | 비율 | Top-1 hit | Top-5 hit |
|---|---:|---:|---:|
| A: oracle ∩ top-5 | 34.7% | 0.8614 | 0.8505 |
| B: oracle ∩ top-5 밖 | 40.2% | 0.0001 | 0.1030 |
| C: oracle 없음 | 25.1% | 0.0 | 0.0 |

- A그룹 비중 18.5% → 34.7% (oracle을 top-5 안으로 끌어오는 데 성공)
- B그룹 비중 56.4% → 40.2% (여전히 최대 실패 원인)
- C그룹 25.1% 고정 — 후보 공간 자체의 한계

**[Prediction Temperature 탐색]**

| temp | R-Hit (Top-10 OOF) |
|---:|---:|
| 0.3 | 0.6412 |
| 0.5 | 0.6442 |
| 0.7 | 0.6453 |
| **1.0** | **0.6461** |
| 1.5 | 0.6452 |
| 2.0 | 0.6443 |

→ `temp=1.0`이 최적 (소프트맥스를 추가로 sharp/soft하게 조절해도 이득 없음)

**[C그룹 분석 기반 후보 공간 확장 준비]**

C그룹(25.1%, 약 2,511샘플)은 현재 후보 50개 중 어떤 것도 1cm 이내에 없어 selector 개선만으로는 해결 불가.

- `analyze.py`에 `c_group_analysis()` 함수 추가
  - C그룹 샘플의 Frenet 파라미터(par/perp/jerk) 분포 계산
  - 현재 후보 커버리지 경계와 비교 → 어느 방향/범위가 미커버인지 파악
  - C그룹에 가장 가까운 기존 후보 top-10 표시
- `analyze()` 본체의 섹션 4 직후에 섹션 4b로 연결 완료

---

### 2026-05-22 (7)

**[C그룹 분석 결과 — analyze.py 4b 섹션 출력]**

```
C그룹: 2511개  (25.1%)
nearest-cand dist  : mean=2.70cm  p50=1.72  p75=3.23  p90=5.81  p95=7.79

True label (Frenet, speed×2 정규화):
par  : mean=0.77  std=0.66  p5=-0.35  p95=1.29
perp : mean=0.15  std=0.59  |p5|=0.02  |p75|=0.45  |p95|=1.02
jerk : mean=0.48  p75=0.55  p95=1.70

현재 후보 커버리지: par=[0, 1.20]  |perp|≤0.60  |jerk|≤0.50
C그룹 중 |perp| 초과: 15.5%
C그룹 중 par 범위 밖: 20.4%
```

**C그룹 nearest 후보 분포**

| 후보 | C그룹 nearest 비율 | 의미 |
|---|---:|---|
| latency_s085 | 21.8% | 시간 보정 필요하나 frenet도 맞지 않음 |
| jerk_xl_pos + jerk_xl_neg | 27.0% | jerk 더 강한 후보 필요 (max 0.50, 필요 1.70) |
| turn_p060 + turn_n060 | 23.5% | perp 더 큰 후보 필요 (max 0.60, 필요 1.02) |

**핵심 갭**

| 파라미터 | 현재 커버리지 | C그룹 p75 | C그룹 p95 | 초과 비율 |
|---|---:|---:|---:|---:|
| `\|jerk\|` | 0.50 | 0.55 | **1.70** | — |
| `\|perp\|` | 0.60 | 0.45 | **1.02** | **15.5%** |
| `par` | [0, 1.20] | — | 1.29 | **20.4%** |

**[candidates.py 수정 — 50 → 60개]**

C그룹 분석 기반으로 10개 후보 추가:

| 계열 | 이름 | jerk | perp | 근거 |
|---|---|---:|---:|---|
| Jerk 확장 | `jerk_xxl_pos/neg` | ±0.80 | 0 | C그룹 nearest 27%, jerk gap |
| | `jerk_xxxl_pos/neg` | ±1.20 | 0 | C그룹 jerk p75=0.55 |
| | `jerk_extreme_pos/neg` | ±1.80 | 0 | C그룹 jerk p95=1.70 |
| Perp 확장 | `turn_p080/n080` | 0 | ±0.80 | C그룹 nearest 23.5%, |perp| gap |
| | `turn_p100/n100` | 0 | ±1.00 | C그룹 \|perp\| p95=1.02 |

- d1 / par 값은 jerk·perp 크기에 따라 단계적으로 감소 (물리적 일관성 유지)
- **50 → 60개, oracle ceiling 78~80% 목표** (현재 74.89%)
- 재학습 후 analyze.py로 C그룹 비율 및 oracle 변화 확인 예정

**60-cand 재학습 결과 및 결론**

| 항목 | 50-cand (LML=0.5) | 60-cand (LML=0.5) |
|---|---:|---:|
| CV R-Hit | 64.61% | 63.75% |
| Oracle R-Hit | 74.89% | **78.31%** (+3.42pp) |
| Selector efficiency | 86.3% | 81.4% (−4.9pp) |
| A그룹 | 34.7% | 27.1% |
| B그룹 | 40.2% | 51.2% |
| Oracle rank mean | 7.8 | 12.0 |

결론: oracle ceiling은 올랐으나 efficiency 하락이 상쇄 → OOF −0.86pp. 후보 추가보다 ranking 개선이 우선.

---

### 2026-05-22 (8)

**[코드 정비 — clean baseline 준비]**

| 파일 | 변경 내용 |
|---|---|
| `config.py` | `SOFT_TEMP` 0.003→0.005 복귀, `LISTMLE_WEIGHT` 0.5→0.0, `TOPK=10` 추가 |
| `candidates.py` | 60→50 복귀 (jerk_xxl~turn_n100 10개 제거) |
| `train.py` | `topk=10` 하드코딩 → `topk=TOPK`, ListMLE 조건부 실행 |
| `predict.py` | `selector_predict(topk=TOPK)` 통일 (기존 default topk=3 버그 수정) |

**핵심 발견: SEQ11이 ListMLE보다 기여가 크다**
- 이전 "SEQ11+LML=0.5 → 0.6461" 실험 당시 SOFT_TEMP=0.003 상태였을 가능성
- clean baseline (SEQ11, LML=0.0, SOFT_TEMP=0.005) 결과: **OOF 0.6466**
- LML=0.5 결과(0.6461)를 오히려 소폭 상회 → SEQ11 피처 자체가 핵심 기여

**[실험 A — clean baseline 결과]**

설정: `CANDIDATES=50, SOFT_TEMP=0.005, PAIRWISE_WEIGHT=0.25, LISTMLE_WEIGHT=0.0, TOPK=10, AUG=yaw`

| Fold | R-Hit |
|---:|---:|
| 1 | 0.6540 |
| 2 | 0.6439 |
| 3 | 0.6481 |
| 4 | 0.6490 |
| 5 | 0.6381 |
| **CV mean** | **0.6466 ± 0.0053** |

| 지표 | 값 |
|---|---|
| OOF R-Hit (Top-10) | **0.6466** |
| Oracle R-Hit | 0.7489 |
| Selector efficiency | **86.3%** |
| Oracle rank mean / median | 21.9 / 22 |
| Oracle rank p75 / p90 | 39 / 46 |
| Oracle in Top-1 / Top-3 / Top-5 / Top-7 | 11.1% / 21.2% / 28.3% / 33.1% |
| A그룹 (oracle ∩ top-5) | 20.0% |
| B그룹 (oracle ∩ top-5 밖) | **54.9%** ← 핵심 병목 |
| C그룹 (oracle 없음) | 25.1% |

**B그룹 특이점**: oracle이 top-5 밖에 있어도 Top-1 hit이 **78.6%**. Top-10 가중 평균이 oracle 없이도 1cm 이내 예측을 만들어내는 경우가 많다는 의미 → oracle rank 개선이 OOF로 연결되는 한계 효율이 낮은 구간.

**[다음 실험: ListMLE weight 그리드 탐색]**

| 실험 | LISTMLE_WEIGHT | 목적 |
|---|---:|---|
| A (완료) | 0.0 | clean baseline |
| **B (진행 중)** | **0.05** | 약한 oracle ranking 압력 |
| C | 0.10 | 중간 |
| D | 0.20 | 강한 (0.5는 이미 OOF 하락 확인) |

관찰 기준: oracle rank mean < 15 + OOF ≥ 0.6466이면 해당 weight 유효.

**[실험 B — LISTMLE_WEIGHT=0.05 결과]**

| 지표 | 실험 A | 실험 B | 변화 |
|---|---:|---:|---:|
| OOF (Top-10) | 0.6466 | **0.6489** | +0.23pp |
| Selector efficiency | 86.3% | **86.6%** | +0.3pp |
| Oracle rank mean | 21.9 | **13.1** | −8.8 |
| Oracle rank median | 22 | **5** | −17 |
| Oracle in Top-5 | 28.3% | **47.1%** | +18.8pp |
| A그룹 | 20.0% | **33.4%** | +13.4pp |
| B그룹 | 54.9% | **41.5%** | −13.4pp |

모든 지표 개선. 특히 oracle rank median 22→5 — ListMLE가 oracle 후보를 상위권으로 효과적으로 끌어올림.

**[실험 C — LISTMLE_WEIGHT=0.10 결과]**

| 지표 | 실험 B | 실험 C | 변화 |
|---|---:|---:|---:|
| OOF (Top-10) | **0.6489** | 0.6480 | −0.09pp |
| Selector efficiency | **86.6%** | 86.5% | −0.1pp |
| Oracle rank mean | 13.1 | **10.0** | −3.1 |
| Oracle rank median | 5 | **4** | −1 |
| Oracle in Top-5 | 47.1% | **51.8%** | +4.7pp |
| A그룹 | 33.4% | **36.0%** | +2.6pp |
| B그룹 | **41.5%** | 38.9% | −2.6pp |

Oracle ranking 지표는 계속 개선됐으나 OOF가 0.09pp 소폭 하락. 노이즈 범위(±0.12pp) 이내지만 B→C 추세가 꺾이기 시작.

**[실험 D — LISTMLE_WEIGHT=0.20 결과]**

| Fold | R-Hit |
|---:|---:|
| 1 | 0.6614 |
| 2 | 0.6453 |
| 3 | 0.6439 |
| 4 | 0.6500 |
| 5 | **0.6315** ← 약화 |
| **CV mean** | **0.6464 ± 0.0097** |

- CV std 0.0053 → **0.0097** (분산 폭증)
- Fold 5 = 0.6315 — PAIRWISE_WEIGHT=0.5 붕괴(0.5974) 초기 징후와 같은 패턴
- OOF 0.6465로 B(0.6489) 대비 명확히 열위

**[ListMLE 그리드 탐색 결론]**

| 실험 | LML | OOF | CV std | 평가 |
|---|---:|---:|---:|---|
| A | 0.00 | 0.6466 | 0.0053 | clean baseline |
| **B** | **0.05** | **0.6489** | **0.0053** | **최적 ★** |
| C | 0.10 | 0.6480 | — | 소폭 하락 |
| D | 0.20 | 0.6465 | 0.0097 | 불안정 |
| (참고) | 0.50 | 0.6461\* | — | 이전 실험, SOFT_TEMP=0.003 혼재 |

\* 이전 LML=0.5 실험은 SOFT_TEMP=0.003 상태로 진행됐을 가능성 있어 단순 비교 불가

**결론:** `LISTMLE_WEIGHT=0.05`가 최적. OOF가 높으면서 std가 안정적인 구간. 이상 config 확정 — 이후 실험의 baseline으로 사용.

**[현재 확정 Config (LML 그리드 이후)]**

```python
CANDIDATES      = 50
AUG_MODE        = 'yaw'       # → 'yaw_speed'로 교체 예정
SOFT_TEMP       = 0.005
PAIRWISE_WEIGHT = 0.25
LISTMLE_WEIGHT  = 0.05
TOPK            = 10
D_MODEL         = 128
```

---

### 2026-05-22 (9)

**[실험 로드맵 수립 및 speed-scale augmentation 구현]**

현재 best OOF 0.6489(≈LB 0.674 예상) 기준 LB 0.70 달성을 위한 단계별 계획 확정.

**목표 갭 분석**

| 단계 | 목표 OOF | 예상 LB | 핵심 변경 |
|---|---:|---:|---|
| 현재 | 0.6489 | ~0.674 | baseline |
| speed-scale | 0.653~0.658 | ~0.678~0.683 | augmentation 일반화 |
| smart-50 | 0.658~0.665 | ~0.683~0.690 | 후보 교체 (수 유지) |
| 3-seed ensemble | 0.665~0.670 | ~0.690~0.695 | 15-model logit avg |
| reg blend | 0.670~0.675 | ~0.695~0.700 | direct regression 보완 |

**[speed-scale augmentation 구현]**

`dataset.py`에 `augment_speed_scale_gpu()` 추가:

```python
def augment_speed_scale_gpu(coords, labels, scale_range=(0.85, 1.15), prob=0.5):
    """마지막 관측점 p0 기준으로 모든 이동량을 스케일링.
    yaw 이후 적용하여 '같은 방향이지만 다른 속도'의 궤적 생성."""
    scale = Uniform(lo, hi), Bernoulli(prob)로 샘플별 스케일 결정
    coords_aug = p0 + scale * (coords - p0)   # p0은 고정
    labels_aug = p0 + scale * (label  - p0)
```

- p0(마지막 관측점)은 불변 — 예측 기준점은 바뀌지 않음
- 후보 생성(d1/acc/jerk)도 동일 비율 스케일 → 후보 피처의 상대 비율 보존
- 목적: selector가 절대 속도보다 Frenet 방향 패턴과 후보 ranking에 집중하도록 유도

`config.py`:
```python
AUG_MODE          = 'yaw_speed'
SPEED_SCALE_RANGE = (0.85, 1.15)
SPEED_SCALE_PROB  = 0.5
```

**[multi-seed 앙상블 인프라 구현]**

train.py / analyze.py / predict.py 에 CLI 인자 추가:

```bash
python train.py --seed 42          # seed별 서브디렉토리에 저장
python analyze.py --seed 42        # 해당 seed 모델 로드
python predict.py --seeds 42 123 777  # 15-model logit 평균 앙상블
```

- 모델 저장 경로: `outputs/seed{N}/selector_fold{i}.pt`
- 앙상블 방식: 각 모델의 logit을 평균 → `selector_predict(topk=TOPK)` (좌표 평균보다 우수)

**[analyze.py 개선]**

- `oracle_selector_decomposition()`: Oracle rank Top-10 추가, 그룹별 `Top-10 hit` 출력
- `oracle_rank < k` 탐색 범위: `[1, 3, 5, 7]` → `[1, 3, 5, 7, 10]`

**[다음 실험]**

```bash
# speed-scale 첫 실험 (AUG_MODE='yaw_speed', RANGE=0.85~1.15)
python dev/experiments/train.py --seed 42
python dev/experiments/analyze.py --seed 42
```

성공 기준: OOF ≥ 0.651, efficiency ≥ 84%, oracle rank mean 하락, B그룹% 감소.

---

### 2026-05-22 (10)

**[실험 E — yaw + speed-scale(0.85~1.15) 결과 및 결론]**

설정: `AUG_MODE='yaw_speed', SPEED_SCALE_RANGE=(0.85, 1.15), SPEED_SCALE_PROB=0.5`

**Fold별 결과**

| Fold | R-Hit |
|---:|---:|
| 1 | 0.6554 |
| 2 | 0.6404 |
| 3 | 0.6486 |
| 4 | 0.6500 |
| 5 | 0.6381 |
| **CV mean** | **0.6465 ± 0.0064** |

**베이스라인(실험 B, yaw-only) 대비 비교**

| 지표 | 실험 B (yaw) | 실험 E (yaw+speed) | 변화 |
|---|---:|---:|---:|
| OOF R-Hit | **0.6489** | 0.6465 | **−0.24pp** |
| Oracle R-Hit | 0.7489 | 0.7489 | 0 |
| Selector efficiency | **86.6%** | 86.3% | −0.3pp |
| Oracle rank mean | 13.1 | 13.0 | −0.1 |
| Oracle rank median | 5 | 5 | 0 |
| Oracle in Top-5 | 47.1% | 46.7% | −0.4pp |
| A그룹 | 33.4% | 33.2% | −0.2pp |
| B그룹 | **41.5%** | 41.7% | +0.2pp |
| C그룹 | 25.1% | 25.1% | 0 |
| Best Top-k | 10 | 10 | — |
| Best temp | 1.0 | 1.0 | — |

**analyze.py 전체 결과 (실험 E)**

```
Oracle candidate rank statistics:
  mean=13.0  median=5  p75=20  p90=43
Oracle in Top- 1: 0.1829
Oracle in Top- 3: 0.3651
Oracle in Top- 5: 0.4669
Oracle in Top- 7: 0.5460
Oracle in Top-10: 0.6236

A: oracle hit & in top-5   :  3316샘플 (33.2%) | Top-1=0.8706  Top-5=0.8890  Top-10=0.8745
B: oracle hit & NOT top-5  :  4173샘플 (41.7%) | Top-1=0.7850  Top-5=0.8277  Top-10=0.8541
C: oracle miss (cand limit):  2511샘플 (25.1%) | Top-1=0.0000  Top-5=0.0008  Top-10=0.0004
```

**C그룹 분석 (실험 E — 기준과 동일)**

```
C그룹: 2511개  (25.1%)
nearest-cand dist  : mean=2.70cm  p50=1.72  p75=3.23  p90=5.81  p95=7.79

True label (Frenet, speed×2 정규화):
par  : mean=0.77  std=0.66  p5=-0.35  p95=1.29
perp : mean=0.15  std=0.59  |p5|=0.02  |p75|=0.45  |p95|=1.02
jerk : mean=0.48  p75=0.55  p95=1.70

현재 후보 커버리지: par=[0, 1.20]  |perp|≤0.60  |jerk|≤0.50
C그룹 중 |perp| 초과: 15.5%
C그룹 중 par 범위 밖: 20.4%

C그룹 nearest 후보:
  latency_s085   21.8%
  jerk_xl_pos    16.1%
  turn_p060      14.1%
  jerk_xl_neg    10.9%
  turn_n060       9.4%
```

**결론: speed-scale(0.85~1.15) 효과 없음 — 0.8~1.2 건너뜀, yaw-only 복귀**

- OOF −0.24pp는 CV std(0.0064) 범위 내 노이즈지만, 개선 조건(OOF ≥ 0.653) 미달
- Oracle rank / A/B/C 그룹 비율 모두 기준과 사실상 동일 → 유의미한 변화 없음
- 왜 효과 없었나:
  - speed-scale은 selector가 절대 속도보다 방향/패턴을 보도록 유도하는 augmentation
  - 그러나 이미 Frenet 피처로 정규화되어 있어 → selector는 이미 상대적 패턴 학습 중
  - 즉, speed-scale이 해결하려는 문제가 현재 구조에서는 존재하지 않음
- `config.py` `AUG_MODE='yaw'` 복귀

**다음 단계: Smart 50-cand 재설계**

speed-scale best config = yaw-only(기존과 동일) → 즉시 Smart 50-cand로 이동 가능.

| 이유 | 근거 |
|---|---|
| C그룹 25.1% 고착 | Oracle ceiling이 74.89%로 묶임 → 후보 공간 자체 개선 필요 |
| B그룹 41.7% 최대 병목 | Selector가 oracle 후보를 top-5 밖으로 밀어냄 → 노이즈 후보 제거로 난이도 완화 |
| 60-cand 실패 교훈 | 후보 수 증가 금지, 50개 유지하면서 저기여 후보를 고가치 후보로 교체 |

**Smart 50-cand 설계 원칙**

제거 후보 기준 (oracle 기여도 하위 + 중복성 높음):
- `frenet_par090_p000`, `frenet_par100_p000`, `frenet_par100_n010` 등 파라미터 차이 작은 Frenet 계열
- `jerk_small_pos/neg` (jerk=±0.08, 기여도 미미 — jerk_l/xl이 이미 상위권 커버)

추가 후보 기준 (C그룹 분석 기반):
- `latency_s080` / `latency_s075`: latency_s085가 C그룹 nearest 21.8% → 더 강한 보정 필요
- C그룹 par 초과(20.4%) 대응: `frenet_par130_n020`, `frenet_par140_n030`
- C그룹 |perp| 초과(15.5%) 대응: `turn_p070`, `turn_n070` (|perp|=0.70)

목표: Oracle ≥ 76%, Efficiency 유지(≥86%), OOF ≥ 0.655
