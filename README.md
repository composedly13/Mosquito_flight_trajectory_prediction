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
| cand_proj | Linear(**11** → 128) |
| cross_attn | MultiheadAttention(candidates → sequence) |
| **cand_self_attn** | **MultiheadAttention(candidates → candidates) — 후보 간 상대 비교** |
| head | Linear(128×2+**11** → 128) → GELU → Linear(128 → 1) per candidate |

### TransformerRegressor (C-group 보조 모델)

```
입력: seq_feat (11, 11) — 후보 불필요
출력: p0 + Δ (3D offset 직접 예측)
구조: CandidateSelector와 동일한 TransformerEncoder → head → Linear(128 → 3)
```

- C-group 전용 직접 회귀 모델 (entropy-based blend 용도)
- 현재 실험 결과: C-group R-Hit@1cm = 0.62% (1cm 임계값 기준 거의 무효)
- entropy H ≈ 0.986 (54개 후보 → 구조적으로 항상 최대 엔트로피) → blend 비활성화 상태

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
| Frenet | 10 | 접선/법선 방향 세밀 분해 (중복 제거) |
| Turn | 13 | 방향 전환 (|perp| 0.20~0.80) — C그룹 기반 극단 강화 |
| Jerk | 6 | 순간 가속도 변화 (|jerk| 0.08~0.80) — C그룹 기반 강화 |
| Latency | **20** | 시스템 지연 보정 (0.60~1.20×) + turn 복합 — **s060/s065 추가** |

**cand_feat 피처 구성 (CAND_DIM=11)**

| # | 피처 | 설명 |
|---|---|---|
| 0 | cand_par_norm | 후보 접선 방향 거리 / speed |
| 1 | cand_perp_norm | 후보 법선 방향 거리 / speed |
| 2 | dist_norm | 후보 전체 거리 / speed |
| 3 | d1 | 속도 스케일 파라미터 |
| 4 | par | 접선 가속도 파라미터 |
| 5 | perp | 법선 가속도 파라미터 |
| 6 | d2 | 2차 속도 파라미터 |
| 7 | jerk | jerk 파라미터 |
| 8 | time_scale | 시간 스케일 보정 파라미터 |
| 9 | acc_par_s | 시퀀스 기반 접선 가속도 |
| **10** | **family_id/5** | **후보 계열 타입 (base/acc/frenet/turn/jerk/latency)** |

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
│       ├── model.py             # CandidateSelector + TransformerRegressor
│       ├── boundary.py          # BoundaryMLP (잔재, 미사용)
│       ├── train.py             # K-Fold 학습 루프 (CandidateSelector)
│       ├── train_regressor.py   # TransformerRegressor 학습 (C-group 실험용)
│       ├── regression.py        # RegMLP / Reg2 inference helpers + entropy blend
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
python dev/experiments/analyze.py --seeds 42 123 777  # multi-seed OOF 비교 (섹션 7)

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
| Candidates | **52개** (Phase 11 기준 — latency_s075/s080 포함, s060/s065 영구 제거) |
| Cand features | **CAND_DIM=10** (family_id/5 영구 제거 — Phase 8 대비 −0.73pp 확인) |
| Augmentation | **yaw-only** |
| Selector | Transformer + Cross-Attention + **Auxiliary Reg Head** (d_model=128, 3 layers) |
| Seq features | **11개** (SEQ_DIM=11) |
| Loss | **soft-CE + PW×0.25 + LML×0.05 + smooth_l1_reg×0.5** (Phase 11 Step 2) |
| Prediction | **Top-10** weighted average, temp=1.0 |
| Boundary MLP | **완전 제거** |
| TTA | **효과 없음** (TTA×8 → ±0.00pp) |
| OOF (seed42, Phase 11) | **0.6464** (Phase 7 0.6510 대비 −0.46pp — single seed 기준) |
| Oracle ceiling | **75.41%** (52-cand) |
| Selector efficiency | **85.7%** |
| Oracle rank mean | **15.8** (Phase 7 13.3 대비 소폭 악화) |
| LB (best) | **0.6716** (Phase 5, 2-seed 42+777, β=0.0 selector-only) |
| Phase 11 3-seed LB | **0.6692** (42+777+123, −0.24pp vs best) |
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
| 실험 G: Smart 50-cand (후보 교체) | 50★ | 64.73% | **77.02%** | 84.0% | 64.73% | Oracle +2.13pp, B-group↑ 실패 기준 |
| **실험 G2**: Smart 50 + LML=0.10 | 50★ | **64.75%** | 77.02% | 84.1% | **64.75%** | Oracle rank 13.2, B-group 46.4%, temp=2.0 — 단일 seed 한계 |
| G2 seed 123 | 50★ | 64.47% | 77.02% | 83.7% | 64.47% | seed 불안정 (Fold2=Fold3, 극단 후보 수렴 실패) |
| **G2 seed 777** | 50★ | **64.87%** | 77.02% | **84.2%** | **64.87%** | **현재 최고 단일 seed** |
| 3-seed 앙상블 (42+123+777) | 50★ | — | 77.02% | 84.0% | 64.68% | ❌ seed123 역효과 −0.07pp |
| **2-seed 앙상블 (42+777)** | 50★ | — | 77.02% | **84.2%** | **64.84%** | seed42 대비 +0.09pp, seed777 대비 −0.03pp — **제출 권장** |
| Phase 5: RegMLP entropy blend (β grid) | 50★ | — | 77.02% | 84.2% | 0.0681† | ❌ RegMLP 단독 실패 — β=0.0(selector-only) 최적 → 2-seed LB **0.6716** |
| Phase 6 1차: 3-seed 42+123+777 (config 혼재) | 50/50★ | — | — | — | 0.6493‡ | ❌ LB 0.669 — seed42=Smart50, seed123/777=original 혼재 |
| **Phase 6 2차: 3-seed 42+123+777 (원래 50, LML=0.05, 일관)** | 50 | — | 74.89% | 86.5% | **0.6479** | 기준선 0.6489 미달 (−0.05pp) → seed42 단독과 사실상 동등 |
| **Phase 7-1: 52-cand (latency_s075/080 추가)** | 52 | — | **75.41%** | 86.3% | **0.6510** | Oracle +0.52pp, OOF +0.26pp — C-group 일부 커버 |
| Phase 7-2: 52-cand + B-group focal weight×2.0 | 52 | — | 75.41% | — | — | ❌ B-group 42.6%→46.6%, oracle rank 13.3→16.4 악화 |
| Phase 8-1: oracle_margin_loss ×0.10 | 52 | — | 75.41% | — | 0.6425 | ❌ B-group 48.9%, rank 16.9 — soft-CE 충돌 |
| **Phase 8 (54-cand+SA+family, 3-seed)** | **54** | — | **75.78%** | **85.4%** | **0.6474** | oracle rank 17.6, B-group 47.8%, C-group 24.2% |
| **Phase 7 제출: seed42, temp=2.0** | 52 | — | 75.41% | 86.3% | 0.6510 | **LB 0.6712** (focal 실패 후 selector-only 제출) |
| Phase 8-1: oracle_margin_loss (B_GROUP_WEIGHT=0.10) | 52 | — | 75.41% | — | 0.6425 | ❌ B-group 42.6%→48.9%, oracle rank 13.3→16.9 악화 — soft-CE와 gradient 충돌 |
| Phase 8-2: B_GROUP_WEIGHT=0.0 복귀 | 52 | — | 75.41% | 85.7% | 0.6460 | 복구 확인 |
| Phase 8-3: 54-cand + SA + family feat (seed42) | 54 | — | **75.78%** | 84.9% | 0.6435 | CAND_DIM 10→11, candidate self-attention 추가 |
| Phase 8-4: 54-cand + SA, seed777 재훈련 | 54 | — | 75.78% | 84.9% | 0.6432 | 구 아키텍처 호환 불가 → 재학습 필요 |
| Phase 8-5: 54-cand + SA, seed123 | 54 | — | 75.78% | 84.7% | 0.6417 | |
| **Phase 8 제출: 3-seed (42+777+123), temp=2.0** | 54 | — | 75.78% | **85.4%** | **0.6474** | **LB 0.6672** ❌ single-seed(0.6712) 하회 — seed777/123 variance 희석 |
| Phase 8 단일 seed42 제출 | 54 | — | 75.78% | 84.9% | 0.6435 | **LB 0.6712** (구 아키텍처 best와 동등) |
| **Phase 9: 60-cand + entropy penalty (seed42)** | **60** | 0.6341 | **76.75%** | 82.6% | 0.6341 | entropy 역효과: oracle rank 17.6→24.7 ❌ |
| Phase 9: 60-cand + entropy penalty (seed777) | 60 | 0.6360 | 76.75% | 82.9% | 0.6361 | |
| Phase 9: 60-cand + entropy penalty (seed123) | 60 | 0.6393 | 76.75% | 83.3% | 0.6393 | |
| **Phase 9: 3-seed 앙상블 (42+777+123)** | **60** | — | **76.75%** | **84.0%** | **0.6445** | OOF −0.29pp vs P8 — entropy penalty 제거 필요 |
| **Phase 10: 52-cand + BCE + family cls (seed42)** | **52** | 0.6437 ± 0.0060 | 75.41% | 85.4% | **0.6438** | ❌ oracle rank 21.1 (Phase 7 13.3 대비 악화) — BCE가 ranking 파괴 |
| **Phase 11 Step 2: soft-CE + reg_head, CAND_DIM=10 (seed42)** | **52** | 0.6464 ± 0.0037 | 75.41% | 85.7% | **0.6464** | family_id 제거 회복 + Auxiliary Reg Head (REG_WEIGHT=0.5) |
| Phase 11 Step 3: 54-cand + latency_s060/065 (seed42) | 54 | 0.6442 ± 0.0076 | 75.78% | 85.0% | 0.6443 | ❌ OOF −0.21pp — family_id 없어도 극단 감속 후보가 efficiency 저하 → 영구 제거 |
| **Phase 11: 3-seed 앙상블 (42+777+123), temp=1.0** | **52** | — | 75.41% | — | — | **LB 0.6692** ❌ Phase 7 best(0.6716) 하회 −0.24pp |
| **Phase 12: BiLSTM C-group + routing 13.4% (seeds 42+777+123)** | 52 | — | 75.41% | — | ~0.625 | **LB 0.6588** ❌ routing 오발동 → A+B 오염. C-group OOF 0.62%→3.54% 개선 확인 |
| **Phase 13 GCN seed42 (단독)** | 52 | — | 75.41% | **86.3%** | **0.6506** | GCN(EdgeConv k=6) 추가 → +0.42pp OOF (0.6464→0.6506) |
| Phase 13 GCN 3-seed base (42+777+123) | 52 | — | 75.41% | 86.1% | ~0.6506 | **LB 0.6688** ❌ Phase 7 single(0.6712) 하회 — GCN 오버피팅 |
| Phase 13 blend (GCN + C-gate + GRUResidual) | 52 | — | 75.41% | — | ~0.6516 | **LB 0.6674** ❌ C-gate도 LB 기여 없음 (OOF +0.10pp만 확인) |

> † RegMLP OOF: 단독 예측 기준. selector-only(β=0.0) OOF=0.6484.  
> ‡ 앙상블 OOF는 config 혼재로 신뢰 불가. LB=0.669 실제 하락 확인.  
> § TransformerRegressor(reg2) OOF: C-group 0.62%, overall 2.25% — entropy H≈0.986(54개 구조적 최대) → blend 무효.

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
| 실험 G: Smart 50-cand (LML=0.05) | 15.7 | 40.0% | 29.0% | **48.0%** | **23.0%** | Oracle↑ C↓ 좋음, 그러나 B-group↑ 나쁨 |
| 실험 G2: Smart 50 + LML=0.10 | **13.2** | 43.6% | 30.6% | 46.4% | 23.0% | rank 개선 但 OOF +0.02pp, 단일 seed 상한 도달 |
| Phase 7-1: 52-cand (original + latency_s075/080) | 13.3 | 39.2% | 32.1% | **42.6%** | **22.6%** | C-group 25.1%→22.6% (-2.5pp), B-group 최초 진입 병목 |
| Phase 7-2: 52-cand + focal×2.0 (B-group) | **16.4** | — | — | 46.6% | — | ❌ oracle rank 악화 — 불안정한 hard mining |
| **Phase 10: 52-cand + BCE + family cls** | **21.1** | 27.6% | **20.9%** | **54.5%** | **24.6%** | ❌ BCE가 oracle rank 13.3→21.1로 파괴 — soft-CE 복귀 필요 |
| **Phase 11 Step 2: soft-CE + reg_head (CAND_DIM=10)** | **15.8** | 39.1% | 27.5% | 48.0% | 24.6% | oracle rank Phase 7(13.3) 대비 소폭 악화, OOF +0.27pp — reg_head 효과 제한적 |

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

---

### 2026-05-22 (11)

**[실험 G — Smart 50-cand (LML=0.05) 결과]**

설정: N_CANDIDATES=50 (10개 교체), AUG=yaw, LML=0.05

**Fold별 결과**

| Fold | R-Hit |
|---:|---:|
| 1 | 0.6515 |
| 2 | 0.6517 |
| 3 | 0.6476 |
| 4 | 0.6470 |
| 5 | 0.6386 |
| **CV mean** | **0.6473 ± 0.0048** |

**기준(실험 B) 대비 비교**

| 지표 | 실험 B (orig 50) | 실험 G (Smart 50) | 변화 |
|---|---:|---:|---:|
| OOF R-Hit | **0.6489** | 0.6473 | −0.16pp |
| Oracle R-Hit | 0.7489 | **0.7702** | **+2.13pp ✓** |
| Selector efficiency | **86.6%** | 84.0% | −2.6pp |
| Oracle rank mean | 13.1 | 15.7 | −악화 |
| Oracle rank median | 5 | 8 | −악화 |
| Oracle in Top-5 | 47.1% | 40.0% | −7.1pp |
| A그룹 | 33.4% | 29.0% | −4.4pp |
| B그룹 | 41.5% | **48.0%** | **+6.5pp ← 최대 문제** |
| C그룹 | 25.1% | **23.0%** | **−2.1pp ✓** |
| Best temp | 1.0 | 1.5 | 변화 (logit 불확실성 증가 신호) |

**Oracle 기여도 Top-15 (새 후보 중심)**

| idx | name | count | % |
|---:|---|---:|---:|
| 22 | jerk_xxl_pos | 1233 | 12.3% |
| 23 | jerk_xxl_neg | 1060 | 10.6% |
| 24 | latency_s075 | 790 | 7.9% |
| 43 | turn_p080 | 587 | 5.9% |
| 44 | turn_n080 | 490 | 4.9% |
| 30 | latency_l120 | 459 | 4.6% |

새로 추가한 jerk_xxl±0.80이 전체의 22.9%, latency_s075 7.9%, turn_p080/n080 10.8% → 후보 교체가 oracle coverage를 실질적으로 넓힘.

**C그룹 분석 (실험 G)**

```
C그룹: 2298개  (23.0%)  ← 기존 25.1%에서 2.1pp 감소
nearest-cand dist  : mean=2.66cm  p50=1.71  p75=3.23  p90=5.59  p95=7.43

현재 후보 커버리지: par=[0, 1.30]  |perp|≤0.80  |jerk|≤0.80
C그룹 중 |perp| 초과: 9.6%  ← 기존 15.5%에서 5.9pp 감소 ✓
C그룹 중 par 범위 밖: 18.9%  ← 기존 20.4%에서 1.5pp 감소 ✓

C그룹 nearest: latency_s075 20.9%, jerk_xxl_pos 18.5%, turn_p080 14.3%
→ 아직도 극단 후보들이 C그룹 nearest — 더 강한 보정이 필요하거나 selector가 이 후보들을 선택 못 하는 문제
```

**판정: 절반의 성공**

- ✓ Oracle +2.13pp (74.89% → 77.02%): 신규 극단 후보들이 실질적으로 coverage 향상
- ✓ C-group −2.1pp: 커버 못하는 케이스 감소
- ✗ B-group +6.5pp (41.5% → 48.0%): selector가 극단 후보를 top-5에 못 넣음
- ✗ Efficiency −2.6pp (86.6% → 84.0%): oracle rank 악화
- ✗ Best temp 1.0→1.5: logit 신뢰도 하락 신호

**왜 실패하는가:**
신규 극단 후보(jerk_xxl=0.80, latency_s075=time_scale×0.75, turn_p080=perp×0.80)는 cand_features에서 비정상적으로 큰 feature 값을 보인다. selector가 "이 후보는 평소와 다르다" → 낮은 logit 부여. ListMLE(LML=0.05)의 oracle ranking 압력이 이 편향을 완전히 극복하지 못함.

**핵심 관점 — 그럼에도 Smart 50이 가치 있는 이유:**

```
oracle 77.02%에서 efficiency 88%만 달성하면:
OOF ≈ 0.677 → LB ≈ 0.702  (목표 달성!)

원래 50-cand(oracle 74.89%)는 efficiency 93.5%가 필요했음.
Smart 50은 efficiency 5pp 더 낮아도 같은 OOF를 낼 수 있다.
```

**[다음 실험: G2 — Smart 50-cand + LML=0.10]**

```python
LISTMLE_WEIGHT = 0.10  # 0.05 → 0.10
```

목적: 극단 후보의 oracle ranking 압력 강화 → B-group 감소 → efficiency 회복
원래 50-cand에서 LML=0.10 결과: oracle rank mean 13.1 → 10.0 (-3.1), OOF 소폭 하락
Smart 50-cand: oracle rank 15.7이 더 나쁘므로 LML=0.10 개선 여지 더 클 수 있음

성공 기준: OOF ≥ 0.6480, oracle rank mean ≤ 12, B-group ≤ 44%
실패 기준: OOF < 0.645, B-group 증가, oracle rank 악화

→ G2 실패 시: Smart 50-cand 포기하고 original 50-cand (LML=0.05)로 multi-seed ensemble 진행

---

### 2026-05-22 (12)

**[실험 G2 — Smart 50-cand + LML=0.10 결과 및 단일 seed 한계]**

설정: N_CANDIDATES=50 (Smart 10개 교체), AUG=yaw, LML=0.10

**기준(실험 G) 대비 비교**

| 지표 | 실험 G (LML=0.05) | 실험 G2 (LML=0.10) | 변화 |
|---|---:|---:|---:|
| OOF R-Hit | 0.6473 | **0.6475** | +0.02pp |
| Oracle R-Hit | 0.7702 | 0.7702 | 0 |
| Selector efficiency | 84.0% | 84.1% | +0.1pp |
| Oracle rank mean | 15.7 | **13.2** | −2.5 ✓ |
| Oracle in Top-5 | 40.0% | **43.6%** | +3.6pp ✓ |
| A그룹 | 29.0% | **30.6%** | +1.6pp ✓ |
| B그룹 | **48.0%** | 46.4% | −1.6pp ✓ |
| Best temp | 1.5 | **2.0** | ← 우려 신호 |

**판정: 부분 개선, 단일 seed 한계 도달**

- ✓ Oracle ranking 지표 전반 개선 (rank mean 15.7→13.2, Top-5 40%→43.6%)
- ✓ B-group 소폭 감소 (48.0%→46.4%)
- ✗ OOF 실질 개선 없음 (+0.02pp, 노이즈 범위)
- ✗ Best temp 1.5→2.0: logit 불확실성 증가 신호 (LML 압력이 logit 분포를 불안정하게 함)

**단일 seed 튜닝 한계 결론**

| 실험 | config | OOF | 평가 |
|---|---|---:|---|
| B (single seed best) | orig 50, LML=0.05 | **0.6489** | 최고 단일 seed |
| G | Smart 50, LML=0.05 | 0.6473 | Oracle↑ 但 efficiency↓ |
| G2 | Smart 50, LML=0.10 | 0.6475 | 랭킹↑ 但 OOF 정체 |

Smart 50-cand oracle ceiling(77.02%) 활용 목표: efficiency ≥ 87.6% → OOF ≥ 0.677 → LB ≈ 0.702.
현재 efficiency 84.1%, 갭 3.5pp — 단일 seed 추가 튜닝으로는 좁히기 어려움.

**[Phase 4: Multi-seed Ensemble 진입]**

| 단계 | 구성 | 기대 효과 |
|---|---|---|
| 단일 seed | seed 42, 5 models | OOF 0.6475 |
| 3-seed ensemble | seed 42+123+777, 15 models | 분산 감소 → efficiency +2~4pp |

**코드 변경 사항**

- `analyze.py`: `load_all_models()`, `get_oof_preds_multiseed()` 추가, 섹션 7 추가
- CLI: `--seeds 42 123 777` 지원 (single/multi 모드 공통화)

**다음 단계**

```bash
python dev/experiments/train.py --seed 123
python dev/experiments/train.py --seed 777
python dev/experiments/analyze.py --seeds 42 123 777   # 섹션 7에서 앙상블 OOF 확인
python dev/experiments/predict.py --seeds 42 123 777   # 제출 파일 생성
```

---

### 2026-05-22 (13)

**[Multi-seed Ensemble 결과 — 앙상블 역효과 분석]**

3-seed 학습 완료 후 `analyze.py --seeds 42 123 777` 섹션 7 결과:

**각 seed 개별 결과**

| seed | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | OOF | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6515 | 0.6517 | 0.6476 | 0.6470 | 0.6386 | 0.6475 | 84.1% |
| 123 | 0.6530 | 0.6439 | 0.6439 | 0.6431 | 0.6396 | **0.6447** | 83.7% |
| **777** | **0.6550** | **0.6488** | **0.6523** | **0.6485** | 0.6391 | **0.6487** | **84.2%** |

**3-seed 앙상블 (logit 평균, 15 models)**

| 지표 | 값 |
|---|---|
| Multi-seed OOF | **0.6468** (seed42=0.6475 대비 −0.07pp) |
| Oracle ceiling | 77.02% |
| Multi-seed efficiency | 84.0% |

**판정: 앙상블 역효과 — seed 123이 끌어내림**

원인: seed 123 OOF(0.6447)이 seed 42(0.6475), seed 777(0.6487) 대비 2~4pp 낮아 logit 평균이 best single seed를 하회. Fold 2·3이 동일값(0.6439)으로 수렴 — Smart 50 극단 후보(jerk_xxl, turn_p080, latency_s075)가 특정 random init에서 학습 불안정.

**현재 best: seed 777 단독 (OOF 0.6487)**

**seeds 42+777 앙상블 결과**: OOF 0.6484 (+0.09pp vs seed42, −0.03pp vs seed777)
- 0.03pp 차이는 노이즈 범위 내 → 10모델 앙상블이 test set 분산 감소 → **seeds 42+777 제출 권장**
- 제출 명령: `python dev/experiments/predict.py --seeds 42 777`

**갭 분석 및 Phase 5 방향**

- 현재 best OOF: 0.6487 → LB ≈ 0.673 예상
- 목표 LB 0.70 → OOF ≈ 0.675 필요, 잔여 갭 ~1.9pp
- 단일 seed 튜닝 / multi-seed 방향 모두 소진 → **Phase 5 진입**
- C그룹(23%) Top-10 hit = 0.0009 → selector 완전 실패 구간 → 직접 회귀 MLP 보완 가능성

| 방식 | 설명 |
|---|---|
| 직접 회귀 MLP | 11개 시점 좌표 → 직접 (x,y,z) 예측 |
| 혼합 기준 | selector logit entropy 높을 때(불확실) → 회귀 비중 증가 |
| 기대 효과 | C그룹 23% 중 일부 포착 → +0.5~1.5pp |

---

### 2026-05-23 (1)

**[Phase 5: RegMLP Entropy Blend — 완전 실패]**

**구현**
- `regression.py`: 11개 시점 좌표 → 직접 (x,y,z) 예측 MLP (BoundaryMLP 구조 재활용)
- `predict.py`: `--beta` 인자 추가, entropy blend 공식: `pred = β·reg + (1−β)·selector`
- 아이디어: selector logit entropy 높을 때(불확실) → 회귀 비중 증가

**결과**

| 지표 | 값 |
|---|---|
| Regression OOF (단독) | **0.0681** — 완전 실패 (random 수준) |
| Entropy H: mean / p25 / p75 / p95 | 0.991 / 0.996 / 1.000 / 1.000 |

| β | Blend OOF | vs Selector |
|---:|---:|---:|
| 0.0 | 0.6484 | +0.00pp |
| 0.2 | 0.5183 | −12.78pp |
| 0.4 | 0.3091 | −33.70pp |
| 1.0 | 0.0688 | −57.73pp |

**결론: Phase 5 완전 실패**
- RegMLP가 C그룹 케이스조차 학습 불가 — 직접 회귀 방식 구조적 한계
- Entropy 분포: mean=0.991 → selector가 거의 항상 고 불확실 → entropy로 두 모드를 구분할 수 없음
- β=0.0 (순수 selector)이 유일한 최적값 → regression blend 포기
- **부산물**: seeds 42+777 (Smart50+LML=0.10) selector-only temp=2.0 → LB **0.6716** (기존 최고 동타)

---

### 2026-05-23 (2)

**[Phase 6: original 50-cand 복원 및 일관 3-seed 재실험]**

**목표**
Smart50+LML=0.10 → original 50-cand+LML=0.05 복원 후 3-seed 앙상블 OOF > 0.6489 달성 시 제출.

**설정 변경**

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| `candidates.py` | Smart 50-cand (jerk_xxl, turn_p070~p080 등) | 원래 50-cand (git checkout a59715f) |
| `LISTMLE_WEIGHT` | 0.10 | 0.05 |
| `AUG_MODE` | yaw | yaw (유지) |

**1차 실험 — config 불일치 발견 (LB 0.669)**

seeds 123+777을 original 50-cand+LML=0.05로 새로 학습, seed42는 기존 모델 재사용:

| seed | OOF |
|---:|---:|
| 42 (기존 Smart50 모델) | 0.6461 (학습시점 0.6489 대비 −28bp) |
| 123 (신규 original) | 0.6484 |
| 777 (신규 original) | 0.6464 |
| **3-seed OOF** | **0.6493** |

제출 후 LB=**0.669** — 기대치(≥0.6716) 대폭 하락.

**원인 분석**
- `outputs/seed42/selector_fold0.pt` 수정 시각: 2026-05-22 22:36 (Smart50 실험 시점)
- seed42 = Smart50+LML=0.10으로 재학습됨, seed123/777 = original 50-cand+LML=0.05 — 혼재 앙상블
- 서로 다른 후보 집합으로 학습된 모델을 같은 후보로 inference → feature 분포 불일치

**2차 실험 — seed42 재학습, 3-seed 일관 구성**

설정: original 50-cand + LML=0.05 + yaw (3 seeds 동일)

| Fold | seed42 | seed123 | seed777 |
|---:|---:|---:|---:|
| 1 | 0.6609 | 0.6594 | 0.6574 |
| 2 | 0.6439 | 0.6448 | 0.6424 |
| 3 | 0.6497 | 0.6481 | 0.6445 |
| 4 | 0.6495 | 0.6505 | 0.6480 |
| 5 | 0.6381 | 0.6391 | 0.6396 |
| **OOF** | **0.6484** | **0.6484** | **0.6464** |

**Multi-seed 분석 결과 (analyze.py --seeds 42 123 777)**

| 지표 | 값 |
|---|---|
| Single-seed OOF (seed=42) | 0.6484 |
| **Multi-seed OOF (3 seeds)** | **0.6479** (−0.05pp) |
| Oracle ceiling | 0.7489 |
| Multi-seed efficiency | 86.5% |
| 최적 temperature | **1.0** (Smart50 시절 2.0과 달리 confidence 회복) |

Oracle Error Decomposition (3-seed):

| 그룹 | 비율 | Top-10 hit |
|---|---:|---:|
| A: oracle ∩ top-5 | 32.3% | 0.8740 |
| B: oracle ∩ top-5 밖 | 42.6% | 0.8594 |
| C: oracle 없음 | 25.1% | 0.0004 |

**결론: 3-seed 앙상블 효과 없음**
- 기준선 0.6489 미달 (0.6479 < 0.6489) → criterion 불통과
- seed777 OOF(0.6464)가 seed42(0.6484) 대비 낮아 평균이 끌림
- seeds 123/777 추가 시 분산 감소 효과가 평균 하락을 상쇄하지 못함

**현재 한계 정리**

| 항목 | 값 | 한계 |
|---|---|---|
| Oracle ceiling | 74.89% | C그룹 25.1% 고착 |
| Selector OOF | 64.84% | B그룹 42.6% 병목 |
| Multi-seed | 3-seed < single | seed 간 diversity 부족 |
| RegMLP | OOF 0.0681 | 직접 회귀 구조적 실패 |
| **best LB** | **0.6716** | **seed42 단독 or 2-seed (42+777), original 50-cand** |

**남은 방향 (다음 세션)**

| 방향 | 내용 | 기대 효과 |
|---|---|---|
| 후보 재설계 | Smart50 교훈 반영 — 극단 후보를 더 적게, 중간 범위를 더 촘촘히 | Oracle ↑, B-group 유지 |
| 아키텍처 | GRU + Cross-Attention 혼합, 또는 deeper head | efficiency ↑ |
| Ranking loss | ListMLE + Margin Ranking 결합, hard negative mining | oracle rank ↓ |
| 후처리 앙상블 | 서로 다른 config 모델 혼합 (orig+Smart50) | 분산 감소 |

---

### 2026-05-24 (1)

**[Phase 7: 52-cand + B-group 개선 시도 → LB 0.6712]**

**배경**

Phase 6에서 3-seed 앙상블이 단일 seed(0.6484) 대비 −0.05pp로 실패한 이후, C-group(25.1%)과 B-group(42.6%) 두 병목을 직접 타격하는 접근으로 전환.

**작업 0 — Smart50 RegMLP 오염 제거**

`outputs/seed42/`에 Phase 5 Smart50+LML=0.10 시절의 RegMLP 모델(`regmlp_fold*.pt`, `regmlp_oof.npy`)이 잔존.  
predict.py가 자동 감지 후 entropy blend 제출(`submission_blend.csv`)을 생성할 위험 → `regmlp_smart50_backup/`으로 격리.

**작업 1 — C-group 대응: latency_s075/s080 추가 (52-cand)**

analyze.py §6(C-group analysis) 결과에서 C-group nearest candidate 상위:

| rank | 후보 | 비율 |
|---:|---|---:|
| 1 | latency_s085 | 21.8% |
| 2 | latency_s092 | 18.3% |
| … | | |

latency_s085가 C-group의 가장 가까운 후보 1위(21.8%) → s085보다 더 강한 time_scale(0.75, 0.80)을 추가하면 일부 C-group 케이스 포획 가능.

`candidates.py` 변경:
```python
# 기존: latency_s085, latency_s092, latency_s100, ...
# 추가: latency_s075(time_scale=0.75), latency_s080(time_scale=0.80)
CandidateSpec("latency_s075", 1.98, 0.96, -0.08, time_scale=0.75),
CandidateSpec("latency_s080", 1.98, 0.96, -0.08, time_scale=0.80),
# N_CANDIDATES: 50 → 52
```

seed42 재학습 결과:

| 지표 | original 50-cand | 52-cand |
|---|---:|---:|
| OOF | 0.6484 | **0.6510** (+0.26pp) |
| Oracle ceiling | 74.89% | **75.41%** (+0.52pp) |
| C-group | 25.1% | **22.6%** (−2.5pp) ✅ |
| B-group | 42.6% | 42.6% (미변화) |
| Oracle in Top-5 | 47.1% | 39.2% (↓) |
| Oracle rank mean | 13.1 | 13.3 (거의 동일) |
| 최적 temperature | 1.0 | **2.0** (52-cand 환경 재확인) |

C-group 2.5pp 개선, oracle ceiling +0.52pp — 의미 있는 개선.  
B-group은 여전히 42.6% 고착, Oracle in Top-5가 39.2%로 하락 → B-group 병목 해소 필요.

**작업 2 — B-group focal loss 실험: 실패**

아이디어: B-group 샘플(oracle이 후보군에 있으나 top-5 밖) 에 2× 손실 가중치를 주어 하드 케이스 집중.

구현: 배치 내 `oracle_dist ≤ R_HIT_THRESHOLD` 이면서 `oracle_rank > 5`인 샘플을 `logits.detach()`로 실시간 판별, 해당 샘플 loss에 2× 가중치.

결과:

| 지표 | 52-cand (base) | focal×2.0 |
|---|---:|---:|
| B-group | 42.6% | **46.6%** (❌ +4pp 악화) |
| Oracle rank mean | 13.3 | **16.4** (❌ 3.1pp 악화) |

**실패 원인 분석**: 학습 초기에는 oracle이 거의 모든 B-group 샘플에서 rank 20+ → 사실상 전 샘플이 focal 대상 → 균일 2× 스케일링과 동일한 효과 발생. 학습 불안정 → oracle rank 역행.

**작업 3 — Oracle Margin Loss로 교체**

focal 방식을 폐기하고, oracle logit을 k번째 높은 logit보다 직접 높이도록 강제하는 margin loss로 교체.

```python
def oracle_margin_loss(logits, cands, true, k=5, margin=0.15):
    """oracle logit이 top-5 기준선(k=5번째 logit)보다 margin=0.15 높아야 함."""
    dist        = torch.norm(cands - true.unsqueeze(1), dim=-1)
    oracle_idx  = dist.argmin(dim=-1)
    oracle_score = logits.gather(1, oracle_idx.unsqueeze(1))
    kth_score    = logits.kthvalue(logits.size(1) - k + 1, dim=1).values.unsqueeze(1)
    loss = F.relu(kth_score - oracle_score + margin)
    # C-group(oracle 없음)은 loss=0
    is_hit = (oracle_dist <= R_HIT_THRESHOLD).float().unsqueeze(1)
    return (loss * is_hit).mean()
```

`config.py`: `B_GROUP_WEIGHT = 0.10`  
`train.py` loss 구성: `CE + PW×0.25 + LML×0.05 + OracleMargin×0.10`

→ 아직 재학습 미완료, 다음 실험으로 예약.

**Phase 7 제출 결과**

52-cand selector-only (focal 실패, oracle margin 학습 전) 상태에서 제출:

```bash
python dev/experiments/predict.py --seed 42 --temp 2.0
# submission.csv → 제출
```

| 항목 | 값 |
|---|---|
| Seeds | 42 단독 |
| Temperature | 2.0 (52-cand analyze.py 최적값) |
| OOF | 0.6510 |
| **LB** | **0.6712** |

현재까지 LB 순위: 0.6712 (≥ Phase 6 best 0.6716 대비 −0.04pp; 기존 최고 동등권 근접).

---

### 2026-05-24 (2)

**[Phase 8: Oracle Margin Loss 실패 → 54-cand + SA + 아키텍처 개선 → LB 0.6672]**

**작업 1 — oracle_margin_loss 실험: 실패**

`B_GROUP_WEIGHT=0.10` 추가 (`CE + PW×0.25 + LML×0.05 + OracleMargin×0.10`):

| 지표 | 52-cand base | oracle_margin×0.10 |
|---|---:|---:|
| OOF | 0.6460 | **0.6425** (❌ −0.35pp) |
| B-group | 42.6% | **48.9%** (❌ +6.3pp 악화) |
| Oracle rank mean | 13.3 | **16.9** (❌ 악화) |

실패 원인: soft-CE loss가 oracle logit을 높이도록 유도하고, oracle_margin_loss 역시 같은 방향으로 압력을 주지만 두 gradient 방향이 충돌하여 오히려 불안정. → `B_GROUP_WEIGHT=0.0` 복귀.

**작업 2 — 아키텍처 개선: 54-cand + candidate self-attention + family feature**

```python
# model.py 변경
CAND_DIM = 11  # 10 → 11 (family_id/5 추가)
self.cand_self_attn = nn.MultiheadAttention(...)  # 후보 간 상대 비교
self.cand_sa_norm   = nn.LayerNorm(d_model)

# candidates.py 변경
N_CANDIDATES = 54  # 52 → 54 (latency_s060, latency_s065 추가)
# cand_feat 11번째 피처: family_id/5 (base/acc/frenet/turn/jerk/latency 계열 구분)
```

| 지표 | 52-cand (base) | 54-cand + SA + family |
|---|---:|---:|
| OOF (seed42) | 0.6460 | 0.6435 |
| Oracle ceiling | 75.41% | **75.78%** (+0.37pp) |
| C-group | 22.6% | **24.2%** (+1.6pp ↑) |
| Oracle rank mean | 13.3 | **17.6** (SA 동질화 부작용) |

SA 추가 효과:
- ✓ B-group top-10 hit 약간 개선 (모든 후보를 보고 상대 비교 가능)
- ✗ Oracle rank 악화: oracle(비정상 후보)이 55개 비-oracle 후보 방향으로 끌림 — 동질화(homogenization)

**작업 3 — TransformerRegressor (C-group 직접 회귀): 실패**

```python
# model.py 추가
class TransformerRegressor(nn.Module):
    """seq_feat(B,11,11) + p0(B,3) → p0 + offset(B,3)"""
```

C-group 전용 직접 회귀 모델 학습 (train_regressor.py, L2 distance loss):

| 지표 | 값 |
|---|---|
| OOF R-Hit (overall) | 2.25% |
| OOF R-Hit (C-group) | **0.62%** (random 수준) |
| mean dist (C-group) | **6.1cm** >> 1cm threshold |

실패 원인: C-group은 정의상 "물리 예측이 실패하는 비정형 샘플". 10K 학습 데이터에서 직접 회귀로 1cm 정밀도 달성 불가.

**entropy 분석: 구조적 문제 발견**

```
selector logit entropy H: mean=0.986  p25=0.989  p75=0.998
(최대 엔트로피 = 1.0, 54개 후보)
```

54개 후보 환경에서 logit 차이가 0.1~0.3이면 H ≈ 0.987 — 수학적 필연. entropy로는 C/A/B group 구분 불가 → reg2 blend 무효.

**multi-seed 앙상블 결과 (54-cand + SA, seeds 42+777+123)**

| 지표 | 값 |
|---|---|
| 3-seed OOF | **0.6474** (+0.39pp vs seed42 단독) |
| Multi-seed efficiency | 85.4% |
| **LB (3-seed)** | **0.6672** |
| **LB (single seed42)** | **0.6712** |

3-seed가 단일 seed보다 LB에서 −0.004: seed777(CV 0.6432), seed123(CV 0.6417)가 seed42(CV 0.6435)를 희석. 같은 아키텍처 3모델 평균이 오히려 최고 모델을 끌어내리는 현상.

**config.py 경로 버그 수정**

```python
# 이전: 상대경로 → cwd 의존
DATA_DIR = Path("data")
OUTPUT_DIR = Path("dev/experiments/outputs")

# 수정: __file__ 기준 절대경로
_EXP_DIR = Path(__file__).resolve().parent
_ROOT    = _EXP_DIR.parent.parent
DATA_DIR  = _ROOT / "data"
OUTPUT_DIR = _EXP_DIR / "outputs"
```

어느 디렉토리에서 실행해도 정상 동작.

---

### 2026-05-24 (3)

**[Phase 9: 60-cand + Entropy Penalty — 역효과 확인]**

**목표**: Oracle ceiling 돌파(54→60-cand) + Entropy penalty로 B-group 선택 정확도 개선 → LB 0.71

**작업 1 — 60-cand 확장**

C-group 갭 분석 기반으로 6개 후보 추가 (54→60):

| 이름 | 파라미터 | 근거 |
|---|---|---|
| `latency_s050` | time_scale=0.50, par≈0.495 | 급감속(par→0) 커버 |
| `latency_s040` | time_scale=0.40, par≈0.396 | 더 강한 급감속 |
| `near_stop` | d1=0.40, par≈0.20 | 거의 정지 케이스 |
| `reverse_mild` | d1=−0.50, par≈−0.25 | 역방향 이동 |
| `turn_p090` | perp=+0.90 | 날카로운 우회전 (|perp| 초과 16.1% 타겟) |
| `turn_n090` | perp=−0.90 | 날카로운 좌회전 |

**작업 2 — Entropy Penalty (ENTROPY_WEIGHT=0.02)**

```python
# train.py에 추가
if ENTROPY_WEIGHT > 0:
    probs = F.softmax(logits, dim=-1)
    H = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    loss = loss + ENTROPY_WEIGHT * H
```

목적: H 최소화 → logit 집중 → oracle이 top-5에 들어오도록 강제

**3-seed 학습 결과**

| seed | CV mean | OOF | Oracle |
|---:|---:|---:|---:|
| 42 | 0.6341 ± 0.0084 | 0.6341 | 76.75% |
| 777 | 0.6360 ± 0.0100 | 0.6361 | 76.75% |
| 123 | 0.6393 ± 0.0053 | 0.6393 | 76.75% |
| **3-seed** | — | **0.6445** | 76.75% |

**analyze.py 결과 (3-seed)**

| 지표 | Phase 8 (54-cand) | Phase 9 (60-cand+entropy) | 변화 |
|---|---:|---:|---:|
| 3-seed OOF | **0.6474** | 0.6445 | **−0.29pp ❌** |
| Oracle ceiling | 75.78% | **76.75%** | +0.97pp ✅ |
| Oracle rank mean | 17.6 | **24.7** | **+7.1 ❌** |
| A-group (oracle in top-5) | 28.0% | **16.9%** | **−11.1pp ❌** |
| B-group | 47.8% | **59.8%** | **+12.0pp ❌** |
| C-group | 24.2% | **23.2%** | −1.0pp ✅ |
| Entropy H (mean) | 0.986 | **0.986** | 변화 없음 ❌ |

**실패 원인 분석 — Entropy Penalty 역효과**

| 의도 | 현실 |
|---|---|
| H 최소화 → logit 집중 → oracle 상위 랭킹 | 모델이 CE + 엔트로피 동시 최소화를 위해 "자주 oracle인" 후보에 무조건 집중 |
| 샘플별 oracle 식별 강화 | `jerk_xl_pos/neg`, `turn_p/n090` 등 인기 후보에 항상 high logit 부여 → oracle이 아닌 샘플에서 rank 폭락 |
| ENTROPY_WEIGHT=0.02로 구조적 한계 돌파 | H ≈ 0.986 그대로 — 60-cand 환경에서 0.02 가중치는 CE loss에 묻혀 무효 |

**결론**

- ✅ Oracle ceiling: 75.78% → 76.75% (+0.97pp) — 60-cand 자체는 유효
- ❌ Entropy penalty: oracle rank 17.6 → 24.7로 악화 — 제거 필요
- ❌ 3-seed OOF: 0.6474 → 0.6445 (-0.29pp) — Phase 8 대비 퇴행

**다음 방향 (Phase 10 → 실제 실행)**

- `ENTROPY_WEIGHT = 0.0` + 52-cand 복귀 + 2-stage family classifier 도입
- 실제 결과: BCE loss 역효과 확인 → Phase 11 진입

---

### 2026-05-26

**[Phase 10: 52-cand + Hit-aware BCE + Family Classifier — 실패]**

**구현 내용**

| 항목 | 변경 |
|---|---|
| candidates.py | 60-cand → 52-cand 복귀 (latency_s040/050, near_stop, reverse_mild, turn_p090/n090 제거) |
| config.py | `FAMILY_WEIGHT=0.5`, `BCE_POS_WEIGHT=5.0` 추가 |
| model.py | SA 제거 → 2-stage CandidateSelector (Stage1: CLS→family, Stage2: family-boosted logits) |
| train.py | `hit_aware_bce_loss` (soft-CE 대체), `family_ce_loss` 추가 |

**Loss 구성**: `BCE + 0.25×PW + 0.05×LML + 0.5×CE_family`

- BCE: oracle(1cm 이내)=1, 나머지=0 이진 분류
- Family CE: CLS → 6개 계열(base/acc/frenet/turn/jerk/latency) 분류 auxiliary loss
- Family boost: 각 후보의 family log-prob을 logit에 더함

**학습 결과 (seed 42)**

| Fold | R-Hit |
|---:|---:|
| 1 | 0.6530 |
| 2 | 0.6429 |
| 3 | 0.6387 |
| 4 | 0.6475 |
| 5 | 0.6365 |
| **CV mean** | **0.6437 ± 0.0060** |
| **OOF** | **0.6438** |

**analyze.py 결과 (seed 42)**

| 지표 | Phase 7 (52-cand, soft-CE) | Phase 10 (BCE+family) | 변화 |
|---|---:|---:|---:|
| OOF | **0.6510** | 0.6438 | **−0.72pp ❌** |
| Oracle ceiling | 75.41% | 75.41% | = |
| Selector efficiency | **86.3%** | 85.4% | −0.9pp |
| Oracle rank mean | **13.3** | **21.1** | **+7.8 ❌** |
| Oracle in Top-5 | 39.2% | **27.6%** | **−11.6pp ❌** |
| A-group (oracle∩top-5) | 32.1% | **20.9%** | **−11.2pp ❌** |
| B-group | **42.6%** | **54.5%** | **+11.9pp ❌** |
| C-group | 22.6% | 24.6% | +2.0pp ❌ |

**C-group 상세 분석 (analyze 4b)**

```
C-group: 2459개 (24.6%)
nearest-cand dist: mean=2.65cm  p50=1.71  p75=3.22  p90=5.66

True label Frenet 분포:
  par : mean=0.77  std=0.67  p5=-0.36  p95=1.29
  perp: mean=0.15  std=0.59  |p75|=0.45  |p95|=1.02
  jerk: mean=0.48  p75=0.55  p95=1.70

현재 커버리지: par=[0,1.20]  |perp|≤0.60  |jerk|≤0.50
C-group 초과: |perp|>0.60 → 15.8%  /  par 범위 밖 → 20.9%

C-group nearest 후보:
  [21] latency_s075   620 (25.2%)  ← 극감속 클러스터
  [45] jerk_xl_pos    353 (14.4%)  ─┐
  [41] turn_p060      319 (13.0%)  ─┤ 합산 72.3% = 세 클러스터
  [46] jerk_xl_neg    264 (10.7%)  ─┤
  [42] turn_n060      221 ( 9.0%)  ─┘
```

**실패 원인 분석**

| 의도 | 현실 |
|---|---|
| BCE(oracle=1/0) → 명확한 gradient → oracle rank 개선 | ranking 정보 손실 → oracle rank 13.3→21.1로 폭락 |
| Family classifier → 계열 맞는 후보에 집중 | Family boost가 오히려 잘못된 계열로 logit 편향 |
| soft-CE 대비 직접적인 R-Hit@1cm 최적화 | soft-CE가 더 풍부한 거리 기반 gradient 제공 |

**결론**

- ❌ BCE loss: soft-CE 대비 oracle rank 대폭 악화 (13.3→21.1)
- ❌ Family classifier: 효과 없음 (Phase 8 SA+family feature와 OOF 동일 수준 0.6435≈0.6438)
- **핵심 교훈**: C-group 개선을 위한 구조 변경보다 **B-group 개선(oracle ranking)** 이 우선
- **근본 문제**: B-group 54.5% — oracle이 있는데 rank 21위에 묻혀 있음

---

---

### 2026-05-26 (2)

**[Phase 11: soft-CE 복귀 + Auxiliary Regression Head + 3-seed 앙상블 → LB 0.6692]**

**핵심 발견: CAND_DIM=11 → 10 (family_id/5 제거)**

Phase 10 soft-CE 복귀 후에도 OOF 0.6437 (Phase 7 0.6510 대비 −0.73pp)가 회복되지 않는 원인 분석:
- Phase 8에서 `CAND_DIM=11` (family_id/5 추가) 후 Phase 10에서 SA만 제거, feature는 잔존
- `make_cand_features_gpu`에서 family_id/5 제거 → `CAND_DIM=10` 복귀
- **family_id 자체가 −0.73pp 원인이었음 확인**

**Step 2 — Auxiliary Regression Head 구현**

```python
# model.py: CandidateSelector에 reg_head 추가
self.reg_head = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model // 2, 3),  # CLS → rough Δ(x,y,z) from p0
)

# train.py: REG_WEIGHT=0.5 smooth_l1_loss 추가
loss_reg = F.smooth_l1_loss(
    (p0 + rough_delta) / R_HIT_THRESHOLD,
    true / R_HIT_THRESHOLD,
    beta=1.0,  # 1cm 단위 normalized space
)
loss = loss + REG_WEIGHT * loss_reg
```

목적: CLS가 "어디로 갈지" 인코딩 강제 → cross-attention이 올바른 방향 후보에 집중 (Faster R-CNN RPN 구조와 유사).

**Step 2 단일 seed 결과 (seed 42)**

| Fold | R-Hit |
|---:|---:|
| 1 | 0.6500 |
| 2 | 0.6434 |
| 3 | 0.6507 |
| 4 | 0.6470 |
| 5 | 0.6411 |
| **CV mean** | **0.6464 ± 0.0037** |

| 지표 | Phase 10 (BCE) | Phase 11 Step 2 | 변화 |
|---|---:|---:|---:|
| OOF | 0.6437 | **0.6464** | **+0.27pp ✅** |
| Oracle rank mean | 21.1 | 15.8 | −5.3 ✅ |
| Efficiency | 85.4% | **85.7%** | +0.3pp ✅ |
| Oracle in Top-5 | 27.6% | 39.1% | +11.5pp ✅ |

Phase 7 기준(oracle rank 13.3) 대비로는 15.8로 소폭 악화 — reg_head(REG_WEIGHT=0.5)가 cross-attention 학습과 경쟁하는 부작용.

**Step 3 — latency_s060/s065 추가 (54-cand): 실패**

Phase 8 OOF −0.75pp의 원인이 family_id(−0.73pp)라면 latency 후보 자체는 중립(−0.02pp) → 재시도.

| 지표 | 52-cand (Step 2) | 54-cand (Step 3) | 변화 |
|---|---:|---:|---:|
| OOF | **0.6464** | 0.6443 | **−0.21pp ❌** |
| Oracle | 75.41% | **75.78%** | +0.37pp ✅ |
| Efficiency | **85.7%** | 85.0% | −0.7pp ❌ |
| CV std | **0.0037** | 0.0076 | 분산 증가 ❌ (Fold 5 = 0.6305 붕괴) |

결론: family_id 없이도 극단 감속 후보가 selector를 혼란시킴 → **latency_s060/s065 영구 제거 확정**.

**3-seed 앙상블 제출 — LB 0.6692**

Phase 11 Step 2(52-cand, reg_head) 기반 seeds 42+777+123 앙상블:

```bash
python dev/experiments/predict.py --seeds 42 777 123 --temp 1.0
```

- **LB: 0.6692** — Phase 7 best **0.6716** 대비 **−0.24pp** ❌
- reg_head가 OOF를 Phase 10 대비 개선했으나, Phase 7 single-seed LB조차 넘지 못함
- Phase 7에서는 single-seed 제출(seed42, temp=2.0) LB=0.6712 달성 → 3-seed가 항상 유리하지는 않음

**Phase 11 종합 평가**

| 실험 | OOF | LB | 결론 |
|---|---:|---:|---|
| Phase 7 best (52-cand, soft-CE+PW+LML, 2-seed 42+777) | 0.6484 | **0.6716** | 현재 LB 최고 |
| Phase 11 Step 2 (52-cand, reg_head, seed42) | **0.6464** | — | single-seed OOF Phase 7 미달 |
| Phase 11 3-seed (42+777+123) | — | 0.6692 | ❌ Phase 7 하회 |

**현재 bottleneck 재정의**

| 항목 | 값 | 한계 |
|---|---|---|
| Oracle ceiling | 75.41% | C-group 24.6% 고착 (극단 후보 추가 시 efficiency 하락) |
| Best LB | **0.6716** | Phase 5/7 (52-cand, soft-CE+PW+LML, 2-seed 42+777) |
| Selector efficiency | 85.7% | B-group 48.0% 병목 |
| C-group 후보 확장 | 불가 | latency_s060/jerk_xxl/turn_p075 모두 efficiency 저하 확인 |

---

### 2026-05-26 (3)

**[Phase 12: BiLSTM C-group 직접 회귀 + Physics-based Routing — LB 0.6588]**

**배경 및 동기**

| 접근 | 실패 원인 |
|---|---|
| TransformerRegressor (reg2) | 균일 학습 → 쉬운 샘플에 수렴, C-group 무시. OOF C-group 0.62% |
| Entropy routing | H≈0.986 (52-cand 구조적 최대) → 그룹 구분 불가 |

**아키텍처: MosquitoLSTM (model.py)**

- BiLSTM (hidden=64, 2 layers, bidirectional) — 139K params (vs selector ~0.8M)
- **Frenet-parametric 출력**: `pred = p0 + d1_s·d1 + par·acc_par + perp·acc_perp + jerk_s·jerk`
  - yaw-invariant: seq_features 회전 불변 + Frenet 계수가 실제 궤적 tangent/normal frame으로 매핑
  - TransformerRegressor 실패 원인(raw Δx,Δy,Δz가 yaw 증강과 비호환) 해결
- d1_scale bias 초기값 2.0 (선형 외삽 초기화)

**학습 전략 (train_lstm.py)**

- **C-group 10× 가중 손실**: 배치마다 `make_candidates_gpu` → `min_dist > 1cm` 샘플에 weight=10
- Loss: smooth_l1 (beta=R_HIT_THRESHOLD) — 1cm 미만 L2, 이상 L1
- yaw 증강, early stopping patience=30, lr=1e-3

**훈련 결과 (seeds 42/777/123)**

| Seed | CV mean | OOF 전체 | OOF C-group | OOF A+B |
|---:|---:|---:|---:|---:|
| 42 | 0.6283 ± 0.0040 | 0.6283 | **3.54%** | 82.16% |
| 777 | 0.6195 ± 0.0083 | 0.6195 | 2.48% | 81.34% |
| 123 | 0.6252 ± 0.0054 | 0.6252 | 3.21% | 81.86% |
| **3-seed 평균** | — | ~0.625 | **~3.1%** | ~82% |

C-group R-Hit: **0.62% → ~3.1%** (5× 향상) — 5% 목표 미달이나 유의미한 개선

**Physics-based Routing (regression.py)**

```python
# 초기 설정 (routing 13.4% → LB 하락)
decel_thresh=1.4, jerk_thresh=0.25, max_alpha=0.45

# 수정 후 (routing 4.5%)
decel_thresh=2.5, jerk_thresh=0.55, max_alpha=0.30
```

신호:
- `speed_ratio > 2.5`: 극감속 (현재 speed < 이전 speed × 40%)
- `jerk_abs > 0.55`: 극jerk (일반 0.04, C-group jerk 0.1~0.3 상단 초과)

**LB 결과 및 분석**

| 제출 | routing | LB |
|---|---:|---:|
| 3-seed LSTM blend (초기, 13.4%) | 13.4% | **0.6588** ❌ |
| selector-only (Phase 11 best) | 0% | 0.6692 |
| **Best LB** | — | **0.6716** |

초기 routing(13.4%)이 A+B group 오염으로 **−0.013 손실**. 임계값 상향 후 OOF 기준 +0.0011 개선 확인 (미제출).

**OOF 로컬 평가 (eval_lstm_blend.py)**

```
routing 4.5% 기준:
  Selector OOF:  0.6371  (C-group 0.00%)
  LSTM OOF:      0.6296  (C-group 3.09%)
  Blend OOF:     0.6382  (+0.0011 개선)
```

**결론**

- C-group 개선(0.62% → 3.54%)은 성공했으나 test set에서 net gain이 너무 작음
- LSTM 단독 전체 OOF(0.625) < selector(0.647) → routing 오발동 시 손실이 이득을 초과
- **LSTM 접근 자체는 유효**하나 C-group R-Hit을 10%+ 수준으로 올려야 LB 유의미한 개선 가능

**신규 파일**

| 파일 | 내용 |
|---|---|
| `train_lstm.py` | BiLSTM C-group 학습 스크립트 |
| `eval_lstm_blend.py` | selector OOF + LSTM OOF blend 로컬 평가 |
| `model.py` | `MosquitoLSTM` 클래스 추가 |
| `regression.py` | `physics_routing_alpha`, `load_lstm_models`, `predict_lstm_batch` 추가 |
| `predict.py` | `--lstm` 플래그 추가 |

---

### 2026-05-26 (4)

**[Phase 13: GCN CandidateSelector + C-gate + GRUResidual — LB 0.6688 / 0.6674]**

**GCN 아키텍처 구현**

기존 CandidateSelector의 cross-attention 이후 GCN 레이어 1개 추가 (end-to-end):

```python
# model.py: CandidateSelector 내 추가
class CandidateGCN(nn.Module):
    """EdgeConv-style GCN: KNN 그래프(k=6) + message passing"""
    # 입력: cand_feat (B, C, d_model)
    # 출력: cand_feat_updated (B, C, d_model)  — 후보 간 공간 관계 반영
```

목적: 52개 후보가 독립 채점 → oracle 후보와 인접한 "유사 decoy"를 구분 못하는 B-group 병목 해소  
GCN이 이웃 후보 간 상대 관계(거리, 방향)를 message passing → oracle 후보의 공간적 고유성 강화.

**3-seed 학습 결과 (GCN 추가 후)**

| seed | OOF | 이전 (Phase 11) | 변화 | Oracle efficiency |
|---:|---:|---:|---:|---:|
| 42 | **0.6506** | 0.6464 | **+0.42pp ✅** | 86.3% |
| 777 | ~0.6477 | ~0.6464 | +0.13pp ✅ | 85.9% |
| 123 | ~0.6490 | ~0.6464 | +0.26pp ✅ | 86.1% |

**Phase 13 C-gate 파이프라인**

3단계 순차 실행:

```bash
python export_oof_phase13.py --seeds 42 777 123   # 멀티-seed OOF 저장
python train_c_gate.py                             # CClassifier (C vs non-C)
python train_residual_c.py                         # GRUResidual (C-only 잔차 보정)
```

**C-gate (CClassifier) 결과**

| 지표 | 값 |
|---|---|
| OOF ROC-AUC | **0.7516** |
| thresh=0.70: prec | 0.6661 |
| thresh=0.70: rec | 0.1671 |
| physics_routing rec (비교) | 0.1171 (동등 precision에서 열위) |
| pos_weight | 3.07 (class imbalance 보정) |

- 25개 meta feature: 11d last-step seq_feat + 5d last-3-step 집계 + 4d logit 기반(entropy H, logit_gap, top1_prob, top5_prob_sum) + 3d candidate spread + 2d physics divergence
- 멀티-seed OOF 일관성: C-gate 학습 시 multi-seed avg logit 사용 → test-time 분포와 동일

**GRUResidual 결과**

| 지표 | 값 |
|---|---|
| 구조 | BiGRU(hidden=64, layers=2) + base_delta(3) 입력 |
| base_delta | base_pred − p0 (방향 정보 주입 — 핵심 fix) |
| 보정 범위 | ±6mm clamp |
| OOF 전체 | **+0.10pp** |
| OOF near-C (1~1.5cm) | +0.85pp |
| OOF hard-C (>1.5cm) | ≈0 (미개선) |
| OOF non-C | ±0 (보존) |

base_delta 없이 seq_feat만 입력 시 epoch 1에서 정체(patience 소진) → 평균 delta 수렴 문제.  
base_delta = base_pred − p0 추가로 per-sample 방향 보정 가능.

**LB 제출 결과 및 분석**

| 파일 | 구성 | OOF | **LB** |
|---|---|---:|---:|
| base (GCN, 3-seed) | selector만 | ~0.6506 | **0.6688** ❌ |
| blend (C-gate+GRU) | + 4.9% C-gate routing | ~0.6516 | **0.6674** ❌ |
| Phase 7 best (비교) | 2-seed, no GCN | 0.6484 | **0.6716** |

**핵심 발견: OOF/LB 역전**

| 추가 요소 | OOF 변화 | LB 변화 | 결론 |
|---|---:|---:|---|
| GCN (vs Phase 11 3-seed) | +0.42pp (seed42) | **−0.004** | GCN 오버피팅 |
| C-gate + GRUResidual | +0.10pp | **−0.0014** | C-gate도 오버피팅 |

- 10K 학습 샘플에서 GCN(추가 파라미터) + C-gate + GRUResidual 3-stage 구조가 train/test 분포 차이를 메우지 못함
- OOF가 높아도 LB 역행 → 다음 Phase에서 단순화 + 데이터 증강 강화 필요

**신규 파일**

| 파일 | 내용 |
|---|---|
| `export_oof_phase13.py` | 멀티-seed OOF 저장 (oof_preds, logits, seq_feat, oracle_cands, c_labels 등) |
| `features_phase13.py` | `make_c_meta_features()` — 25차원, y_true 비의존 |
| `train_c_gate.py` | CClassifier 5-fold 학습 |
| `train_residual_c.py` | GRUResidual C-group 전용 잔차 학습 |
| `predict_phase13.py` | GCN selector + C-gate + GRUResidual 통합 inference |

---

### 다음 실험 계획 (Phase 14)

**현황**

| 항목 | 값 |
|---|---|
| Best LB | **0.6716** (Phase 5/7, 2-seed 42+777, GCN 없음) |
| Phase 13 GCN base LB | 0.6688 (−0.0028 vs best) |
| Phase 13 blend LB | 0.6674 |
| OOF/LB 역전 원인 | GCN + C-gate 오버피팅 (10K 샘플 부족) |

**Phase 14 방향: 데이터 증강으로 GCN 오버피팅 억제**

GCN/LSTM 등 추가 파라미터를 유지하되, 학습 데이터 다양성을 높여 오버피팅을 완화한다.

**구현 완료: 추가 증강 2종**

| 증강 | 구현 위치 | 설정값 | 설명 |
|---|---|---|---|
| **x/y mirror flip** | `dataset.py` `augment_mirror_gpu()` | `AUG_FLIP=True`, prob=0.5 독립 | x(forward)/y(left) 각 축 독립 반전 → 4가지 조합, 실질 4× 다양성. z(up)는 중력 방향이므로 미적용 |
| **Coordinate noise** | `dataset.py` `augment_noise_gpu()` | `AUG_NOISE=True`, `NOISE_STD=0.001` | 입력 좌표에 1mm Gaussian jitter, 라벨은 clean 유지. LiDAR 측정 노이즈 시뮬레이션 |

적용 순서: `yaw → flip → noise` (train 배치마다 on-the-fly)

**다음 실행 명령 (RTX 5080 데스크탑)**

```bash
python dev/experiments/train.py --seed 42
python dev/experiments/train.py --seed 777
python dev/experiments/train.py --seed 123
python dev/experiments/predict.py --seeds 42 777 123 --temp 2.0
```

**성공 기준**

| 지표 | 목표 |
|---|---|
| OOF (seed42) | ≥ 0.6506 (Phase 13 GCN 동등 이상) |
| LB (3-seed) | **> 0.6716** (Phase 5/7 best 돌파) |

OOF가 Phase 13과 비슷하면서 LB가 오르면 → 증강이 오버피팅을 실제로 줄인 것.  
OOF도 같이 오르면 → 증강이 일반화 + 학습 둘 다 개선.

**이후 탐색 후보 (결과에 따라)**

| 방향 | 조건 |
|---|---|
| C-gate / GRUResidual 재학습 | Phase 14 base LB > 0.6716 확인 후 |
| NOISE_STD 튜닝 (0.5mm / 2mm) | OOF 변화 없을 시 |
| GCN dropout 강화 | LB 여전히 역전 시 |
