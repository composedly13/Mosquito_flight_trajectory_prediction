# Experiments Log

모든 실험은 **Phase 5 재확립 baseline (LB 0.6732)** 을 기준으로 비교합니다.

---

## ★ Current Baseline (Phase 5 재확립)

| 항목 | 값 |
|---|---|
| **LB** | **0.6732** ✅ (2026-05-27) |
| Seeds | 42 (patience=40) + 777 (patience=80) |
| Candidates | Smart 50-cand |
| Oracle | 77.02% |
| OOF seed42 | 0.6478 (std=0.0062) |
| OOF seed777 | 0.6479 (std=0.0012, 매우 안정) |
| Selector efficiency | 84.1% |
| Loss | soft-CE + PW×0.25 + LML×0.10 |
| temp | 2.0 |
| Augmentation | yaw-only |
| CAND_DIM | 10 (family_id 없음) |

```bash
# 재현 명령어
python dev/experiments/train.py --seed 42  --patience 40
python dev/experiments/train.py --seed 777 --patience 80
python dev/experiments/predict.py --seeds 42 777
```

---

## 실험 히스토리

### Phase 1~4 (원본 아키텍처 단계)

| Phase | 핵심 변경 | LB | OOF | 비고 |
|---|---|---|---|---|
| Phase 1 | 35-cand Attn-GRU 기준 | — | 0.6428 | 원본 참고 솔루션 PB=0.6822 |
| Phase 2 | 50-cand + d128+yaw | — | 0.6401~0.6461 | LML grid 탐색 |
| Phase 3 | Smart 50-cand (G2) | — | 0.6475 | Oracle 77.02% 달성 |
| Phase 4 | RegMLP entropy blend | 0.6716 | 0.6484 | β=0.0(selector-only) 최적 확인 |

### Phase 5 재확립 (Current Baseline)

**Parent**: 없음 (전체 기준선)

LB 0.6716 달성 이후 GCN/C-gate 실험이 OOF는 소폭 개선(+0.42pp)됐으나 LB 기여 없음 확인.
Phase 5 Smart 50-cand 2-seed로 복귀, cuDNN 비결정성으로 G2 완전 재현 불가이나
새 학습(seed42 P=40, seed777 P=80)으로 LB 0.6732 달성.

```
seed42  OOF=0.6478, std=0.0062  (P=40)
seed777 OOF=0.6479, std=0.0012  (P=80, Fold5 0.6355→0.6461 회복)
2-seed ensemble → LB 0.6732 ✅ (+0.0016 vs 0.6716)
```

---

### Phase 6~14 (실험 기록 — LB baseline 미달)

| Phase | 핵심 변경 | LB | OOF | vs Baseline |
|---|---|---|---|---|
| Phase 6-1 | 3-seed 42+123+777 (config 혼재) | 0.6690 | 0.6493 | ❌ 혼재 |
| Phase 6-2 | 3-seed (orig 50, LML=0.05, 일관) | 0.6685 | 0.6479 | ❌ −0.47pp LB |
| **Phase 7** | 52-cand (latency_s075/080 추가) | **0.6712** | 0.6510 | ❌ −0.0020 LB |
| Phase 7-2 | 52-cand + B-group focal×2.0 | — | 악화 | ❌ oracle rank 악화 |
| Phase 8 | 54-cand + cand_self_attn + family feat | **0.6712** | 0.6474 | ❌ 동등 |
| Phase 9 | 60-cand + entropy penalty | — | 0.6445 | ❌ −0.29pp OOF |
| Phase 10 | 52-cand + BCE + family cls | — | 0.6438 | ❌ BCE가 oracle rank 파괴 |
| **Phase 11** | 52-cand + Aux Reg Head (LML×0.05, temp=1.0) | **0.6692** | 0.6464 | ❌ −0.0040 LB |
| Phase 12 | BiLSTM C-group routing 13.4% | 0.6588 | ~0.625 | ❌ A+B 오염 |
| Phase 13 | GCN(EdgeConv k=6) + C-gate + GRUResidual | 0.6688/0.6674 | 0.6506~0.6516 | ❌ LB 기여 없음 |
| Phase 14 | x/y mirror flip + coordinate noise aug | — | — | ❌ 미검증, Phase 5 복귀 |

---

## 이후 실험 계획 (LB 0.6732 기준)

이하 실험은 모두 Phase 5 재확립 (LB 0.6732)을 **parent**로 합니다.
실험 결과가 LB > 0.6732를 기록할 때만 새 baseline으로 승격합니다.

### Next Experiments

| 실험 | 핵심 변경 | 예상 OOF | 비고 |
|---|---|---|---|
| Phase 15-A | C-gate (Phase 13 기반) — Smart 50-cand parent로 재학습 | 0.649~0.651 | Phase 13 OOF +0.10pp 확인 필요 |
| Phase 15-B | GCN (EdgeConv) — Smart 50-cand parent로 재학습 | 0.648~0.652 | Phase 13 단독 OOF +0.42pp 재확인 |
| Phase 15-C | BiLSTM 재설계 — routing 범위 재검토 (A/B 오염 방지) | 0.650~0.654 | Phase 12 실패 원인 분석 후 |
| Phase 16 | 3-seed (42+777+X) — seed 추가 | — | seed123 불안정 확인됨, 대안 seed 탐색 |

### 실험 추가 원칙

1. 새 실험 시작 전 이 파일에 항목 추가 (`Parent: Phase 5 재확립, LB 0.6732`)
2. OOF 결과 즉시 기록 (cv mean ± std, seed, patience)
3. LB 제출 후 결과 기록 (LB 값, vs baseline diff)
4. 실패 원인 분석 필수 (`❌` 항목도 중요한 실험 자산)
