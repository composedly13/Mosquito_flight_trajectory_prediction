"""
Predict grid: ensemble weight × temperature × TopK 조합 grid 탐색.
학습 없이 기존 seed42+seed777 모델로 제출 파일 다수 생성.

Test inference (submission):
  python predict_grid.py                               # 추천 grid 전체
  python predict_grid.py --mode test                  # 동일
  python predict_grid.py --mode test --quick           # 추천 5종만

OOF inference (val R-Hit 확인):
  python predict_grid.py --mode oof
  python predict_grid.py --mode oof --quick

Custom single run:
  python predict_grid.py --seeds 42 777 --weights 0.4 0.6 --temp 2.0 --topk 10
  python predict_grid.py --seeds 42 777 --temps 1.5 2.0 --topk 10  # seed별 temp

출력 파일명 규칙:
  submission_s42t20_s777t25_w04_06_top10.csv
  (seed, temp×10, weight×10, topk)
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TEST_DIR, TRAIN_DIR, LABELS_PATH, SUBMISSION_PATH,
    OUTPUT_DIR, SEED, N_FOLDS, BATCH_SIZE,
)
from dataset import load_all
from model import CandidateSelector, selector_predict
from candidates import make_candidates_gpu, make_seq_features_gpu, make_cand_features_gpu
import hashlib


# ─────────────────────────────────────────────────────────────────────────────
# Grid 설정 (baseline: 추천 제출 후보)
# ─────────────────────────────────────────────────────────────────────────────

RECOMMENDED_GRID = [
    # (seeds, weights, temps, topk, label)
    # seeds: [42, 777]
    # weights: [w42, w777]  — None = 동일 (0.5:0.5)
    # temps: [t42, t777]   — scalar = 동일 temp
    ([42, 777], [0.5, 0.5], 2.0,        10, "baseline"),
    ([42, 777], [0.4, 0.6], 2.0,        10, "w0406_t20_top10"),
    ([42, 777], [0.3, 0.7], 2.0,        10, "w0307_t20_top10"),
    ([42, 777], [0.6, 0.4], 2.0,        10, "w0604_t20_top10"),
    ([42, 777], [0.5, 0.5], 2.0,         7, "w0505_t20_top7"),
    ([42, 777], [0.5, 0.5], 2.0,        12, "w0505_t20_top12"),
    ([42, 777], [0.5, 0.5], 1.5,        10, "w0505_t15_top10"),
    ([42, 777], [0.5, 0.5], 2.5,        10, "w0505_t25_top10"),
    # seed별 다른 temp
    ([42, 777], [0.5, 0.5], [1.5, 2.0], 10, "s42t15_s777t20_top10"),
    ([42, 777], [0.5, 0.5], [2.0, 1.5], 10, "s42t20_s777t15_top10"),
    ([42, 777], [0.5, 0.5], [2.0, 2.5], 10, "s42t20_s777t25_top10"),
    ([42, 777], [0.4, 0.6], [2.0, 2.5], 10, "s42t20_s777t25_w0406_top10"),
    ([42, 777], [0.5, 0.5], [2.5, 2.0], 10, "s42t25_s777t20_top10"),
]

QUICK_GRID = [
    ([42, 777], [0.4, 0.6], 2.0,        10, "w0406_t20_top10"),
    ([42, 777], [0.3, 0.7], 2.0,        10, "w0307_t20_top10"),
    ([42, 777], [0.5, 0.5], [1.5, 2.0], 10, "s42t15_s777t20_top10"),
    ([42, 777], [0.5, 0.5], [2.0, 2.5], 10, "s42t20_s777t25_top10"),
    ([42, 777], [0.5, 0.5], 2.0,         7, "w0505_t20_top7"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def fold_id(sample_id: str, n_folds: int = N_FOLDS) -> int:
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % n_folds


def r_hit(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= 0.01))


def load_selectors(seeds: list, device: torch.device) -> dict[int, list]:
    """Returns {seed: [fold0_model, ..., fold4_model]}"""
    seed_models = {}
    for seed in seeds:
        seed_dir = OUTPUT_DIR / f"seed{seed}"
        models = []
        for fold in range(N_FOLDS):
            path = seed_dir / f"selector_fold{fold}.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing: {path}\n  Run: python train.py --seed {seed}"
                )
            m = CandidateSelector().to(device)
            m.load_state_dict(torch.load(path, map_location=device), strict=False)
            m.eval()
            models.append(m)
        seed_models[seed] = models
        print(f"  seed{seed}: {len(models)} fold models loaded")
    return seed_models


def _temps_for(temps, idx: int) -> float:
    """temps가 scalar면 모든 seed에 동일, list면 idx번째 값 사용."""
    if isinstance(temps, (int, float)):
        return float(temps)
    return float(temps[idx])


def make_sub_name(seeds, weights, temps, topk, label=None):
    """자동 파일명 생성."""
    if label:
        return f"submission_grid_{label}.csv"
    parts = []
    for i, s in enumerate(seeds):
        t = _temps_for(temps, i)
        w = weights[i]
        parts.append(f"s{s}t{int(t*10):02d}w{int(w*10):02d}")
    parts.append(f"top{topk}")
    return "submission_grid_" + "_".join(parts) + ".csv"


# ─────────────────────────────────────────────────────────────────────────────
# 추론 코어
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_item(
    seed_models: dict,
    seeds: list,
    weights: list,
    temps,       # float or list[float]
    topk: int,
    data: tuple, # (ids, coords)
    device: torch.device,
    oof_labels: np.ndarray = None,
    oof_fold_ids: np.ndarray = None,
) -> tuple[np.ndarray, float | None]:
    """
    단일 grid 설정으로 추론.
    oof_labels가 None이 아니면 OOF R-Hit도 계산 (학습 데이터 기준).
    """
    ids, coords = data
    N = len(ids)

    # ── Test inference ──────────────────────────────────────────────────────
    # 각 seed의 전체 fold 평균 logit을 seed별 weight로 혼합
    all_preds = np.zeros((N, 3), dtype=np.float32)

    # 분모 정규화
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    for s_idx, (seed, w) in enumerate(zip(seeds, norm_weights)):
        seed_temp = _temps_for(temps, s_idx)
        models = seed_models[seed]
        # seed 내 5 fold 모델 평균 logit
        seed_logits_sum = None
        for batch_start in range(0, N, BATCH_SIZE):
            pass  # placeholder: do below in batch loop
        # batch loop for this seed
        seed_preds_batches = []
        seed_logits_batches = []
        for start in range(0, N, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N)
            c = torch.tensor(coords[start:end]).to(device)
            with torch.no_grad():
                cands_t = make_candidates_gpu(c)
                seq_t   = make_seq_features_gpu(c)
                cand_t  = make_cand_features_gpu(c, cands_t)
                avg_logits = sum(m(seq_t, cand_t) for m in models) / len(models)
                pred = selector_predict(avg_logits, cands_t, topk=topk, temp=seed_temp)
            seed_preds_batches.append(pred.cpu().numpy())
            seed_logits_batches.append(avg_logits.cpu().numpy())
        seed_preds = np.concatenate(seed_preds_batches, axis=0)  # (N, 3)
        all_preds += w * seed_preds

    # ── OOF R-Hit (선택적) ─────────────────────────────────────────────────
    oof_hit = None
    if oof_labels is not None and oof_fold_ids is not None:
        # OOF: 각 샘플을 자신이 val fold인 모델로 예측
        oof_preds = np.zeros((N, 3), dtype=np.float32)
        for s_idx, (seed, w) in enumerate(zip(seeds, norm_weights)):
            seed_temp = _temps_for(temps, s_idx)
            models = seed_models[seed]
            seed_oof = np.zeros((N, 3), dtype=np.float32)
            for fold_idx, m in enumerate(models):
                val_mask = (oof_fold_ids == fold_idx)
                if not val_mask.any():
                    continue
                val_coords = coords[val_mask]
                val_preds_list = []
                for start in range(0, len(val_coords), BATCH_SIZE):
                    end = min(start + BATCH_SIZE, len(val_coords))
                    c = torch.tensor(val_coords[start:end]).to(device)
                    with torch.no_grad():
                        cands_t = make_candidates_gpu(c)
                        seq_t   = make_seq_features_gpu(c)
                        cand_t  = make_cand_features_gpu(c, cands_t)
                        logits  = m(seq_t, cand_t)
                        pred    = selector_predict(logits, cands_t, topk=topk, temp=seed_temp)
                    val_preds_list.append(pred.cpu().numpy())
                seed_oof[val_mask] = np.concatenate(val_preds_list, axis=0)
            oof_preds += w * seed_oof
        oof_hit = r_hit(oof_preds, oof_labels)

    return all_preds, oof_hit


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prediction grid: ensemble weight × temp × TopK")
    parser.add_argument("--mode", choices=["test", "oof", "both"], default="test",
                        help="'test': submit CSV only / 'oof': OOF R-Hit only / 'both': 둘 다")
    parser.add_argument("--quick", action="store_true",
                        help="추천 5종만 빠르게 실행")
    # Custom single run
    parser.add_argument("--seeds",   type=int, nargs="+", default=None)
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Each seed weight (sum will be normalized)")
    parser.add_argument("--temp",    type=float, default=None,
                        help="단일 temperature (모든 seed 공통)")
    parser.add_argument("--temps",   type=float, nargs="+", default=None,
                        help="Seed별 temperature (e.g. --temps 1.5 2.0 for seed42/777)")
    parser.add_argument("--topk",    type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 어떤 grid를 돌릴지 결정 ──────────────────────────────────────────────
    custom_mode = (args.seeds is not None)
    if custom_mode:
        if args.temps and args.temp:
            parser.error("--temp과 --temps를 동시에 사용할 수 없습니다.")
        seeds   = args.seeds
        weights = args.weights if args.weights else [1.0] * len(seeds)
        temps   = args.temps if args.temps else (args.temp if args.temp else 2.0)
        topk    = args.topk if args.topk else 10
        grid    = [(seeds, weights, temps, topk, None)]
    else:
        grid = QUICK_GRID if args.quick else RECOMMENDED_GRID

    # ── 필요한 seed 목록 추출 ────────────────────────────────────────────────
    all_seeds = set()
    for item in grid:
        all_seeds.update(item[0])
    all_seeds = sorted(all_seeds)

    seed_models = load_selectors(all_seeds, device)

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    do_oof  = args.mode in ("oof", "both")
    do_test = args.mode in ("test", "both")

    test_ids = test_coords = None
    if do_test:
        test_ids, test_coords, _ = load_all(TEST_DIR)
        print(f"Test samples: {len(test_ids)}")

    train_ids = train_coords = train_labels = fold_ids_arr = None
    if do_oof:
        train_ids, train_coords, train_labels = load_all(TRAIN_DIR, LABELS_PATH)
        fold_ids_arr = np.array([fold_id(i) for i in train_ids])
        print(f"Train samples: {len(train_ids)}")

    # ── Grid 실행 ────────────────────────────────────────────────────────────
    sub_tmpl = pd.read_csv(SUBMISSION_PATH, index_col="id")
    results  = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_dir = OUTPUT_DIR / "grid"
    grid_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Grid: {len(grid)} 항목  |  mode={args.mode}  |  quick={args.quick}")
    print('='*60)

    for item in tqdm(grid, desc="Grid"):
        seeds_i, weights_i, temps_i, topk_i, label_i = item
        fname = make_sub_name(seeds_i, weights_i, temps_i, topk_i, label_i)
        temps_str = (f"{temps_i}" if isinstance(temps_i, float)
                     else "/".join(f"{t}" for t in temps_i))
        print(f"\n[{fname}]  seeds={seeds_i}  w={weights_i}  temp={temps_str}  topk={topk_i}")

        oof_hit = None
        if do_oof:
            _, oof_hit = run_grid_item(
                seed_models, seeds_i, weights_i, temps_i, topk_i,
                (train_ids, train_coords), device,
                oof_labels=train_labels, oof_fold_ids=fold_ids_arr,
            )
            print(f"  OOF R-Hit: {oof_hit:.4f}")

        if do_test:
            preds, _ = run_grid_item(
                seed_models, seeds_i, weights_i, temps_i, topk_i,
                (test_ids, test_coords), device,
            )
            df = pd.DataFrame(preds, index=test_ids, columns=sub_tmpl.columns)
            df.index.name = "id"
            out_path = grid_dir / fname
            df.to_csv(out_path)
            print(f"  Saved: {out_path}")

        results.append({
            "file": fname,
            "seeds": str(seeds_i),
            "weights": str(weights_i),
            "temps": str(temps_i),
            "topk": topk_i,
            "oof_hit": f"{oof_hit:.4f}" if oof_hit else "—",
        })

    # ── 결과 요약 ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("결과 요약")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x["oof_hit"] if x["oof_hit"] != "—" else "0", reverse=True):
        print(f"  OOF={r['oof_hit']:>6}  {r['file']}")
    print(f"\n출력 디렉토리: {grid_dir}")


if __name__ == "__main__":
    main()
