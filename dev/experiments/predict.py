import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import *
from preprocess import MosquitoDataset
from model import PhysicsCorrector


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MosquitoDataset(TEST_DIR)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PhysicsCorrector().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for x, physics in loader:
            pred = model(x.to(device), physics.to(device)).cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds)

    sample = pd.read_csv(SUBMISSION_PATH)
    test_ids = sorted([p.stem for p in TEST_DIR.glob("*.csv")])
    submission = pd.DataFrame({
        "id": test_ids,
        "x":  preds[:, 0],
        "y":  preds[:, 1],
        "z":  preds[:, 2],
    })
    submission.to_csv("submissions/submission.csv", index=False)
    print("Saved: submissions/submission.csv")


if __name__ == "__main__":
    predict()
