from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os

root_path = "/home/zengy2/isilon/MRI-1-30-2025"
OUT = "dicom_table.csv"

all_folders = os.listdir(root_path)
test_dates = []
patient_IDs = []

rows = []

for folder in os.listdir(root_path):
    cur_path = os.path.join(root_path, folder)
    if not os.path.isdir(cur_path):
        continue

    ds = None
    for f in os.listdir(cur_path):
        if f.startswith('.'):
            continue
        try:
            ds = pydicom.dcmread(
                os.path.join(cur_path, f),
                stop_before_pixels=True
            )
            break
        except Exception:
            continue

    if ds is None:
        continue

    rows.append({
        "uid": folder,
        "patient_ID": ds.get("PatientID"),
        "test_date": (
            ds.get("AcquisitionDate")
            or ds.get("SeriesDate")
            or ds.get("StudyDate")
        )
    })

df = pd.DataFrame(rows)

df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")

gap = (
    df.dropna(subset=["test_date"])
      .groupby("patient_ID")["test_date"]
      .agg(first="min", last="max", n_dates="nunique")
      .assign(gap_days=lambda x: (x["last"] - x["first"]).dt.days)
      .reset_index()
)

df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
d = df.dropna(subset=["test_date"]).copy()

def summarize_patient(g):
    x = g["test_date"].drop_duplicates().sort_values()
    n = x.size

    max_gap = (x.iloc[-1] - x.iloc[0]).days if n >= 1 else np.nan
    min_gap = x.diff().dt.days.min() if n >= 2 else np.nan

    return pd.Series({
        "first": x.iloc[0].date().isoformat() if n >= 1 else None,
        "last":  x.iloc[-1].date().isoformat() if n >= 1 else None,
        "n_dates": n,
        "max_gap_days": max_gap,
        "min_gap_days": min_gap,
        "n_rows": len(g)
    })

gap_df = (
    d.sort_values(["patient_ID", "test_date"])
     .groupby("patient_ID", group_keys=False)
     .apply(summarize_patient)
     .reset_index()
)

gap_df.head()

fig, ax_left = plt.subplots(figsize=(8, 5))

ax_left.plot(
    gap_df.index,
    gap_df["min_gap_days"],
    marker="o",
    linestyle=""
)
ax_left.set_ylabel("Minimum gap (days)")
ax_left.axhline(90, linestyle="--", color="red")
ax_left.axhline(180, linestyle="--", color="red")

ax_right = ax_left.twinx()
ax_right.plot(
    gap_df.index,
    gap_df["n_dates"],
    color="gray",
    alpha=0.6
)
ax_right.set_ylabel("Number of scan dates")

ax_left.set_xlabel("Patient index")
plt.title("Patient-level minimum gap and scan count")
ax_left.text(
    0.02, 0.95,
    "90 ≤ min gap ≤ 180 days: 411",
    transform=ax_left.transAxes,
    fontsize=11,
    va="top",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
)
plt.tight_layout()
plt.show()
