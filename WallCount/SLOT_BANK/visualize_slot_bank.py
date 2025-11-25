# visualize_slot_bank_pkl.py
# 作用：可视化 slot_bank.pkl 中的 (u,v) 聚类中心，并导出为 CSV

import os, pickle, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKL_PATH = "./slot_bank.pkl"
OUT_DIR  = "./slot_bank_vis"
os.makedirs(OUT_DIR, exist_ok=True)

with open(PKL_PATH, "rb") as f:
    obj = pickle.load(f)

# 期望键名：centers_uv -> (K,2)
uv = np.array(obj["centers_uv"], dtype=np.float32)
assert uv.ndim == 2 and uv.shape[1] == 2, f"bad centers_uv shape: {uv.shape}"
K = uv.shape[0]

# 2D 散点 + 标注 cluster id
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(uv[:,0], uv[:,1], s=60, alpha=0.9)
for i, (u, v) in enumerate(uv):
    ax.text(u, v, str(i), fontsize=8, ha="center", va="center")
ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("u (tangent normalized)"); ax.set_ylabel("v (vertical normalized)")
ax.set_title(f"Slot Bank Centers (K={K})")
plt.tight_layout()
png_path = os.path.join(OUT_DIR, "uv_centers.png")
plt.savefig(png_path, dpi=200); plt.close(fig)

# 导出 CSV 方便其他工具复用
csv_path = os.path.join(OUT_DIR, "slot_bank_uv.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cluster_id", "u", "v"])
    for i, (u, v) in enumerate(uv):
        w.writerow([i, float(u), float(v)])

print(f"[OK] centers plotted → {png_path}")
print(f"[OK] CSV exported → {csv_path}")
