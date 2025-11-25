# slot_bank_build.py
# 作用：从真实数据中统计 (u,v) 偏移分布，并聚类为 K 个锚位，保存到 slot_bank.pkl

import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ALL_GRAPH_DIR = "./ALL_graph"
K_CLUSTERS = 8   # 你可以改成 6/8/10 看数据规模
UP = np.array([0., 0., 1.], dtype=np.float32)

# 尽量鲁棒地取列
def pick(df, *cands):
    cols = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns: return c
        if c.lower() in cols: return cols[c.lower()]
    raise RuntimeError(f"列缺失：{cands}")

def wall_frame(w_center, b_center):
    # 外法线 ~ 从楼中心指向墙中心（在 XY 平面）
    n = w_center - b_center
    n[2] = 0.0
    n_norm = np.linalg.norm(n) + 1e-8
    n = n / n_norm
    # 水平切向：up × n
    t1 = np.cross(UP, n)
    if np.linalg.norm(t1) < 1e-8:
        # 极端情况（n与up共线）给一个默认方向
        t1 = np.array([1., 0., 0.], dtype=np.float32)
    else:
        t1 = t1 / (np.linalg.norm(t1) + 1e-8)
    t2 = UP.copy()
    return n, t1, t2

def collect_uv(all_graph_dir=ALL_GRAPH_DIR):
    uv_rows = []
    node_files = sorted(glob.glob(os.path.join(all_graph_dir, "*_nodes.csv")))
    for nf in node_files:
        ef = nf.replace("_nodes.csv", "_edges.csv")
        if not os.path.exists(ef):
            continue

        dn = pd.read_csv(nf)
        de = pd.read_csv(ef)

        c_id   = pick(dn, "node_id")
        c_type = pick(dn, "node_type")
        c_bx   = pick(dn, "bounding_box_x")
        c_by   = pick(dn, "bounding_box_y")
        c_bz   = pick(dn, "bounding_box_z")
        c_cx   = pick(dn, "coordinate_x")
        c_cy   = pick(dn, "coordinate_y")
        c_cz   = pick(dn, "coordinate_z")

        # 建邻接
        ids = dn[c_id].astype(int).tolist()
        nbrs = {i: [] for i in ids}

        e_src = pick(de, "source")
        e_tgt = pick(de, "target")
        for _, r in de.iterrows():
            u, v = int(r[e_src]), int(r[e_tgt])
            if u in nbrs and v in nbrs:
                nbrs[u].append(v); nbrs[v].append(u)

        # 楼中心
        b_center = np.array([
            dn[c_cx].mean(), dn[c_cy].mean(), dn[c_cz].mean()
        ], dtype=np.float32)

        # 行扫描
        row_map = {int(r[c_id]): r for _, r in dn.iterrows()}

        for wid in ids:
            wr = row_map[wid]
            wtype = int(wr[c_type])
            if wtype not in (1, 2):  # 只取墙/骨架
                continue

            w_center = np.array([float(wr[c_cx]), float(wr[c_cy]), float(wr[c_cz])], np.float32)
            w_bbox   = np.array([float(wr[c_bx]), float(wr[c_by]), float(wr[c_bz])], np.float32)

            n, t1, t2 = wall_frame(w_center, b_center)
            half_h = max(w_bbox[2]*0.5, 1e-3)   # 竖直半高
            half_w = max(max(w_bbox[0], w_bbox[1])*0.5, 1e-3)  # 水平近似半宽

            # 遍历邻居中的 type=3 作为真实子件
            for cid in nbrs[wid]:
                cr = row_map.get(cid, None)
                if cr is None: continue
                if int(cr[c_type]) != 3: continue

                c_center = np.array([float(cr[c_cx]), float(cr[c_cy]), float(cr[c_cz])], np.float32)
                d = c_center - w_center
                u = float(np.dot(d, t1) / half_w)
                v = float(np.dot(d, t2) / half_h)
                uv_rows.append([wtype, half_w*2, half_h*2, u, v])

    if len(uv_rows) == 0:
        raise RuntimeError("没有采集到 (u,v) 样本，请检查数据。")
    return np.array(uv_rows, dtype=np.float32)

if __name__ == "__main__":
    uv = collect_uv(ALL_GRAPH_DIR)
    km = KMeans(n_clusters=K_CLUSTERS, random_state=0).fit(uv[:, -2:])
    centers = km.cluster_centers_.astype(np.float32)  # (K, 2)
    os.makedirs(".", exist_ok=True)
    with open("./slot_bank.pkl", "wb") as f:
        pickle.dump({"centers_uv": centers}, f)
    print(f"slot_bank.pkl saved. K={len(centers)} centers.")
