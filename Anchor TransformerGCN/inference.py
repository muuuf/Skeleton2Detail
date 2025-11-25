# full_inference.py
#
# 最终推理管线：
# 1. 对每栋楼的骨架墙节点 (type=1/2)，用 WallCountModel 预测每面墙需要多少个子构件
# 2. 按照预测的数量，为该墙生成对应数量的slot（每个slot用一个anchor_offset来区分）
# 3. 对每个slot，使用 AnchorTransformerGCN 回归该子构件的几何(bbox_x,y,z, coord_x,y,z)
# 4. 组装成完整预测图，输出到 ./final_infer_output/<building>_pred_nodes.csv / _pred_edges.csv
#
# 依赖：
#   pip install torch torch_geometric pandas numpy networkx matplotlib
#
# 注意：
#   - WallCountModel 是 TransformerEncoder + MLP，用来预测 child_count
#   - AnchorTransformerGCN 是骨架+slot → 几何回归模型
#   - 我们假设两个checkpoint都已经训练好
#
# 使用：
#   python full_inference.py
#
# 输出：
#   ./final_infer_output/model_001_pred_nodes.csv
#   ./final_infer_output/model_001_pred_edges.csv
#
# 之后直接用 visualize_3d.py (你上一个版本) 指向这些csv即可画图


import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data
from anchor_model import AnchorTransformerGCN  # 你已有的几何回归模型类

############################################
# 路径 & 推理配置
############################################

ALL_GRAPH_DIR = "./ALL_graph"

COUNT_CKPT_PATH  = "./wall_count_checkpoints/wallcount_epoch_300.pt"   # 改成你最好的一轮
ANCHOR_CKPT_PATH = "./anchor_checkpoints/model_epoch_050.pt"          # 改成你最好的一轮

OUTPUT_DIR = "./final_infer_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 这个offset列表用于把同一面墙上的多个子构件slot分散开
# 注意：真实项目里你会用更聪明的offset生成逻辑（比如沿墙分布）
DEFAULT_OFFSETS = [
    np.array([ 0.0,  0.0,  0.0], dtype=np.float32),
    np.array([ 1.0,  0.0,  0.0], dtype=np.float32),
    np.array([-1.0,  0.5,  0.0], dtype=np.float32),
    np.array([ 0.5, -0.5,  0.0], dtype=np.float32),
    np.array([ 0.0,  1.0,  0.0], dtype=np.float32),
]

SKELETON_TYPES = {1, 2}
CHILD_TYPE = 3

IN_DIM_WALL     = 7    # [node_type, bbox_x,y,z, coord_x,y,z]
HIDDEN_DIM_WALL = 256  # must match training
NHEAD           = 4    # must match training
NUM_LAYERS      = 2    # must match training

ANCHOR_IN_DIM   = 10   # [parent_full(7) + offset(3)]
ANCHOR_HIDDEN   = 256  # must match training
ANCHOR_OUT_DIM  = 6    # [bbox_x,bbox_y,bbox_z, coord_x,coord_y,coord_z]


############################################
# WallCountModel 复刻（跟训练时保持一致）
############################################

class WallCountModel(nn.Module):
    """
    TransformerEncoder + MLP decoder，用于预测每个墙节点的child_count
    forward输入:
        feats_pad: [B,L,7]
        mask_pad:  [B,L] bool (True=有效，False=padding)
    输出:
        pred_count: [B,L] float (回归出来的子构件数量)
    """

    def __init__(self,
                 in_dim=IN_DIM_WALL,
                 hidden_dim=HIDDEN_DIM_WALL,
                 nhead=NHEAD,
                 num_layers=NUM_LAYERS):
        super().__init__()

        self.in_proj = nn.Linear(in_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            batch_first=True,
            activation="relu"
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats_pad, mask_pad):
        # feats_pad: [B,L,7]
        # mask_pad:  [B,L] bool
        h = self.in_proj(feats_pad)  # [B,L,H]

        # src_key_padding_mask=True 意味着这是padding，需要mask掉
        src_key_padding_mask = (~mask_pad)  # padding=True
        h_enc = self.encoder(
            h,
            src_key_padding_mask=src_key_padding_mask
        )  # [B,L,H]

        out = self.decoder_head(h_enc).squeeze(-1)  # [B,L]
        return out


############################################
# 图数据读入 & 结构构建工具
############################################

def load_building_graph(nodes_csv, edges_csv):
    """
    读取原始图 (model_xxx_nodes.csv/model_xxx_edges.csv)
    返回:
      G_full: nx.Graph
        每个节点:
          node_type
          features_full: [7] float32
          features_pred: [6] float32
    """
    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edges_csv)

    df_nodes = df_nodes.rename(columns=lambda c: c.strip())
    df_edges = df_edges.rename(columns=lambda c: c.strip())

    all_cols = list(df_nodes.columns)
    if "node_id" not in all_cols or "node_type" not in all_cols:
        raise RuntimeError(f"{nodes_csv}: 需要 node_id / node_type 列")

    node_id_col = "node_id"
    node_type_col = "node_type"
    rest_cols = [c for c in all_cols if c not in [node_id_col, node_type_col]]
    if len(rest_cols) < 6:
        raise RuntimeError(
            f"{nodes_csv}: 至少需要 node_id,node_type 之后的6列(bbox3+coord3)，但只有 {len(rest_cols)} 列"
        )
    bbox_coord_cols = rest_cols[:6]

    G = nx.Graph()

    for _, row in df_nodes.iterrows():
        nid = int(row[node_id_col])
        ntype = int(row[node_type_col])

        bbox_x = float(row[bbox_coord_cols[0]])
        bbox_y = float(row[bbox_coord_cols[1]])
        bbox_z = float(row[bbox_coord_cols[2]])

        cx = float(row[bbox_coord_cols[3]])
        cy = float(row[bbox_coord_cols[4]])
        cz = float(row[bbox_coord_cols[5]])

        features_full = np.array([
            ntype,
            bbox_x, bbox_y, bbox_z,
            cx, cy, cz
        ], dtype=np.float32)

        features_pred = np.array([
            bbox_x, bbox_y, bbox_z,
            cx, cy, cz
        ], dtype=np.float32)

        G.add_node(
            nid,
            node_type=ntype,
            features_full=features_full,
            features_pred=features_pred
        )

    if "source" not in df_edges.columns or "target" not in df_edges.columns:
        raise RuntimeError(f"{edges_csv}: 缺少 source / target 列")

    for _, row in df_edges.iterrows():
        u = int(row["source"])
        v = int(row["target"])
        if u in G.nodes and v in G.nodes:
            G.add_edge(u, v)

    return G


def build_skeleton_subgraph(G_full):
    """
    取出骨架部分(所有 type in {1,2} 的节点)
    返回：
      skel_nodes:  list[int] (升序)
      skel_feats:  dict[node_id] -> features_full(7)
      skel_edges:  list[(u,v)] 仅骨架之间的边(无向对)
    """
    skel_nodes = [n for n, attr in G_full.nodes(data=True)
                  if attr["node_type"] in SKELETON_TYPES]
    skel_nodes.sort()

    skel_feats = {n: G_full.nodes[n]["features_full"] for n in skel_nodes}

    skel_edges = []
    for u, v in G_full.edges():
        if u in skel_feats and v in skel_feats:
            skel_edges.append((u, v))

    return skel_nodes, skel_feats, skel_edges


############################################
# 准备 WallCountModel 的输入
############################################

def build_wall_count_batch_from_skeleton(skel_nodes, skel_feats):
    """
    根据一栋楼的骨架节点，构造 WallCountModel 的输入：
      feats_pad: [1, L, 7]
      mask_pad:  [1, L]
    L = 骨架墙节点数量
    节点顺序 = skel_nodes 升序 (和训练时一致)

    返回:
      feats_1L7 (torch.float32)
      mask_1L  (torch.bool)
      ordered_node_ids (list[int])  # 顺序对应L维度，用于回填预测
    """
    ordered_node_ids = list(skel_nodes)  # already sorted
    feats_list = []
    for nid in ordered_node_ids:
        f_full = skel_feats[nid]  # 7维: [node_type, bbox3, coord3]
        feats_list.append(f_full.astype(np.float32))

    if len(feats_list) == 0:
        feats_arr = np.zeros((1, 0, IN_DIM_WALL), dtype=np.float32)
        mask_arr  = np.zeros((1, 0), dtype=bool)
    else:
        feats_arr = np.stack(feats_list, axis=0)[None, ...]  # [1,L,7]
        mask_arr  = np.ones((1, len(feats_list)), dtype=bool)

    feats_tensor = torch.tensor(feats_arr, dtype=torch.float32)
    mask_tensor  = torch.tensor(mask_arr, dtype=torch.bool)
    return feats_tensor, mask_tensor, ordered_node_ids


############################################
# 构造 slot → PyG Data（喂 AnchorTransformerGCN）
############################################

def make_inference_pyg_data_for_slot(
    skel_nodes,
    skel_feats,
    skel_edges,
    parent_id,
    anchor_offset
):
    """
    构建一个PyG Data:
      - 包含所有骨架节点 (type1/2)
      - 加上一个slot节点 (表示某个候选附件)
      - slot节点特征 = [parent_full(7), anchor_offset(3)] (10维)
      - edge_index 里：骨架边(双向) + slot<->parent
      - slot_mask 标记哪个是slot节点，后续模型会只对slot节点回归几何
    """

    local_ids = {nid: i for i, nid in enumerate(skel_nodes)}
    Ns = len(skel_nodes)

    # 骨架节点特征 (Ns,10) = [features_full(7), (0,0,0)]
    skel_x_list = []
    for nid in skel_nodes:
        f_full = skel_feats[nid]  # shape (7,)
        pad = np.zeros((3,), dtype=np.float32)
        feat10 = np.concatenate([f_full, pad], axis=0)  # -> (10,)
        skel_x_list.append(feat10)

    if Ns > 0:
        skel_x_arr = np.stack(skel_x_list, axis=0)
    else:
        skel_x_arr = np.zeros((0, ANCHOR_IN_DIM), dtype=np.float32)

    # slot节点特征
    parent_full = skel_feats[parent_id]  # (7,)
    slot_feat10 = np.concatenate([parent_full, anchor_offset.astype(np.float32)], axis=0)  # (10,)

    # 拼成 x_all
    x_all = np.concatenate([skel_x_arr, slot_feat10[None,:]], axis=0)  # [Ns+1,10]

    # 边
    edges_local = []
    for (u,v) in skel_edges:
        u_loc = local_ids[u]
        v_loc = local_ids[v]
        edges_local.append([u_loc, v_loc])
        edges_local.append([v_loc, u_loc])

    parent_loc = local_ids[parent_id]
    slot_loc   = Ns

    edges_local.append([slot_loc, parent_loc])
    edges_local.append([parent_loc, slot_loc])

    if len(edges_local) > 0:
        edge_index_arr = np.array(edges_local, dtype=np.int64).T  # [2,E]
    else:
        edge_index_arr = np.zeros((2,0), dtype=np.int64)

    # slot_mask: (Ns+1,), slot位置为1
    slot_mask_arr = np.zeros((Ns+1,), dtype=np.float32)
    slot_mask_arr[slot_loc] = 1.0

    # batch全0：单图推理
    batch_arr = np.zeros((Ns+1,), dtype=np.int64)

    data = Data(
        x=torch.tensor(x_all, dtype=torch.float32),                 # [Ns+1,10]
        edge_index=torch.tensor(edge_index_arr, dtype=torch.long),  # [2,E]
        slot_mask=torch.tensor(slot_mask_arr, dtype=torch.float32), # [Ns+1]
        batch=torch.tensor(batch_arr, dtype=torch.long),            # [Ns+1]
    )
    return data


############################################
# 单栋楼推理
############################################

def run_full_inference_for_building(
    building_id,
    model_count,
    model_anchor,
    default_offsets=DEFAULT_OFFSETS
):
    """
    逻辑：
      1. 读整栋楼图
      2. 抽骨架图 (1/2类节点, skel_nodes/feats/edges)
      3. 用 model_count 预测每面墙的 child_count
      4. 对每面墙，根据 child_count 生成 slots
      5. 每个slot丢进 model_anchor 回归几何
      6. 组完整G_pred，导出csv
    """

    nodes_csv = os.path.join(ALL_GRAPH_DIR, f"{building_id}_nodes.csv")
    edges_csv = os.path.join(ALL_GRAPH_DIR, f"{building_id}_edges.csv")
    if not (os.path.exists(nodes_csv) and os.path.exists(edges_csv)):
        raise FileNotFoundError(f"缺少 {nodes_csv} 或 {edges_csv}")

    # step1: 完整图
    G_full = load_building_graph(nodes_csv, edges_csv)

    # step2: 骨架
    skel_nodes, skel_feats, skel_edges = build_skeleton_subgraph(G_full)

    # step3: 用 WallCountModel 预测每面墙该有多少个子构件
    feats_1L7, mask_1L, ordered_node_ids = build_wall_count_batch_from_skeleton(
        skel_nodes, skel_feats
    )
    feats_1L7 = feats_1L7.to(DEVICE)  # [1,L,7]
    mask_1L   = mask_1L.to(DEVICE)    # [1,L]

    with torch.no_grad():
        model_count.eval()
        count_pred_1L = model_count(feats_1L7, mask_1L)  # [1,L]
        count_pred_1L = count_pred_1L[0].cpu().numpy()   # [L]

    # 为每个墙节点生成一个int数量 Ki
    # Ki = max(0, round(pred_count))
    wall_to_K = {}
    for i, nid in enumerate(ordered_node_ids):
        raw_val = count_pred_1L[i]
        Ki = int(round(max(0.0, raw_val)))
        wall_to_K[nid] = Ki

    # step4-5: 对每个墙节点，按 Ki 生成slot并回归几何
    # 首先建立最终预测图 G_pred：先放所有骨架节点进去
    G_pred = nx.Graph()
    for nid in skel_nodes:
        f_full = skel_feats[nid]  # 7维: [type,bbox_x,y,z, cx,cy,cz]
        node_type = int(f_full[0])
        bbox_x, bbox_y, bbox_z = map(float, f_full[1:4])
        cx, cy, cz = map(float, f_full[4:7])
        G_pred.add_node(
            nid,
            node_type=node_type,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_z=bbox_z,
            coord_x=cx,
            coord_y=cy,
            coord_z=cz,
            generated=False,
        )

    # 骨架边（只加一次）
    for (u,v) in skel_edges:
        if u in G_pred.nodes and v in G_pred.nodes:
            G_pred.add_edge(u,v)

    model_anchor.eval()
    next_new_id = (max(skel_nodes) + 1) if len(skel_nodes) > 0 else 100000
    generated_edges_records = []

    with torch.no_grad():
        for parent_id in skel_nodes:
            # Ki = 这面墙要多少个附属构件
            Ki = wall_to_K.get(parent_id, 0)
            if Ki <= 0:
                continue

            for slot_i in range(Ki):
                anchor_offset = default_offsets[slot_i % len(default_offsets)]

                # 构建这个slot对应的PyG图
                pyg_data = make_inference_pyg_data_for_slot(
                    skel_nodes=skel_nodes,
                    skel_feats=skel_feats,
                    skel_edges=skel_edges,
                    parent_id=parent_id,
                    anchor_offset=anchor_offset
                ).to(DEVICE)

                # 用 AnchorTransformerGCN 回归几何
                pred_feat = model_anchor(pyg_data)  # [1,6]
                pred_feat = pred_feat[0].cpu().numpy()

                bbox_x, bbox_y, bbox_z = map(float, pred_feat[:3])
                cx, cy, cz             = map(float, pred_feat[3:6])

                gen_id = next_new_id
                next_new_id += 1

                G_pred.add_node(
                    gen_id,
                    node_type=CHILD_TYPE,
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_z=bbox_z,
                    coord_x=cx,
                    coord_y=cy,
                    coord_z=cz,
                    generated=True,
                    parent_wall=parent_id,
                    anchor_dx=float(anchor_offset[0]),
                    anchor_dy=float(anchor_offset[1]),
                    anchor_dz=float(anchor_offset[2]),
                    count_pred=float(wall_to_K[parent_id])
                )

                G_pred.add_edge(parent_id, gen_id)
                generated_edges_records.append({
                    "source": parent_id,
                    "target": gen_id,
                    "generated_edge": True
                })

    # step6: 导出成 CSV
    # nodes
    all_nodes_records = []
    for nid, attr in G_pred.nodes(data=True):
        rec = {
            "node_id": nid,
            "node_type": int(attr["node_type"]),
            "bbox_x": float(attr["bbox_x"]),
            "bbox_y": float(attr["bbox_y"]),
            "bbox_z": float(attr["bbox_z"]),
            "coord_x": float(attr["coord_x"]),
            "coord_y": float(attr["coord_y"]),
            "coord_z": float(attr["coord_z"]),
            "generated": bool(attr.get("generated", False)),
            "parent_wall": int(attr["parent_wall"]) if "parent_wall" in attr else -1,
            "anchor_dx": float(attr["anchor_dx"]) if "anchor_dx" in attr else 0.0,
            "anchor_dy": float(attr["anchor_dy"]) if "anchor_dy" in attr else 0.0,
            "anchor_dz": float(attr["anchor_dz"]) if "anchor_dz" in attr else 0.0,
            "wall_pred_count": float(attr["count_pred"]) if "count_pred" in attr else -1.0,
        }
        all_nodes_records.append(rec)

    pred_nodes_df = pd.DataFrame(all_nodes_records)

    # edges
    all_edges_records = []
    # 先把G_pred里的所有边拉出来
    for u, v in G_pred.edges():
        if u < v:
            all_edges_records.append({
                "source": u,
                "target": v,
                "generated_edge": False
            })

    # 把“父墙->新构件”的边标记成 True
    gen_edge_set = set()
    for rec in generated_edges_records:
        a, b = rec["source"], rec["target"]
        gen_edge_set.add(tuple(sorted((a,b))))

    for rec in all_edges_records:
        edge_key = tuple(sorted((rec["source"], rec["target"])))
        if edge_key in gen_edge_set:
            rec["generated_edge"] = True

    pred_edges_df = pd.DataFrame(all_edges_records)

    # 写盘
    nodes_out_csv = os.path.join(OUTPUT_DIR, f"{building_id}_pred_nodes.csv")
    edges_out_csv = os.path.join(OUTPUT_DIR, f"{building_id}_pred_edges.csv")

    pred_nodes_df.to_csv(nodes_out_csv, index=False)
    pred_edges_df.to_csv(edges_out_csv, index=False)

    print(f"[DONE] {building_id}: wrote")
    print(f"       {nodes_out_csv}")
    print(f"       {edges_out_csv}")

    return pred_nodes_df, pred_edges_df, G_pred


############################################
# main
############################################

def main():
    # 1. 载入墙计数模型
    model_count = WallCountModel(
        in_dim=IN_DIM_WALL,
        hidden_dim=HIDDEN_DIM_WALL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    state_count = torch.load(COUNT_CKPT_PATH, map_location=DEVICE)
    model_count.load_state_dict(state_count)
    model_count.eval()

    # 2. 载入构件几何回归模型
    model_anchor = AnchorTransformerGCN(
        in_dim=ANCHOR_IN_DIM,
        hidden_dim=ANCHOR_HIDDEN,
        out_dim=ANCHOR_OUT_DIM
    ).to(DEVICE)

    state_anchor = torch.load(ANCHOR_CKPT_PATH, map_location=DEVICE)
    model_anchor.load_state_dict(state_anchor)
    model_anchor.eval()

    # 3. 遍历 ALL_GRAPH_DIR 下所有 model_xxx
    node_files = sorted(glob.glob(os.path.join(ALL_GRAPH_DIR, "*_nodes.csv")))
    building_ids = []
    for nf in node_files:
        if nf.endswith("_nodes.csv"):
            bid = os.path.basename(nf)[:-len("_nodes.csv")]  # "model_001"
            ef = os.path.join(ALL_GRAPH_DIR, f"{bid}_edges.csv")
            if os.path.exists(ef):
                building_ids.append(bid)

    # 4. 对每栋楼跑完整推理
    for bid in building_ids:
        print(f"[INFER] {bid}")
        run_full_inference_for_building(
            building_id=bid,
            model_count=model_count,
            model_anchor=model_anchor,
            default_offsets=DEFAULT_OFFSETS
        )


if __name__ == "__main__":
    main()
