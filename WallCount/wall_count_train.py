# wall_count_train.py
#
# 任务：
#   预测每个墙体节点（node_type=1或2）应该有多少个附属构件（node_type=3，邻接在它上面）。
#
# 模型结构：
#   - TransformerEncoder 作为全局上下文编码器
#       输入 = 每个墙节点的基础几何/语义特征
#       输出 = 每个墙节点的上下文化embedding
#   - MLP decoder head
#       输入 = 每个墙节点的embedding
#       输出 = 该墙节点的child_count预测 (回归)
#
# 数据格式：
#   对每一栋建筑(model_xxx)，我们构建一个样本：
#       feats:  [N_walls,  F]   (F=7，目前为 [node_type, bbox_x,bbox_y,bbox_z, coord_x,coord_y,coord_z])
#       target: [N_walls]       (child_count，=与该墙直接相连的 type=3 节点数量)
#   DataLoader 会对一批建筑做padding:
#       feats -> [B, L, F]
#       target -> [B, L]
#       mask -> [B, L]  True表示有效位置，False表示padding
#
# 损失：
#   MSE，仅在mask=True的墙节点上计算
#
# 推理用法：
#   训练完后，对一栋楼跑forward，得到每个墙节点的child_count_pred。
#   这个 count 就可以喂给后续的“slot生成/几何回归”推理流程来控制生成数量。
#
# 依赖：
#   pip install torch pandas numpy
#   (TransformerEncoder使用的是nn.TransformerEncoderLayer)
#
# 注意：
#   Transformer对序列的顺序敏感，这里我们采用 node_id 排序的墙节点序列，保证一致性。
#   你可以后续加入learned positional embedding；当前实现中没有pos enc，
#   纯靠几何特征 + self-attention 的全局交互就足够做一个baseline。


import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


########################################
# 路径 & 训练超参
########################################

ALL_GRAPH_DIR = "./ALL_graph"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4        # batch里是4栋楼
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 20

HIDDEN_DIM = 256      # Transformer/MLP隐藏维度
NHEAD = 4             # Multi-head attention头数
NUM_LAYERS = 2        # TransformerEncoder层数

# 我们的墙节点基本特征维度：
# [node_type, bbox_x,bbox_y,bbox_z, coord_x,coord_y,coord_z] = 7
IN_DIM = 7


########################################
# 数据预处理函数
########################################

SKELETON_TYPES = {1, 2}  # 墙/结构类节点
CHILD_TYPE = 3           # 附属构件类节点（要计数的目标）

def load_full_graph(nodes_csv, edges_csv):
    """
    读取一栋楼的原始图:
      - nodes_csv: model_xxx_nodes.csv
      - edges_csv: model_xxx_edges.csv
    假设nodes_csv列类似：
        node_id, node_type,
        bounding_? (3列),
        coordinate_? (3列),
        node_degree(可存在或不存在)
    假设edges_csv列：
        edge_type, source, target

    返回:
        G_nodes: dict[node_id] = {
            "node_type": int,
            "feat7": np.array([node_type, bbox_x,bbox_y,bbox_z, coord_x,coord_y,coord_z], float32)
        }
        edges: list[(u,v)] (无向边)
    """

    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edges_csv)

    # 清理列名头尾空格
    df_nodes = df_nodes.rename(columns=lambda c: c.strip())
    df_edges = df_edges.rename(columns=lambda c: c.strip())

    # 我们假定最少具备以下列：
    # node_id, node_type, 然后至少6列是 bbox(3) + coord(3)
    all_cols = list(df_nodes.columns)
    if "node_id" not in all_cols or "node_type" not in all_cols:
        raise RuntimeError(f"{nodes_csv}: 需要包含 node_id 和 node_type")

    node_id_col = "node_id"
    node_type_col = "node_type"
    rest_cols = [c for c in all_cols if c not in [node_id_col, node_type_col]]
    if len(rest_cols) < 6:
        raise RuntimeError(
            f"{nodes_csv}: 期望除了 node_id/node_type 至少还有6列(bbox_x,bbox_y,bbox_z,coord_x,coord_y,coord_z)，"
            f"但只发现 {len(rest_cols)} 列: {rest_cols}"
        )
    bbox_coord_cols = rest_cols[:6]

    # 组装节点
    G_nodes = {}
    for row in df_nodes.itertuples(index=False):
        nid = int(getattr(row, node_id_col))
        ntype = int(getattr(row, node_type_col))

        bbox_x = float(getattr(row, bbox_coord_cols[0]))
        bbox_y = float(getattr(row, bbox_coord_cols[1]))
        bbox_z = float(getattr(row, bbox_coord_cols[2]))

        cx = float(getattr(row, bbox_coord_cols[3]))
        cy = float(getattr(row, bbox_coord_cols[4]))
        cz = float(getattr(row, bbox_coord_cols[5]))

        feat7 = np.array([
            ntype,
            bbox_x, bbox_y, bbox_z,
            cx, cy, cz
        ], dtype=np.float32)

        G_nodes[nid] = {
            "node_type": ntype,
            "feat7": feat7
        }

    # 边
    if "source" not in df_edges.columns or "target" not in df_edges.columns:
        raise RuntimeError(f"{edges_csv}: 缺少 source / target 列")

    edges = []
    for row in df_edges.itertuples(index=False):
        u = int(getattr(row, "source"))
        v = int(getattr(row, "target"))
        if (u in G_nodes) and (v in G_nodes):
            edges.append((u,v))
            edges.append((v,u))  # 我们把它当成无向，用双向存
    return G_nodes, edges


def extract_wall_features_and_targets(G_nodes, edges):
    """
    给定整栋楼的图（节点和边），我们要为"墙/骨架节点"构建监督：
        对每个 node_type in {1,2} 的节点:
            feature = feat7
            target  = child_count = 有多少个直接相连的 node_type==3 的邻居

    返回:
        feats_tensor: [N_walls, 7] float32
        target_tensor:[N_walls]   float32
        wall_ids:      [N_walls]  int (记录对应原始node_id，方便debug/可视化)

    注意：我们会把墙节点按 node_id 升序排，这个顺序就是 Transformer 的序列顺序。
    """

    # 先建邻接表
    nbrs = {nid: [] for nid in G_nodes.keys()}
    for (u,v) in edges:
        if u in nbrs:
            nbrs[u].append(v)

    # 选出墙节点
    wall_ids = [
        nid for nid, attr in G_nodes.items()
        if attr["node_type"] in SKELETON_TYPES
    ]
    wall_ids.sort()

    feats = []
    targets = []
    for wid in wall_ids:
        feat7 = G_nodes[wid]["feat7"]  # [7]

        # 数邻居中 node_type==3 的数量
        c = 0
        for nb in nbrs[wid]:
            if nb in G_nodes and G_nodes[nb]["node_type"] == CHILD_TYPE:
                c += 1

        feats.append(feat7)
        targets.append(float(c))

    if len(feats) == 0:
        # 这栋楼可能没有墙节点？那就返回空
        feats_tensor = torch.zeros((0,7), dtype=torch.float32)
        targets_tensor = torch.zeros((0,), dtype=torch.float32)
    else:
        feats_tensor = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)  # [N,7]
        targets_tensor = torch.tensor(np.array(targets, dtype=np.float32), dtype=torch.float32)  # [N]

    return feats_tensor, targets_tensor, wall_ids


########################################
# Dataset + collate
########################################

class WallCountDataset(Dataset):
    """
    每个样本 = 一栋楼 (model_xxx)
    返回：
        {
          "feats":   [N_walls,7] float32
          "targets": [N_walls]   float32
          "mask":    [N_walls]   float32 (这里先不pad，后面collate再pad)
          "wall_ids":[N_walls]   int (debug用，不进模型)
        }
    """

    def __init__(self, graph_dir=ALL_GRAPH_DIR):
        super().__init__()
        # 找到所有 *_nodes.csv
        node_files = sorted(glob.glob(os.path.join(graph_dir, "*_nodes.csv")))
        self.buildings = []
        for nf in node_files:
            if not nf.endswith("_nodes.csv"):
                continue
            bid = os.path.basename(nf)[:-len("_nodes.csv")]  # model_001
            ef = os.path.join(graph_dir, f"{bid}_edges.csv")
            if os.path.exists(ef):
                self.buildings.append((bid, nf, ef))

        if len(self.buildings) == 0:
            raise RuntimeError(f"在 {graph_dir} 下没有找到成对的 *_nodes.csv / *_edges.csv")

    def __len__(self):
        return len(self.buildings)

    def __getitem__(self, idx):
        bid, nodes_csv, edges_csv = self.buildings[idx]
        G_nodes, G_edges = load_full_graph(nodes_csv, edges_csv)
        feats, targets, wall_ids = extract_wall_features_and_targets(G_nodes, G_edges)

        # mask: 当前楼里哪些位置有效（无pad），全1
        mask = torch.ones((feats.shape[0],), dtype=torch.bool)

        return {
            "building_id": bid,
            "feats": feats,         # [N,7]
            "targets": targets,     # [N]
            "mask": mask,           # [N] bool
            "wall_ids": wall_ids    # list[int]，debug
        }


def wall_collate_fn(batch_list):
    """
    把一批不同大小的楼 (N_i各不相同) pad 到统一长度 L=max_i N_i

    返回:
        feats_pad   [B, L, 7] float32
        targets_pad [B, L]    float32
        mask_pad    [B, L]    bool
        meta info: building_ids (list), wall_ids (list[list[int]])
    """
    B = len(batch_list)
    lens = [item["feats"].shape[0] for item in batch_list]
    L = max(lens) if len(lens)>0 else 0

    feats_pad   = torch.zeros((B, L, IN_DIM), dtype=torch.float32)
    targets_pad = torch.zeros((B, L), dtype=torch.float32)
    mask_pad    = torch.zeros((B, L), dtype=torch.bool)

    building_ids = []
    wall_ids_all = []

    for i, item in enumerate(batch_list):
        n = item["feats"].shape[0]
        building_ids.append(item["building_id"])
        wall_ids_all.append(item["wall_ids"])

        feats_pad[i, :n, :] = item["feats"]          # [n,7]
        targets_pad[i, :n]  = item["targets"]        # [n]
        mask_pad[i, :n]     = item["mask"]           # [n] True

    return {
        "feats": feats_pad,         # [B,L,7]
        "targets": targets_pad,     # [B,L]
        "mask": mask_pad,           # [B,L] bool
        "building_ids": building_ids,
        "wall_ids": wall_ids_all
    }


########################################
# 模型
########################################

class WallCountModel(nn.Module):
    """
    TransformerEncoder + MLP decoder head

    输入: feats_pad [B,L,7]
          mask_pad  [B,L]  True表示有效
    输出: pred_count [B,L] (float，回归 child_count)

    结构:
      1. 线性投影: 7 -> HIDDEN_DIM
      2. TransformerEncoder (batch_first=True)
         - src_key_padding_mask 传 (~mask_pad)，也就是 padding位置不参与注意力
      3. MLP head: HIDDEN_DIM -> 1
    """

    def __init__(self,
                 in_dim=IN_DIM,
                 hidden_dim=HIDDEN_DIM,
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
            nn.Linear(hidden_dim, 1)  # 回归 child_count
        )

    def forward(self, feats_pad, mask_pad):
        """
        feats_pad: [B,L,7] float
        mask_pad:  [B,L]   bool (True=有效,False=padding)
        """
        # 1. 投影
        h = self.in_proj(feats_pad)  # [B,L,H]

        # 2. TransformerEncoder
        # PyTorch的src_key_padding_mask=True表示位置是padding，应被mask掉
        src_key_padding_mask = (~mask_pad)  # [B,L] True=PAD
        h_enc = self.encoder(
            h,
            src_key_padding_mask=src_key_padding_mask
        )  # [B,L,H]

        # 3. MLP head逐token输出
        pred = self.decoder_head(h_enc).squeeze(-1)  # [B,L]

        return pred


########################################
# 训练循环
########################################

def train_loop():
    dataset = WallCountDataset(ALL_GRAPH_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=wall_collate_fn
    )

    model = WallCountModel(
        in_dim=IN_DIM,
        hidden_dim=HIDDEN_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    mse_loss = nn.MSELoss(reduction="none")  # 我们手动用mask做加权

    global_step = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0

        for batch in loader:
            feats = batch["feats"].to(DEVICE)        # [B,L,7]
            targets = batch["targets"].to(DEVICE)    # [B,L]
            mask = batch["mask"].to(DEVICE)          # [B,L] bool

            pred = model(feats, mask)                # [B,L]

            # 计算有mask位置的MSE
            per_elem_loss = mse_loss(pred, targets)  # [B,L]
            masked_loss = per_elem_loss[mask]        # 只拿有效墙
            if masked_loss.numel() == 0:
                # 这批里可能某栋楼没有墙节点
                loss = torch.zeros([], device=DEVICE)
            else:
                loss = masked_loss.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_count += 1
            global_step += 1

            if global_step % PRINT_EVERY == 0:
                # 额外打印一下平均child_count大小，帮你直觉看预测尺度
                with torch.no_grad():
                    gt_avg = targets[mask].mean().item() if mask.any() else 0.0
                    pd_avg = pred[mask].mean().item() if mask.any() else 0.0
                print(
                    f"[epoch {epoch:03d} step {global_step:06d}] "
                    f"loss={loss.item():.4f} "
                    f"gt_avg_count={gt_avg:.2f} "
                    f"pred_avg_count={pd_avg:.2f} "
                    f"batch_size={feats.size(0)}"
                )

        scheduler.step()
        epoch_avg_loss = epoch_loss_sum / max(1, epoch_count)
        print(f"[epoch {epoch:03d}] avg_loss={epoch_avg_loss:.4f}")

        # 每个epoch存个checkpoint
        os.makedirs("./wall_count_checkpoints", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"./wall_count_checkpoints/wallcount_epoch_{epoch:03d}.pt"
        )


if __name__ == "__main__":
    train_loop()
