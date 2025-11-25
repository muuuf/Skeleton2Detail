# anchor_train.py

import os
import glob
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np # [新增]

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter # [新增]

from anchor_model import AnchorTransformerGCN, OUT_DIM, IN_DIM

############################################
# 路径 & 超参
############################################

ANCHOR_DATA_DIR = "./anchor_data"  # 和预处理保持一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LOG_INTERVAL = 50  # 打印频率
LOG_DIR = "./runs/anchor_experiment" # [新增] TensorBoard 日志目录

############################################
# Dataset
############################################

class AnchorGraphData(Data):
    """
    自定义 Data，确保 edge_index 在batch拼接时正确偏移。
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            # 告诉DataLoader，在batch的时候要对edge_index加上当前图节点数
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class AnchorDataset(torch.utils.data.Dataset):
    """
    每个样本是 anchor_preprocess.py 生成的一个 .pt
      - x: [N,10]
      - edge_index: [2,E]
      - slot_mask: [N]
      - target_feat: [1,6]
    """
    def __init__(self, data_dir=ANCHOR_DATA_DIR):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt samples found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx], map_location="cpu")

        x = item["x"].float()                    # [N,10]
        edge_index = item["edge_index"].long()   # [2,E]
        slot_mask = item["slot_mask"].float()    # [N]
        target_feat = item["target_feat"].float()# [1,6]

        data = AnchorGraphData(
            x=x,
            edge_index=edge_index,
            slot_mask=slot_mask,
            target_feat=target_feat
        )
        return data


############################################
# 训练逻辑
############################################

def train_loop():
    # [新增] 初始化 TensorBoard writer
    print(f"Initializing TensorBoard writer at {LOG_DIR}")
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    dataset = AnchorDataset(ANCHOR_DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AnchorTransformerGCN(
        in_dim=IN_DIM,
        hidden_dim=256,
        out_dim=OUT_DIM
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss() # [修改] 总损失，用于反向传播

    global_step = 0

    print("Starting training...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        
        # [修改] 增加各项损失的累加器
        running_loss = 0.0
        running_per_dim_loss = torch.zeros(OUT_DIM).to(DEVICE) # [新增]
        num_batches = 0

        for batch in loader:
            batch = batch.to(DEVICE)

            # 模型前向
            pred_feat = model(batch)  # [B,6]

            # GT 整理成 [B,6]
            target_feat = batch.target_feat.to(DEVICE)  # [B,6]

            # [修改] 计算总损失 (用于反向传播)
            loss = criterion(pred_feat, target_feat)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item()

            # [新增] 计算并累加 "各项损失" (每个维度的MSE)
            with torch.no_grad():
                # (pred - target)^2，然后按 batch 维度求均值，得到 [6]
                per_dim_loss_batch = (pred_feat - target_feat).pow(2).mean(dim=0)
                running_per_dim_loss += per_dim_loss_batch

            num_batches += 1

            if global_step % LOG_INTERVAL == 0:
                print(f"[epoch {epoch:03d} step {global_step:06d}] "
                      f"loss={loss.item():.4f}  batch_size={pred_feat.size(0)}")
                
                # [新增] 记录 batch loss 到 TensorBoard
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)

            global_step += 1

        scheduler.step()

        # [修改] 计算 epoch 平均损失
        avg_loss = running_loss / max(1, num_batches)
        avg_per_dim_loss = running_per_dim_loss / max(1, num_batches) # [新增]
        
        print(f"[epoch {epoch:03d}] avg_loss={avg_loss:.4f}")
        
        # [新增] 记录 epoch 级别的各项损失到 TensorBoard
        # 1. 记录总的平均损失
        writer.add_scalar('Loss/train_epoch_avg', avg_loss, epoch)
        
        # 2. 记录 "各项" 损失 (每个维度)
        for i in range(OUT_DIM):
            writer.add_scalar(f'Loss_Dim/dim_{i}', avg_per_dim_loss[i].item(), epoch)
            
        # 3. 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate', current_lr, epoch)


        # 存模型
        os.makedirs("./anchor_checkpoints", exist_ok=True)
        torch.save(model.state_dict(),
                   f"./anchor_checkpoints/model_epoch_{epoch:03d}.pt")

    # [新增] 训练结束后关闭 writer
    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    train_loop()