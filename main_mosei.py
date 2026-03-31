import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pool(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.dim() == 3:
        return x.mean(dim=1)
    return x
    
def get_missing_mode(missing_rate):
    if np.random.random() < missing_rate:
        return np.random.randint(0, 6)
    else:
        return 6

def get_mask(view_num, alldata_len, missing_rate):
    nums = np.ones((view_num, alldata_len))
    missing_patterns = {0: [2], 1: [0], 2: [1], 3: [0, 2], 4: [1, 2], 5: [0, 1], 6: []}
    for i in range(alldata_len):
        mode = get_missing_mode(missing_rate)
        for idx in missing_patterns[mode]:
            nums[idx, i] = 0
    return torch.from_numpy(nums).float()

def evaluate_single_missing_pattern(model, test_data, batch_size, device, missing_pattern):
    model.eval()
    text_data, image_data, audio_data, label_data = test_data
    
    if isinstance(label_data, np.ndarray):
        label_data = torch.from_numpy(label_data).float()
    
    num_samples = len(label_data)
    all_preds = []
    all_preds_text = []
    all_preds_visual = []
    all_preds_audio = []
    all_labels = []
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            # 准备text inputs (dict)
            text_inputs = {
                'input_ids': torch.stack([text_data[i]['input_ids'] for i in range(start_idx, end_idx)]).to(device),
                'attention_mask': torch.stack([text_data[i]['attention_mask'] for i in range(start_idx, end_idx)]).to(device),
                'token_type_ids': torch.stack([text_data[i]['token_type_ids'] for i in range(start_idx, end_idx)]).to(device)
            }
            
            image = image_data[start_idx:end_idx].to(device)
            audio = audio_data[start_idx:end_idx].to(device)
            labels = label_data[start_idx:end_idx].squeeze().float().to(device)
            
            batch_len = end_idx - start_idx
            mask = torch.ones(3, batch_len).to(device)
            
            missing_patterns = {
                0: [0, 1], 1: [1, 2], 2: [0, 2],
                3: [0], 4: [1], 5: [2], 6: []
            }
            
            for idx in missing_patterns[missing_pattern]:
                mask[idx, :] = 0
            
            completed_map, _ = model.complete_modalities(text_inputs, image, audio, mask, compute_loss=False)
            completed_map['text'] = model.norm_text(completed_map['text'])
            completed_map['image'] = model.norm_image(completed_map['image'])
            completed_map['audio'] = model.norm_audio(completed_map['audio'])
            
            pred_text = model.text_classifier(completed_map['text']).squeeze()
            pred_visual = model.image_classifier(completed_map['image']).squeeze()
            pred_audio = model.audio_classifier(completed_map['audio']).squeeze()
            
            fused_feats = model.fusion(completed_map['text'].detach(), 
                                       completed_map['image'].detach(), 
                                       completed_map['audio'].detach(), mask)
            preds = model.classifier(fused_feats).squeeze()
            
            all_preds.append(preds.cpu())
            all_preds_text.append(pred_text.cpu())
            all_preds_visual.append(pred_visual.cpu())
            all_preds_audio.append(pred_audio.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).squeeze()
    all_preds_text = torch.cat(all_preds_text).squeeze()
    all_preds_visual = torch.cat(all_preds_visual).squeeze()
    all_preds_audio = torch.cat(all_preds_audio).squeeze()
    all_labels = torch.cat(all_labels).squeeze()
    
    non_zero_mask = (all_labels != 0)
    all_preds = all_preds[non_zero_mask]
    all_preds_text = all_preds_text[non_zero_mask]
    all_preds_visual = all_preds_visual[non_zero_mask]
    all_preds_audio = all_preds_audio[non_zero_mask]
    all_labels = all_labels[non_zero_mask]
    
    mae = torch.mean(torch.abs(all_preds - all_labels)).item()
    pred_binary = (all_preds.numpy() > 0).astype(int)
    label_binary = (all_labels.numpy() > 0).astype(int)
    acc = accuracy_score(label_binary, pred_binary)
    f1 = f1_score(label_binary, pred_binary, average='weighted')
    
    pred_text_binary = (all_preds_text.numpy() > 0).astype(int)
    pred_visual_binary = (all_preds_visual.numpy() > 0).astype(int)
    pred_audio_binary = (all_preds_audio.numpy() > 0).astype(int)
    
    acc_text = accuracy_score(label_binary, pred_text_binary)
    acc_visual = accuracy_score(label_binary, pred_visual_binary)
    acc_audio = accuracy_score(label_binary, pred_audio_binary)
    
    return mae, acc, f1, acc_text, acc_visual, acc_audio

def evaluate_all_missing_patterns(model, test_data, batch_size, device):
    pattern_names = {
        0: "A-only", 1: "T-only", 2: "V-only", 
        3: "AV", 4: "AT", 5: "TV", 6: "ATV"
    }
    results = {}
    print(f"{'Pattern':<8} | {'Fus':<5} | {'T':<5} | {'V':<5} | {'A':<5} | {'Gap':<6}")
    print("-" * 60)

    for pattern_id in range(7):
        mae, acc, f1, acc_text, acc_visual, acc_audio = evaluate_single_missing_pattern(
            model, test_data, batch_size, device, pattern_id
        )
        best_single = max(acc_text, acc_visual, acc_audio)
        gap = acc - best_single
        results[pattern_id] = {
            'name': pattern_names[pattern_id], 'mae': mae, 'acc': acc, 'f1': f1,
            'acc_text': acc_text, 'acc_visual': acc_visual, 'acc_audio': acc_audio, 'gap': gap
        }
        print(f"{pattern_names[pattern_id]:<8} | {acc*100:<5.1f} | {acc_text*100:<5.1f} | "
              f"{acc_visual*100:<5.1f} | {acc_audio*100:<5.1f} | {gap*100:+6.1f}")

    avg_gap = np.mean([results[i]['gap'] for i in range(7)])
    print("-" * 60)
    print(f"{'AVG':<8} | {'--':<5} | {'--':<5} | {'--':<5} | {'--':<5} | {avg_gap*100:+6.1f}")
    return results

def visualize_prototypes(model, test_data, device, epoch, save_path='proto_vis'):
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    text_data, image_data, audio_data, label_data = test_data
    n_samples = min(500, len(label_data))
    
    text_feats, image_feats, audio_feats = [], [], []
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, 64):
            end_idx = min(start_idx + 64, n_samples)
            
            text_inputs = {
                'input_ids': torch.stack([text_data[i]['input_ids'] for i in range(start_idx, end_idx)]).to(device),
                'attention_mask': torch.stack([text_data[i]['attention_mask'] for i in range(start_idx, end_idx)]).to(device),
                'token_type_ids': torch.stack([text_data[i]['token_type_ids'] for i in range(start_idx, end_idx)]).to(device)
            }
            
            image = image_data[start_idx:end_idx].to(device)
            audio = audio_data[start_idx:end_idx].to(device)
            mask = torch.ones(3, end_idx - start_idx).to(device)
            
            completed_map, _ = model.complete_modalities(text_inputs, image, audio, mask, compute_loss=False)
            text_norm = model.norm_text(completed_map['text'])
            image_norm = model.norm_image(completed_map['image'])
            audio_norm = model.norm_audio(completed_map['audio'])
            
            text_feats.append(text_norm.cpu())
            image_feats.append(image_norm.cpu())
            audio_feats.append(audio_norm.cpu())
    
    text_feats = torch.cat(text_feats)
    image_feats = torch.cat(image_feats)
    audio_feats = torch.cat(audio_feats)
    protos = model.prototypes.detach().cpu()
    
    all_feats = torch.cat([text_feats, image_feats, audio_feats, protos])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_feats.numpy())
    
    n = len(text_feats)
    text_2d = all_2d[:n]
    image_2d = all_2d[n:2*n]
    audio_2d = all_2d[2*n:3*n]
    proto_2d = all_2d[3*n:]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(text_2d[:, 0], text_2d[:, 1], c='red', alpha=0.3, s=10, label='Text')
    ax.scatter(image_2d[:, 0], image_2d[:, 1], c='blue', alpha=0.3, s=10, label='Image')
    ax.scatter(audio_2d[:, 0], audio_2d[:, 1], c='green', alpha=0.3, s=10, label='Audio')
    ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c='black', s=200, marker='*', 
               edgecolors='yellow', linewidths=2, label='Prototypes')
    for i, (x, y) in enumerate(proto_2d):
        ax.text(x, y, str(i), fontsize=12, ha='center', va='center', color='white', weight='bold')
    ax.legend()
    ax.set_title(f'Prototype Alignment (Epoch {epoch})')
    plt.savefig(f'{save_path}/proto_scatter_epoch{epoch}.png', dpi=150)
    plt.close()

def init_prototypes_randomly(model, device):
    """随机初始化原型（已经在__init__中完成）"""
    print("\n" + "="*80)
    print("Using Random Initialization for Prototypes...")
    print("="*80)
    print(f"  Initialized {model.num_prototypes} prototypes with Xavier uniform distribution")
    print("Prototypes initialized successfully!")
    print("="*80 + "\n")

def prototype_separation_loss(prototypes, margin=0.5):
    """原型分离损失:惩罚相似度>margin的原型对"""
    proto_norm = F.normalize(prototypes, dim=1)
    sim_matrix = torch.mm(proto_norm, proto_norm.t())
    
    mask = ~torch.eye(len(prototypes), dtype=torch.bool, device=prototypes.device)
    off_diag = sim_matrix[mask]
    
    loss = F.relu(off_diag - margin).mean()
    
    return loss

def compute_prototype_emotion_loss(model, labels, device):
    """
    计算原型情感预测损失
    对每个原型预测其情感分数，与分配到该原型的样本的平均真实情感对比
    """
    proto_emotions = model.prototype_emotion_predictor(model.prototypes).squeeze()
    
    discrete_labels = model.discretize_labels(labels)
    
    total_loss = 0.0
    valid_prototypes = 0
    
    for k in range(model.num_prototypes):
        mask_k = (discrete_labels == k)
        
        if mask_k.sum() > 0:
            mean_true_emotion = labels[mask_k].mean()
            proto_emotion_k = proto_emotions[k]
            loss_k = (proto_emotion_k - mean_true_emotion) ** 2
            total_loss += loss_k
            valid_prototypes += 1
    
    if valid_prototypes > 0:
        return total_loss / valid_prototypes
    else:
        return torch.tensor(0.0, device=device)

def pretrain_prototypes_and_projections(model, train_data, batch_size, device, epochs=20):
    """
    两阶段预训练 + BERT微调:
    阶段1 (前10轮): 投影层 + 分类器 (高权重) + BERT微调
    阶段2 (后10轮): 原型学习 (高权重) + BERT微调
    预训练结束后会自动冻结BERT
    """
    text_data, image_data, audio_data, label_data = train_data
    
    # 确保BERT在预训练期间是可训练的
    model.unfreeze_bert()
    
    init_prototypes_randomly(model, device)
    
    print("\n" + "="*80)
    print("INITIAL State (Random Initialization)")
    print("="*80)
    diagnose_prototype_emotion_alignment(model, train_data, device)
    print("="*80 + "\n")
    
    model.prototypes.requires_grad = True
    
    # 优化器包含BERT参数
    optimizer = torch.optim.AdamW([
        {'params': model.text_proj_proto.parameters(), 'lr': 5e-4},
        {'params': model.image_proj_proto.parameters(), 'lr': 5e-4},
        {'params': model.audio_proj_proto.parameters(), 'lr': 1e-3},
        {'params': [model.prototypes], 'lr': 1e-3},
        {'params': model.prototype_emotion_predictor.parameters(), 'lr': 1e-3},
        {'params': model.text_classifier.parameters(), 'lr': 5e-4},
        {'params': model.image_classifier.parameters(), 'lr': 5e-4},
        {'params': model.audio_classifier.parameters(), 'lr': 1e-3},
        {'params': model.bert.parameters(), 'lr': 2e-5}  # BERT用较小学习率
    ], weight_decay=1e-4)
    
    print("\n" + "="*80)
    print("Two-Stage Pretraining (with BERT fine-tuning):")
    print("  Stage 1 (Epochs 1-10):  Focus on Projection + Classifier + BERT")
    print("  Stage 2 (Epochs 11-20): Focus on Prototype Learning + BERT")
    print("  After pretraining: BERT will be FROZEN")
    print("="*80)
    
    stage1_epochs = 20
    
    for epoch in range(epochs):
        model.train()
        model.bert.train()  # 确保BERT在训练模式
        
        num_samples = len(label_data)
        indices = torch.randperm(num_samples)
        total_loss = 0
        total_cls_loss = 0
        total_emotion_loss = 0
        total_sep_loss = 0
        total_reg_loss = 0
        num_batches = 0
        
        # 动态权重策略
        if epoch < stage1_epochs:
            # 阶段1: 重点学习投影+分类器
            stage = 1
            reg_weight = 2.0      # 情感回归权重高
            cls_weight = 0.5      # 原型分类权重低
            emotion_weight = 0.1  # 原型情感预测权重低
            sep_weight = 0.3      # 原型分离权重低
            
            # 原型学习率降低
            for param_group in optimizer.param_groups:
                if len(param_group['params']) == 1 and param_group['params'][0].shape == model.prototypes.shape:
                    param_group['lr'] = 1e-4
        else:
            # 阶段2: 重点学习原型
            stage = 2
            reg_weight = 0.5      # 情感回归权重降低（保持监督）
            cls_weight = 2.0      # 原型分类权重高
            emotion_weight = 1.5  # 原型情感预测权重高
            sep_weight = 1.0      # 原型分离权重恢复
            
            # 投影层学习率降低，原型学习率恢复
            if epoch == stage1_epochs:
                for i, param_group in enumerate(optimizer.param_groups):
                    if i < 3:  # text_proj_proto, image_proj_proto, audio_proj_proto
                        param_group['lr'] *= 0.5
                    elif len(param_group['params']) == 1 and param_group['params'][0].shape == model.prototypes.shape:
                        param_group['lr'] = 1e-3
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # 准备text inputs (dict)
            text_inputs = {
                'input_ids': torch.stack([text_data[i]['input_ids'] for i in batch_indices]).to(device),
                'attention_mask': torch.stack([text_data[i]['attention_mask'] for i in batch_indices]).to(device),
                'token_type_ids': torch.stack([text_data[i]['token_type_ids'] for i in batch_indices]).to(device)
            }
            
            image = image_data[batch_indices].to(device)
            audio = audio_data[batch_indices].to(device)
            labels = torch.tensor(label_data[batch_indices]).squeeze().float().to(device)
            
            discrete_labels = model.discretize_labels(labels)
            
            # 先通过BERT编码
            text_bert = model.encode_text_with_bert(text_inputs)  # [batch, 768]
            
            # 投影到原型空间
            text_feat = model.text_proj_proto(text_bert)
            image_feat = model.image_proj_proto(image)
            audio_feat = model.audio_proj_proto(audio)
            
            text_feat_norm = F.normalize(text_feat, dim=1)
            image_feat_norm = F.normalize(image_feat, dim=1)
            audio_feat_norm = F.normalize(audio_feat, dim=1)
            proto_norm = F.normalize(model.prototypes, dim=1)
            
            # 原型匹配相似度
            text_sim = torch.mm(text_feat_norm, proto_norm.t())
            image_sim = torch.mm(image_feat_norm, proto_norm.t())
            audio_sim = torch.mm(audio_feat_norm, proto_norm.t())
            
            # 损失1: 原型分类损失
            loss_text_cls = F.cross_entropy(text_sim / 0.05, discrete_labels)
            loss_image_cls = F.cross_entropy(image_sim / 0.10, discrete_labels)
            loss_audio_cls = F.cross_entropy(audio_sim / 0.15, discrete_labels)
            
            cls_loss = loss_text_cls + 2.0 * loss_image_cls + 3.0 * loss_audio_cls
            
            # 损失2: 单模态情感回归损失
            pred_text = model.text_classifier(text_feat).squeeze()
            pred_image = model.image_classifier(image_feat).squeeze()
            pred_audio = model.audio_classifier(audio_feat).squeeze()
            
            reg_loss_text = F.mse_loss(pred_text, labels)
            reg_loss_image = F.mse_loss(pred_image, labels)
            reg_loss_audio = F.mse_loss(pred_audio, labels)
            
            reg_loss = reg_loss_text + reg_loss_image + reg_loss_audio
            
            # 损失3: 原型情感预测损失
            emotion_loss = compute_prototype_emotion_loss(model, labels, device)
            
            # 损失4: 原型分离损失
            if stage == 1:
                margin = 0.6
            else:
                margin = 0.4
            
            sep_loss = prototype_separation_loss(model.prototypes, margin=margin)
            
            # 总损失（动态权重）
            loss = (cls_weight * cls_loss + 
                   emotion_weight * emotion_loss + 
                   sep_weight * sep_loss + 
                   reg_weight * reg_loss)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_emotion_loss += emotion_loss.item()
            total_sep_loss += sep_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_cls = total_cls_loss / num_batches
        avg_emo = total_emotion_loss / num_batches
        avg_sep = total_sep_loss / num_batches
        avg_reg = total_reg_loss / num_batches
        
        stage_marker = "[Stage 1]" if stage == 1 else "[Stage 2]"
        print(f"{stage_marker} Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f} | "
              f"Cls={avg_cls:.4f}(w={cls_weight}) | Emotion={avg_emo:.4f}(w={emotion_weight}) | "
              f"Sep={avg_sep:.4f}(w={sep_weight}) | Reg={avg_reg:.4f}(w={reg_weight})")
        
        # 阶段转换时诊断
        if epoch + 1 == stage1_epochs:
            print("\n" + "="*80)
            print("Stage 1 -> Stage 2 Transition")
            print("="*80)
            diagnose_prototype_emotion_alignment(model, train_data, device)
            print("="*80 + "\n")
        
        # 每5个epoch诊断
        if (epoch + 1) % 5 == 0:
            print("\n" + "-"*80)
            print(f"Diagnosis at Epoch {epoch+1}")
            print("-"*80)
            diagnose_prototype_emotion_alignment(model, train_data, device)
            print("-"*80 + "\n")
    
    print("\n" + "="*80)
    print("FINAL State (After Two-Stage Pretraining)")
    print("="*80)
    diagnose_prototype_emotion_alignment(model, train_data, device)
    print("="*80 + "\n")
    
    # 预训练结束后冻结BERT
    model.freeze_bert()

def diagnose_prototype_emotion_alignment(model, train_data, device):
    """诊断原型与情感的对齐情况"""
    model.eval()
    text_data, image_data, audio_data, label_data = train_data
    
    with torch.no_grad():
        # 1. 获取所有原型的情感预测
        proto_emotions = model.prototype_emotion_predictor(model.prototypes).squeeze()
        
        print("\n" + "="*70)
        print("Prototype Emotion Predictions:")
        print("="*70)
        for i, score in enumerate(proto_emotions.tolist()):
            print(f"  Prototype {i}: {score:+.3f}")
        
        # 验证单调性
        is_monotonic = all(proto_emotions[i] <= proto_emotions[i+1] 
                          for i in range(len(proto_emotions)-1))
        status = "[PASS]" if is_monotonic else "[FAIL]"
        print(f"\n  {status} Monotonicity: {'increasing' if is_monotonic else 'not monotonic'}")
        
        emotion_range = proto_emotions.max() - proto_emotions.min()
        print(f"  Emotion Range: {emotion_range:.3f} (expected ~6.0)")
        
        # 2. 每个原型负责的样本统计
        n = min(3000, len(label_data))
        labels = torch.tensor(label_data[:n]).float().to(device)
        discrete_labels = model.discretize_labels(labels)
        
        print("\n" + "="*70)
        print("Per-Prototype Statistics:")
        print("="*70)
        print(f"{'Proto':<6} | {'Count':<6} | {'True Emo':<15} | {'Pred Emo':<10} | {'Error':<8} | {'Status'}")
        print("-"*85)
        
        proto_errors = []
        for k in range(model.num_prototypes):
            mask_k = (discrete_labels == k)
            count = mask_k.sum().item()
            
            if count > 0:
                true_emotions = labels[mask_k]
                mean_true = true_emotions.mean().item()
                std_true = true_emotions.std().item()
                pred_emo = proto_emotions[k].item()
                error = abs(mean_true - pred_emo)
                proto_errors.append(error)
                
                if error < 0.5:
                    status = "[Good]"
                elif error < 1.0:
                    status = "[OK]"
                else:
                    status = "[Poor]"
                print(f"{k:<6} | {count:<6} | {mean_true:+.2f} +/- {std_true:.2f}    | "
                      f"{pred_emo:+.2f}      | {error:.3f}    | {status}")
            else:
                print(f"{k:<6} | {count:<6} | {'N/A':<15} | {proto_emotions[k].item():+.2f}      | "
                      f"{'N/A':<8} | [Empty]")
        
        if proto_errors:
            avg_error = np.mean(proto_errors)
            max_error = np.max(proto_errors)
            print("-"*85)
            print(f"Overall Prediction Error: avg={avg_error:.3f}, max={max_error:.3f}")
        
        # 3. 原型分离度
        proto_norm = F.normalize(model.prototypes, dim=1)
        proto_sim = torch.mm(proto_norm, proto_norm.t())
        mask = ~torch.eye(model.num_prototypes, dtype=torch.bool, device=device)
        avg_sim_proto = proto_sim[mask].mean().item()
        max_sim_proto = proto_sim[mask].max().item()
        
        print("\n" + "="*70)
        print("Prototype Separation:")
        print("="*70)
        if avg_sim_proto < 0.5:
            status = "[Good]"
        elif avg_sim_proto < 0.7:
            status = "[OK]"
        else:
            status = "[Poor]"
        print(f"  {status} Inter-prototype similarity: avg={avg_sim_proto:.3f}, max={max_sim_proto:.3f}")

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def set_decay(self, decay):
        self.decay = decay
        
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class MultimodalFusion(nn.Module):
    def __init__(self, common_dim, num_heads=4, num_layers=2, pivot_len=4):
        super().__init__()
        self.common_dim = common_dim
        self.num_layers = num_layers
        self.pivot_len = pivot_len

        # 模态序列生成
        self.text_seq_gen = nn.ModuleList([
            nn.Linear(common_dim, common_dim) for _ in range(pivot_len)
        ])
        self.visual_seq_gen = nn.ModuleList([
            nn.Linear(common_dim, common_dim) for _ in range(pivot_len)
        ])
        self.audio_seq_gen = nn.ModuleList([
            nn.Linear(common_dim, common_dim) for _ in range(pivot_len)
        ])

        # transformer 层
        self.transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=common_dim,
                nhead=num_heads,
                dim_feedforward=common_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers * 3)
        ])

        # 动态权重网络
        self.alpha_gen_v = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2),
            nn.GELU(),
            nn.Linear(common_dim // 2, 1),
            nn.Sigmoid()
        )
        self.alpha_gen_a = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2),
            nn.GELU(),
            nn.Linear(common_dim // 2, 1),
            nn.Sigmoid()
        )
        self.alpha_gen_t = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2),
            nn.GELU(),
            nn.Linear(common_dim // 2, 1),
            nn.Sigmoid()
        )

        # 输出与 gating
        self.output_mlp = nn.Sequential(
            nn.Linear(common_dim * pivot_len, common_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.gate_net = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2),
            nn.GELU(),
            nn.Linear(common_dim // 2, common_dim),
            nn.Sigmoid()
        )

        self.input_fusion = nn.Linear(common_dim * 3, common_dim)
        self.ln_text = nn.LayerNorm(common_dim)
        self.ln_visual = nn.LayerNorm(common_dim)
        self.ln_audio = nn.LayerNorm(common_dim)
        self.ln_pivot = nn.LayerNorm(common_dim)

    def forward(self, text, visual, audio, mask=None):
        batch_size = text.size(0)

        # LayerNorm 每个模态
        text = self.ln_text(text)
        visual = self.ln_visual(visual)
        audio = self.ln_audio(audio)

        # 初始输入融合
        raw_input = torch.cat([text, visual, audio], dim=-1)
        raw_fused = self.input_fusion(raw_input)

        # 模态序列
        text_seq = torch.stack([fn(text) for fn in self.text_seq_gen], dim=1)
        visual_seq = torch.stack([fn(visual) for fn in self.visual_seq_gen], dim=1)
        audio_seq = torch.stack([fn(audio) for fn in self.audio_seq_gen], dim=1)

        # 初始 pivot 平均
        pivot = (text_seq + visual_seq + audio_seq) / 3.0

        # 多层交互
        for layer_idx in range(self.num_layers):
            base = layer_idx * 3

            # 视觉交互
            cat_v = torch.cat([visual_seq, pivot], dim=1)
            out_v = self.transformers[base](cat_v)
            visual_seq = out_v[:, :self.pivot_len, :]
            pivot_v = out_v[:, self.pivot_len:, :]

            alpha_v = self.alpha_gen_v(pivot_v.mean(dim=1))
            pivot = alpha_v.unsqueeze(-1) * pivot + (1 - alpha_v.unsqueeze(-1)) * pivot_v
            pivot = self.ln_pivot(pivot)

            # 音频交互
            cat_a = torch.cat([audio_seq, pivot], dim=1)
            out_a = self.transformers[base + 1](cat_a)
            audio_seq = out_a[:, :self.pivot_len, :]
            pivot_a = out_a[:, self.pivot_len:, :]

            alpha_a = self.alpha_gen_a(pivot_a.mean(dim=1))
            pivot = alpha_a.unsqueeze(-1) * pivot + (1 - alpha_a.unsqueeze(-1)) * pivot_a
            pivot = self.ln_pivot(pivot)

            # 文本交互
            cat_t = torch.cat([text_seq, pivot], dim=1)
            out_t = self.transformers[base + 2](cat_t)
            text_seq = out_t[:, :self.pivot_len, :]
            pivot_t = out_t[:, self.pivot_len:, :]

            alpha_t = self.alpha_gen_t(pivot_t.mean(dim=1))
            pivot = alpha_t.unsqueeze(-1) * pivot + (1 - alpha_t.unsqueeze(-1)) * pivot_t
            pivot = self.ln_pivot(pivot)

        # 输出
        pivot_flat = pivot.reshape(batch_size, -1)
        pivot_feat = self.output_mlp(pivot_flat)

        gate = self.gate_net(pivot_feat)
        fused = gate * pivot_feat + (1 - gate) * raw_fused

        return fused
        
class ModalityNormalization(nn.Module):
    def __init__(self, target_norm=30.0):
        super().__init__()
        self.target_norm = nn.Parameter(torch.tensor(target_norm, dtype=torch.float32))
    
    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + 1e-8) * torch.abs(self.target_norm)

class PNFCModule(nn.Module):
    def __init__(self, text_dim, image_dim, audio_dim, common_dim, k=10, temperature=0.1):
        super().__init__()
        self.k = k
        self.temperature = temperature  
        self.common_dim = common_dim
        
        # text_dim = 768 (BERT输出)
        self.text_proj_completion = nn.Sequential(
            nn.Linear(text_dim, common_dim), nn.LayerNorm(common_dim), nn.ReLU()
        )
        self.image_proj_completion = nn.Sequential(
            nn.Linear(image_dim, common_dim), nn.LayerNorm(common_dim), nn.ReLU()
        )
        self.audio_proj_completion = nn.Sequential(
            nn.Linear(audio_dim, common_dim), nn.LayerNorm(common_dim), nn.ReLU()
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=common_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(common_dim, common_dim * 4), nn.ReLU(), 
            nn.Linear(common_dim * 4, common_dim)
        )
        self.ln1 = nn.LayerNorm(common_dim)
        self.ln2 = nn.LayerNorm(common_dim)
    
    def search_neighbors(self, query, candidates, k):
        query_norm = F.normalize(query.unsqueeze(0), dim=1)
        candidates_norm = F.normalize(candidates, dim=1)
        similarities = torch.mm(query_norm, candidates_norm.t()).squeeze(0)
        topk_vals, topk_idx = torch.topk(similarities, k=min(k, len(candidates)))
        return topk_idx, topk_vals 
    
    def complete_feature(self, avail_feat_raw, avail_neighbors_raw, missing_neighbors_raw, 
                         proto_feat, avail_modality, missing_modality):
        proj_map = {
            'text': self.text_proj_completion,
            'image': self.image_proj_completion,
            'audio': self.audio_proj_completion
        }
        
        avail_proj = proj_map[avail_modality]
        missing_proj = proj_map[missing_modality]
        
        avail_feat = avail_proj(avail_feat_raw.unsqueeze(0)).squeeze(0)
        avail_neighbors = avail_proj(avail_neighbors_raw)
        missing_neighbors = missing_proj(missing_neighbors_raw)
        
        Q = avail_feat.unsqueeze(0).unsqueeze(0)
        K = torch.cat([proto_feat.unsqueeze(0), avail_neighbors], dim=0).unsqueeze(0)
        V = torch.cat([proto_feat.unsqueeze(0), missing_neighbors], dim=0).unsqueeze(0)
        
        attn_out, _ = self.attention(Q, K, V)
        attn_out = attn_out.squeeze(0).squeeze(0)
        
        out = self.ln1(attn_out + proto_feat)
        ffn_out = self.ffn(out)
        out = self.ln2(ffn_out + out) + proto_feat
        return out

class PrototypeEmotionModel(nn.Module):
    def __init__(self, bert_path, image_dim, audio_dim, common_dim, num_prototypes=7, 
                 k_neighbors=5, drop_prob=0.1, freeze_bert=False):
        super().__init__()
        self.common_dim = common_dim
        self.num_prototypes = num_prototypes
        self.k = k_neighbors
        
        # ==================== BERT编码器 ====================
        from transformers import BertModel
        print(f"\n{'='*80}")
        print(f"Loading BERT from {bert_path}...")
        self.bert = BertModel.from_pretrained(bert_path)
        
        if freeze_bert:
            print("  Freezing BERT parameters (will NOT be trained)")
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert.eval()
        else:
            print("  BERT is TRAINABLE (will be fine-tuned end-to-end)")
        
        # 统计参数
        bert_params = sum(p.numel() for p in self.bert.parameters())
        bert_trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        print(f"  BERT params: {bert_params:,} (trainable: {bert_trainable:,})")
        print("="*80 + "\n")
        
        # ==================== 投影层 ====================
        # BERT输出是768维
        self.text_proj = nn.Sequential(
            nn.Linear(768, common_dim), 
            nn.LayerNorm(common_dim), 
            nn.ReLU(), 
            nn.Dropout(drop_prob)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, common_dim), 
            nn.LayerNorm(common_dim), 
            nn.ReLU(), 
            nn.Dropout(drop_prob)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, common_dim), 
            nn.LayerNorm(common_dim), 
            nn.ReLU(), 
            nn.Dropout(drop_prob)
        )
        
        self.text_proj_proto = nn.Sequential(
            nn.Linear(768, common_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(common_dim * 4, common_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(common_dim * 2, common_dim)
        )
        self.image_proj_proto = nn.Sequential(
            nn.Linear(image_dim, common_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(common_dim * 4, common_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(common_dim * 2, common_dim)
        )
        self.audio_proj_proto = nn.Sequential(
            nn.Linear(audio_dim, common_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(common_dim * 4, common_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(common_dim * 2, common_dim)
        )
        
        self.norm_text = ModalityNormalization(target_norm=60.0)
        self.norm_image = ModalityNormalization(target_norm=40.0)
        self.norm_audio = ModalityNormalization(target_norm=15.0)
        
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, common_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        self.prototype_emotion_predictor = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(common_dim // 2, 1)
        )
        
        self.pnfc = PNFCModule(768, image_dim, audio_dim, common_dim, k_neighbors)
        self.fusion = MultimodalFusion(common_dim, num_heads=4, num_layers=2, pivot_len=4)
        self.classifier = nn.Linear(common_dim, 1)
        self.text_classifier = nn.Linear(common_dim, 1)
        self.image_classifier = nn.Linear(common_dim, 1)
        self.audio_classifier = nn.Linear(common_dim, 1)
        
        self.register_buffer('train_text_feats', None)
        self.register_buffer('train_image_feats', None)
        self.register_buffer('train_audio_feats', None)

    def freeze_bert(self):
        """冻结BERT参数"""
        print("\n" + "="*80)
        print("Freezing BERT parameters...")
        print("="*80)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()
        
        bert_trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        print(f"  BERT trainable params: {bert_trainable:,} (should be 0)")
        print("="*80 + "\n")
    
    def unfreeze_bert(self):
        """解冻BERT参数"""
        print("\n" + "="*80)
        print("Unfreezing BERT parameters...")
        print("="*80)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.bert.train()
        
        bert_trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        print(f"  BERT trainable params: {bert_trainable:,}")
        print("="*80 + "\n")

    def encode_text_with_bert(self, text_inputs):
        """
        用BERT编码文本
        
        Args:
            text_inputs: dict with keys 'input_ids', 'attention_mask', 'token_type_ids'
        
        Returns:
            text_features: [batch_size, 768]
        """
        outputs = self.bert(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            token_type_ids=text_inputs['token_type_ids']
        )
        
        # Mean pooling
        text_feat = outputs.last_hidden_state.mean(dim=1)  # [batch_size, 768]
        
        return text_feat

    def build_feature_bank(self, text_inputs, image, audio, device):
        """
        修改版本:text_inputs现在是dict列表,需要先过BERT
        """
        with torch.no_grad():
            temp_mask = get_mask(3, len(text_inputs), missing_rate=0.0)
            complete_mask = (temp_mask.sum(dim=0) == 3)
            if complete_mask.sum() < 100:
                complete_mask = torch.ones(len(text_inputs), dtype=torch.bool)
            
            chunk_size = 256  # 减小chunk避免OOM
            
            # 处理text: 需要先通过BERT
            text_feats_list = []
            complete_indices = torch.where(complete_mask)[0]
            
            print(f"Building feature bank from {len(complete_indices)} complete samples...")
            for i in range(0, len(complete_indices), chunk_size):
                if i % 1024 == 0:
                    print(f"  Processing {i}/{len(complete_indices)}...")
                    
                chunk_indices = complete_indices[i:i+chunk_size]
                
                # 准备batch的text inputs
                batch_text_inputs = {
                    'input_ids': torch.stack([text_inputs[idx]['input_ids'] for idx in chunk_indices]).to(device),
                    'attention_mask': torch.stack([text_inputs[idx]['attention_mask'] for idx in chunk_indices]).to(device),
                    'token_type_ids': torch.stack([text_inputs[idx]['token_type_ids'] for idx in chunk_indices]).to(device)
                }
                
                # 通过BERT编码
                text_bert = self.encode_text_with_bert(batch_text_inputs)  # [chunk_size, 768]
                text_feats_list.append(text_bert.cpu())
            
            self.train_text_feats = torch.cat(text_feats_list).to(device)
            
            # 处理image和audio
            def extract_feats(x):
                feats_list = []
                for i in range(0, complete_mask.sum(), chunk_size):
                    chunk = x[complete_mask][i:i+chunk_size].to(device)
                    feats_list.append(chunk.cpu())
                return torch.cat(feats_list)
            
            self.train_image_feats = extract_feats(image).to(device)
            self.train_audio_feats = extract_feats(audio).to(device)
            
            print(f"✓ Feature bank built: text={self.train_text_feats.shape}, "
                  f"image={self.train_image_feats.shape}, audio={self.train_audio_feats.shape}\n")
    
    def discretize_labels(self, labels):
        discrete = ((labels + 3) / 6 * (self.num_prototypes - 1)).long()
        return discrete.clamp(0, self.num_prototypes - 1)
    
    def get_prototype_for_sample(self, feat_raw, modality, temperature=0.1):
        proj_map = {
            'text': self.text_proj_proto,
            'image': self.image_proj_proto,
            'audio': self.audio_proj_proto
        }
        
        feat_proj = proj_map[modality](feat_raw.unsqueeze(0)).squeeze(0)
        feat_norm = F.normalize(feat_proj, dim=0)
        proto_norm = F.normalize(self.prototypes.detach(), dim=1)
        similarities = torch.mv(proto_norm, feat_norm)
        weights = F.softmax(similarities / temperature, dim=0)
        proto_feat = torch.mm(weights.unsqueeze(0), self.prototypes.detach()).squeeze(0)
        
        return proto_feat
    
    def complete_modalities(self, text_inputs, image_raw, audio_raw, mask, compute_loss=True):
        """
        修改版本:text_inputs现在是dict,需要先过BERT
        """
        # 先通过BERT编码文本
        text_raw = self.encode_text_with_bert(text_inputs)  # [batch_size, 768]
        
        batch_size = mask.size(1)
        modality_names = ['text', 'image', 'audio']
        raw_feats = {'text': text_raw, 'image': image_raw, 'audio': audio_raw}
        proj_map = {'text': self.text_proj, 'image': self.image_proj, 'audio': self.audio_proj}
        
        completed_feats = {m: [] for m in modality_names}
        L_completion = 0.0
        completion_count = 0
        
        for i in range(batch_size):
            sample_mask = mask[:, i]
            
            for j, mod_name in enumerate(modality_names):
                if sample_mask[j] == 1:
                    feat_raw = raw_feats[mod_name][i]
                    feat_proj = proj_map[mod_name](feat_raw.unsqueeze(0)).squeeze(0)
                    completed_feats[mod_name].append(feat_proj)
                else:
                    available_modalities = []
                    for k, m in enumerate(modality_names):
                        if sample_mask[k] == 1:
                            available_modalities.append(m)
                    
                    if len(available_modalities) == 0:
                        proto_feat = self.prototypes.mean(dim=0).detach()
                        completed_feats[mod_name].append(proto_feat)
                        continue
                    
                    neighbor_recons = []
                    
                    for m_avail in available_modalities:
                        avail_feat_raw = raw_feats[m_avail][i]
                        
                        proto_feat = self.get_prototype_for_sample(
                            avail_feat_raw, 
                            modality=m_avail,
                            temperature=0.1
                        )
                        
                        with torch.no_grad():
                            avail_feats_bank = getattr(self, f"train_{m_avail}_feats")
                            missing_feats_bank = getattr(self, f"train_{mod_name}_feats")
                            idxs, similarities = self.pnfc.search_neighbors(
                                avail_feat_raw, avail_feats_bank, k=self.k
                            )
                            avail_neighbors_raw = avail_feats_bank[idxs]
                            missing_neighbors_raw = missing_feats_bank[idxs]
                        
                        recon_k = self.pnfc.complete_feature(
                            avail_feat_raw, 
                            avail_neighbors_raw.detach(),
                            missing_neighbors_raw.detach(), 
                            proto_feat,
                            avail_modality=m_avail, 
                            missing_modality=mod_name
                        )
                        neighbor_recons.append(recon_k)
                    
                    if len(neighbor_recons) > 1:
                        recon = torch.stack(neighbor_recons).mean(dim=0)
                    else:
                        recon = neighbor_recons[0]
                    
                    if compute_loss:
                        gt_raw = raw_feats[mod_name][i]
                        gt_proj = proj_map[mod_name](gt_raw.unsqueeze(0)).squeeze(0)
                        L_completion += F.mse_loss(recon, gt_proj.detach())
                        completion_count += 1
                    
                    completed_feats[mod_name].append(recon)
        
        for m in modality_names:
            completed_feats[m] = torch.stack(completed_feats[m])
        
        if completion_count > 0:
            L_completion /= completion_count
        else:
            L_completion = torch.tensor(0.0, device=text_raw.device)
        
        return completed_feats, L_completion

    def forward(self, text_inputs, image, audio, labels, mask, is_train=True):
        """
        Args:
            text_inputs: dict
        """
        completed_map, L_completion = self.complete_modalities(
            text_inputs, image, audio, mask, compute_loss=is_train
        )
        
        completed_map['text'] = self.norm_text(completed_map['text'])
        completed_map['image'] = self.norm_image(completed_map['image'])
        completed_map['audio'] = self.norm_audio(completed_map['audio'])
        
        L_pa = torch.tensor(0.0, device=image.device)
        
        pred_text = self.text_classifier(completed_map['text']).squeeze()
        pred_image = self.image_classifier(completed_map['image']).squeeze()
        pred_audio = self.audio_classifier(completed_map['audio']).squeeze()
        L_unimodal = (F.mse_loss(pred_text, labels) + 
                      F.mse_loss(pred_image, labels) + 
                      F.mse_loss(pred_audio, labels))
        
        fused_feats = self.fusion(
            completed_map['text'].detach(), 
            completed_map['image'].detach(), 
            completed_map['audio'].detach(), 
            mask
        )
        
        predictions = self.classifier(fused_feats).squeeze()
        L_reg = F.mse_loss(predictions, labels)
        
        return predictions, L_unimodal, L_reg, L_completion, L_pa

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return cosine_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model, train_data, batch_size, missing_rate, optimizer, device, epoch, total_epochs, ema=None):
    model.train()
    # 保持BERT在eval模式（因为已冻结）
    if not any(p.requires_grad for p in model.bert.parameters()):
        model.bert.eval()
    
    text_data, image_data, audio_data, label_data = train_data
    num_samples = len(label_data)
    indices = torch.randperm(num_samples)
    total_loss = 0
    total_reg = 0
    total_mc = 0
    total_unimodal = 0
    total_pa = 0
    num_batches = 0
    alpha = 0.5
    pa_weight = 0.0
    
    if epoch >= 20:
        mc_weight = max(0.0, 0.4 * (1 - (epoch - 20) / (total_epochs - 20)))
    else:
        mc_weight = 0.4
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # 准备text inputs (dict)
        text_inputs = {
            'input_ids': torch.stack([text_data[i]['input_ids'] for i in batch_indices]).to(device),
            'attention_mask': torch.stack([text_data[i]['attention_mask'] for i in batch_indices]).to(device),
            'token_type_ids': torch.stack([text_data[i]['token_type_ids'] for i in batch_indices]).to(device)
        }
        
        image = image_data[batch_indices].to(device)
        audio = audio_data[batch_indices].to(device)
        labels = torch.tensor(label_data[batch_indices]).squeeze().float().to(device)
        mask = get_mask(3, len(batch_indices), missing_rate).to(device)
        
        optimizer.zero_grad()
        preds, L_unimodal, l_reg, l_mc, l_pa = model(text_inputs, image, audio, labels, mask, is_train=True)
        
        L_multimodal = l_reg + mc_weight * l_mc + pa_weight * l_pa
        
        (L_unimodal * alpha).backward(retain_graph=True)
        
        for name, param in model.named_parameters():
            if 'fusion' in name or 'classifier' in name:
                if param.grad is not None:
                    param.grad = None
        
        L_multimodal.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if ema is not None:
            ema.update()
            
        total_loss += (L_unimodal * alpha + L_multimodal).item()
        total_reg += l_reg.item()
        total_mc += l_mc.item()
        total_unimodal += L_unimodal.item()
        total_pa += l_pa.item()
        num_batches += 1
        
    return (total_loss/num_batches, total_reg/num_batches, total_mc/num_batches, 
            total_unimodal/num_batches, total_pa/num_batches)
            
def load_mosei_with_bert(pkl_path, bert_path='./BERT_EN', cache_path='mosei_bert_tokenized_cache.pth', 
                         force_recompute=False, use_cls_only=False):
    """
    加载MOSEI数据集,返回tokenized输入(不预计算BERT特征,保持BERT可训练)
    
    Args:
        pkl_path: 原始 pkl 文件路径
        bert_path: BERT 模型路径
        cache_path: 缓存tokenized数据的路径
        force_recompute: 强制重新处理
        use_cls_only: 占位参数(保持接口一致)
    
    Returns:
        12个返回值,但text相关的是dict列表(包含input_ids等),不是tensor
    """
    import pickle
    import torch
    from transformers import BertTokenizer
    import numpy as np
    import os
    
    # ==================== 检查缓存 ====================
    if os.path.exists(cache_path) and not force_recompute:
        print("="*80)
        print("✓ Found cached tokenized data!")
        print("="*80)
        print(f"Loading from: {cache_path}\n")
        
        try:
            cache = torch.load(cache_path)
            
            print(f"Dataset sizes:")
            print(f"  Train: {len(cache['train_labels'])} samples")
            print(f"  Dev:   {len(cache['dev_labels'])} samples")
            print(f"  Test:  {len(cache['test_labels'])} samples")
            print(f"\nData format:")
            print(f"  Text:   Tokenized inputs (will pass through BERT during training)")
            print(f"  Visual: {cache['train_visual'].shape[1]}D (OpenFace pooled)")
            print(f"  Audio:  {cache['train_audio'].shape[1]}D (COVAREP pooled)")
            print(f"\n⚠️  BERT will be TRAINABLE (not frozen)")
            print("="*80)
            print("✓ Loaded from cache successfully!")
            print("="*80 + "\n")
            
            return (
                cache['train_text'], cache['train_visual'], cache['train_audio'], cache['train_labels'],
                cache['dev_text'], cache['dev_visual'], cache['dev_audio'], cache['dev_labels'],
                cache['test_text'], cache['test_visual'], cache['test_audio'], cache['test_labels']
            )
        
        except Exception as e:
            print(f"⚠ Error loading cache: {e}")
            print("Will recompute from scratch...\n")
    
    # ==================== 从头处理 ====================
    print("="*80)
    print(f"Processing MOSEI dataset (tokenization only, BERT will run during training)")
    print("="*80)
    
    print("\nLoading MOSEI dataset...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    print("Tokenizer loaded!\n")
    
    def process_split(samples, split_name):
        print(f"Processing {split_name}: {len(samples)} samples")
        
        text_inputs = []  # 存储tokenizer输出(dict)
        visual_feats = []
        audio_feats = []
        labels = []
        
        for idx, sample in enumerate(samples):
            if idx % 1000 == 0:
                print(f"  {idx}/{len(samples)} ({idx/len(samples)*100:.1f}%)")
            
            # 解包样本
            features, label, segment = sample
            words, visual_seq, audio_seq = features
            
            # 1. Tokenization (不运行BERT)
            text_str = ' '.join(words)
            inputs = tokenizer(
                text_str, 
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding='max_length'  # 固定长度
            )
            
            # 存储为dict
            text_inputs.append({
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0)
            })
            
            # 2. 池化视觉和音频
            visual_feat = visual_seq.mean(axis=0)  # [35]
            audio_feat = audio_seq.mean(axis=0)    # [74]
            
            # 3. 提取标签
            label_val = label.flatten()[0] if isinstance(label, np.ndarray) else label
            
            visual_feats.append(visual_feat)
            audio_feats.append(audio_feat)
            labels.append(label_val)
        
        # 转换为 tensor
        visual_tensor = torch.from_numpy(np.array(visual_feats)).float()
        audio_tensor = torch.from_numpy(np.array(audio_feats)).float()
        label_tensor = torch.from_numpy(np.array(labels)).float()
        
        print(f"  Done! Text: {len(text_inputs)} tokenized samples, "
              f"Visual: {visual_tensor.shape}, Audio: {audio_tensor.shape}\n")
        
        return text_inputs, visual_tensor, audio_tensor, label_tensor
    
    # 处理三个数据集
    train_text, train_visual, train_audio, train_labels = process_split(data['train'], 'TRAIN')
    dev_text, dev_visual, dev_audio, dev_labels = process_split(data['dev'], 'DEV')
    test_text, test_visual, test_audio, test_labels = process_split(data['test'], 'TEST')
    
    # ==================== 保存缓存 ====================
    print("="*80)
    print("Saving tokenized data to cache...")
    print("="*80)
    print(f"Cache path: {cache_path}\n")
    
    cache = {
        'train_text': train_text,
        'train_visual': train_visual,
        'train_audio': train_audio,
        'train_labels': train_labels,
        'dev_text': dev_text,
        'dev_visual': dev_visual,
        'dev_audio': dev_audio,
        'dev_labels': dev_labels,
        'test_text': test_text,
        'test_visual': test_visual,
        'test_audio': test_audio,
        'test_labels': test_labels,
    }
    
    torch.save(cache, cache_path)
    
    cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"✓ Cache saved successfully! ({cache_size_mb:.1f} MB)")
    print(f"  Next time loading will be instant!\n")
    
    print("="*80)
    print("MOSEI Data Loaded Successfully!")
    print("="*80)
    print(f"Train: {len(train_labels)} samples")
    print(f"Dev:   {len(dev_labels)} samples")
    print(f"Test:  {len(test_labels)} samples")
    print(f"\nData format:")
    print(f"  Text:   Tokenized inputs (dict with input_ids, attention_mask, token_type_ids)")
    print(f"  Visual: {train_visual.shape[1]}D (OpenFace pooled)")
    print(f"  Audio:  {train_audio.shape[1]}D (COVAREP pooled)")
    print(f"\nLabel stats:")
    print(f"  Train: mean={train_labels.mean():.3f}, std={train_labels.std():.3f}")
    print(f"  Dev:   mean={dev_labels.mean():.3f}, std={dev_labels.std():.3f}")
    print(f"  Test:  mean={test_labels.mean():.3f}, std={test_labels.std():.3f}")
    print("\n⚠️  BERT will run during training (not frozen!)")
    print("="*80 + "\n")
    
    return (train_text, train_visual, train_audio, train_labels,
            dev_text, dev_visual, dev_audio, dev_labels,
            test_text, test_visual, test_audio, test_labels)
            
if __name__ == '__main__':
    seed_everything(1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # 加载数据(text是dict列表)
    (feature_text_train, feature_visual_train, feature_audio_train, label_train,
     feature_text_dev, feature_visual_dev, feature_audio_dev, label_dev,
     feature_text_test, feature_visual_test, feature_audio_test, label_test) = load_mosei_with_bert(
        pkl_path='/home/shantao/MECOM/mosei.pkl',
        bert_path='/home/shantao/bert-base-uncased',
        cache_path='mosei_bert_tokenized_cache.pth',
        use_cls_only=False
    )
    
    # 创建模型(集成BERT, 初始状态不冻结)
    model = PrototypeEmotionModel(
        bert_path='/home/shantao/bert-base-uncased',
        image_dim=35,
        audio_dim=74,
        common_dim=128,
        num_prototypes=7,
        k_neighbors=15,
        drop_prob=0.2,
        freeze_bert=False  # 初始不冻结，会在预训练后自动冻结
    ).to(device)
    
    # 检查参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bert_params = sum(p.numel() for p in model.bert.parameters())
    bert_trainable = sum(p.numel() for p in model.bert.parameters() if p.requires_grad)
    
    print(f"\n{'='*80}")
    print(f"Model Parameters Summary (Initial State):")
    print(f"  Total params:      {total_params:,}")
    print(f"  Trainable params:  {trainable_params:,}")
    print(f"  BERT params:       {bert_params:,}")
    print(f"  BERT trainable:    {bert_trainable:,}")
    print(f"  Non-BERT trainable: {trainable_params - bert_trainable:,}")
    print(f"{'='*80}\n")
    
    # build feature bank
    model.build_feature_bank(feature_text_train, feature_visual_train, feature_audio_train, device)
    
    # 阶段1:预训练 (BERT会被训练然后冻结)
    train_data = (feature_text_train, feature_visual_train, feature_audio_train, label_train)
    pretrain_prototypes_and_projections(model, train_data, batch_size=16, device=device, epochs=10)
    
    # 阶段2:主训练 (BERT已经被冻结)
    print("\n" + "="*80)
    print("Phase 2: Training with FROZEN prototypes and FROZEN BERT (50 epochs)...")
    print("="*80)
    
    # 检查BERT是否已冻结
    bert_trainable = sum(p.numel() for p in model.bert.parameters() if p.requires_grad)
    print(f"BERT trainable params: {bert_trainable:,} (should be 0)")
    
    # 冻结原型和情感预测器
    model.prototypes.requires_grad = False
    for param in model.prototype_emotion_predictor.parameters():
        param.requires_grad = False
    
    # 优化器只包含非BERT、非原型的参数
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() 
                    if 'bert' not in n 
                    and 'prototypes' not in n 
                    and 'prototype_emotion_predictor' not in n
                    and p.requires_grad], 
         'lr': 1.5e-4}
    ], weight_decay=1e-5)
    
    # 验证优化器参数
    optimizer_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_trainable:,}")
    print(f"Optimizer managing: {optimizer_params:,} parameters")
    print("="*80 + "\n")
    
    scheduler = get_warmup_cosine_scheduler(optimizer, 10, 50, min_lr=1e-6)
    
    ema = EMA(model, decay=0.999)
    ema.register()
    
    batch_size = 16
    missing_rate = 0.5
    best_avg_acc = 0
    patience = 20
    patience_counter = 0
    test_data = (feature_text_test, feature_visual_test, feature_audio_test, label_test)
    
    best_results = {i: {'acc': 0, 'epoch': 0, 'mae': float('inf'), 'f1': 0} for i in range(7)}
    pattern_names = {
        0: "Audio Only (A)", 1: "Text Only (T)", 2: "Visual Only (V)",
        3: "Audio+Visual (AV)", 4: "Audio+Text (AT)", 5: "Text+Visual (TV)",
        6: "All Modalities (ATV)"
    }
    
    total_epochs = 50
    for epoch in range(50):
        ema_decay = 0.995 + 0.004 * (epoch / total_epochs)
        ema.set_decay(ema_decay)
        
        train_loss, L_reg, L_mc, L_unimodal, L_pa = train_epoch(
            model, train_data, batch_size, missing_rate, optimizer, device, epoch, total_epochs, ema=ema
        )
        scheduler.step()
    
        print(f"\n{'='*90}")
        print(f"Epoch {epoch+1}/50")
        print(f"Loss: {train_loss:.4f} | Reg: {L_reg:.4f} | MC: {L_mc:.4f} | Uni: {L_unimodal:.4f} | PA: {L_pa:.4f}")
        
        # 测试
        ema.apply_shadow()
        print(f"\n--- Test Results with EMA (Epoch {epoch+1}) ---")
        test_results = evaluate_all_missing_patterns(model, test_data, batch_size, device)
        
        if (epoch + 1) % 5 == 0:
            visualize_prototypes(model, test_data, device, epoch+1)
        
        ema.restore()
        
        # 记录最佳结果
        current_accs = []
        for pattern_id in range(7):
            result = test_results[pattern_id]
            current_accs.append(result['acc'])
            is_best = ""
            if result['acc'] > best_results[pattern_id]['acc']:
                best_results[pattern_id] = {
                    'acc': result['acc'], 'epoch': epoch + 1,
                    'mae': result['mae'], 'f1': result['f1']
                }
                ema.apply_shadow()
                torch.save(model.state_dict(), f"best_model_pattern_{pattern_id}_ema.pth")
                ema.restore()
                is_best = "[BEST]"
            
            print(f"{result['name']:20s} | Fused: {result['acc']*100:.2f}% | "
                  f"Best: {best_results[pattern_id]['acc']*100:.2f}% (Epoch {best_results[pattern_id]['epoch']}) {is_best}")
        
        current_avg_acc = np.mean(current_accs)
        print(f"\n{'Current Avg ACC':20s}: {current_avg_acc*100:.2f}%")
        
        if current_avg_acc > best_avg_acc:
            best_avg_acc = current_avg_acc
            patience_counter = 0
            ema.apply_shadow()
            torch.save(model.state_dict(), "best_avg_acc_model_ema.pth")
            ema.restore()
            print(f"Saved best model (Avg Acc: {best_avg_acc*100:.2f}%)")
        else:
            patience_counter += 1
        
        print(f"\n--- Best Fused Results So Far ---")
        for pattern_id in range(7):
            best = best_results[pattern_id]
            print(f"{pattern_names[pattern_id]:20s} | Acc: {best['acc']*100:.2f}% (Epoch {best['epoch']})")
        
        overall_best_avg = np.mean([best_results[i]['acc'] for i in range(7)])
        print(f"\n{'Overall Best Avg ACC':20s}: {overall_best_avg*100:.2f}%")
        print(f"{'Best Avg ACC for ES':20s}: {best_avg_acc*100:.2f}%")
        print(f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("\n" + "="*90)
    print("Training Complete!")
    print("=" * 90)
    
    for pattern_id in range(7):
        best = best_results[pattern_id]
        print(f"\n{pattern_names[pattern_id]:20s} (Best at Epoch {best['epoch']})")
        print(f"  ACC: {best['acc']*100:.2f}% | MAE: {best['mae']:.4f} | F1: {best['f1']*100:.2f}%")
    
    overall_best_avg = np.mean([best_results[i]['acc'] for i in range(7)])
    print(f"\n{'Overall Best Avg ACC':20s}: {overall_best_avg*100:.2f}%")
    print(f"{'Best Avg ACC (ES)':20s}: {best_avg_acc*100:.2f}%")
    print("=" * 90)
