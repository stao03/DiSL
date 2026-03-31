import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from sklearn.metrics import f1_score, accuracy_score

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
        label_data = torch.from_numpy(label_data).long()
    else:
        label_data = label_data.long()
    
    num_samples = len(label_data)
    all_preds = []
    all_preds_text = []
    all_preds_visual = []
    all_preds_audio = []
    all_labels = []
    
    missing_patterns = {
        0: [0, 1], 1: [1, 2], 2: [0, 2],
        3: [0], 4: [1], 5: [2]
    }
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            text = text_data[start_idx:end_idx].to(device)
            image = image_data[start_idx:end_idx].to(device)
            audio = audio_data[start_idx:end_idx].to(device)
            labels = label_data[start_idx:end_idx].to(device)
            
            batch_len = len(text)
            mask = torch.ones(3, batch_len).to(device)
            for idx in missing_patterns[missing_pattern]:
                mask[idx, :] = 0
            
            completed_map, _ = model.complete_modalities(text, image, audio, mask, compute_loss=False)
            completed_map['text'] = model.norm_text(completed_map['text'])
            completed_map['image'] = model.norm_image(completed_map['image'])
            completed_map['audio'] = model.norm_audio(completed_map['audio'])
            
            pred_text = model.text_classifier(completed_map['text']).view(-1, 4, 2)
            pred_visual = model.image_classifier(completed_map['image']).view(-1, 4, 2)
            pred_audio = model.audio_classifier(completed_map['audio']).view(-1, 4, 2)
            
            fused_feats = model.fusion(completed_map['text'].detach(), 
                                       completed_map['image'].detach(), 
                                       completed_map['audio'].detach(), mask)
            preds = model.classifier(fused_feats).view(-1, 4, 2)
            
            all_preds.append(preds.cpu())
            all_preds_text.append(pred_text.cpu())
            all_preds_visual.append(pred_visual.cpu())
            all_preds_audio.append(pred_audio.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_preds_text = torch.cat(all_preds_text)
    all_preds_visual = torch.cat(all_preds_visual)
    all_preds_audio = torch.cat(all_preds_audio)
    all_labels = torch.cat(all_labels)
    
    acc_per_emo = []
    f1_weighted_per_emo = []
    f1_macro_per_emo = []
    
    for emo_idx in range(4):
        pred_classes = torch.argmax(all_preds[:, emo_idx, :], dim=1).numpy()
        label_classes = all_labels[:, emo_idx].numpy()
        
        acc_per_emo.append(accuracy_score(label_classes, pred_classes))
        f1_weighted_per_emo.append(f1_score(label_classes, pred_classes, average='weighted'))
        f1_macro_per_emo.append(f1_score(label_classes, pred_classes, average='macro'))
    
    acc = np.mean(acc_per_emo)
    f1_weighted = np.mean(f1_weighted_per_emo)
    f1_macro = np.mean(f1_macro_per_emo)
    
    acc_text = np.mean([accuracy_score(all_labels[:, i].numpy(), 
                                       torch.argmax(all_preds_text[:, i, :], dim=1).numpy()) 
                       for i in range(4)])
    acc_visual = np.mean([accuracy_score(all_labels[:, i].numpy(), 
                                         torch.argmax(all_preds_visual[:, i, :], dim=1).numpy()) 
                         for i in range(4)])
    acc_audio = np.mean([accuracy_score(all_labels[:, i].numpy(), 
                                        torch.argmax(all_preds_audio[:, i, :], dim=1).numpy()) 
                        for i in range(4)])
    
    return acc, f1_weighted, f1_macro, acc_text, acc_visual, acc_audio

def evaluate_complete_modality(model, test_data, batch_size, device):
    model.eval()
    text_data, image_data, audio_data, label_data = test_data
    
    if isinstance(label_data, np.ndarray):
        label_data = torch.from_numpy(label_data).long()
    else:
        label_data = label_data.long()
    
    num_samples = len(label_data)
    all_preds = []
    all_preds_text = []
    all_preds_visual = []
    all_preds_audio = []
    all_labels = []
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            text = text_data[start_idx:end_idx].to(device)
            image = image_data[start_idx:end_idx].to(device)
            audio = audio_data[start_idx:end_idx].to(device)
            labels = label_data[start_idx:end_idx].to(device)
            
            batch_len = len(text)
            mask = torch.ones(3, batch_len).to(device)  
            
            completed_map, _ = model.complete_modalities(text, image, audio, mask, compute_loss=False)
            completed_map['text'] = model.norm_text(completed_map['text'])
            completed_map['image'] = model.norm_image(completed_map['image'])
            completed_map['audio'] = model.norm_audio(completed_map['audio'])
            
            pred_text = model.text_classifier(completed_map['text']).view(-1, 4, 2)
            pred_visual = model.image_classifier(completed_map['image']).view(-1, 4, 2)
            pred_audio = model.audio_classifier(completed_map['audio']).view(-1, 4, 2)
            
            fused_feats = model.fusion(completed_map['text'].detach(), 
                                       completed_map['image'].detach(), 
                                       completed_map['audio'].detach(), mask)
            preds = model.classifier(fused_feats).view(-1, 4, 2)
            
            all_preds.append(preds.cpu())
            all_preds_text.append(pred_text.cpu())
            all_preds_visual.append(pred_visual.cpu())
            all_preds_audio.append(pred_audio.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_preds_text = torch.cat(all_preds_text)
    all_preds_visual = torch.cat(all_preds_visual)
    all_preds_audio = torch.cat(all_preds_audio)
    all_labels = torch.cat(all_labels)
    
    acc_per_emo = []
    f1_weighted_per_emo = []
    f1_macro_per_emo = []
    
    for emo_idx in range(4):
        pred_classes = torch.argmax(all_preds[:, emo_idx, :], dim=1).numpy()
        label_classes = all_labels[:, emo_idx].numpy()
        
        acc_per_emo.append(accuracy_score(label_classes, pred_classes))
        f1_weighted_per_emo.append(f1_score(label_classes, pred_classes, average='weighted'))
        f1_macro_per_emo.append(f1_score(label_classes, pred_classes, average='macro'))
    
    acc = np.mean(acc_per_emo)
    f1_weighted = np.mean(f1_weighted_per_emo)
    f1_macro = np.mean(f1_macro_per_emo)
    
    acc_text = np.mean([accuracy_score(all_labels[:, i].numpy(), 
                                       torch.argmax(all_preds_text[:, i, :], dim=1).numpy()) 
                       for i in range(4)])
    acc_visual = np.mean([accuracy_score(all_labels[:, i].numpy(), 
                                         torch.argmax(all_preds_visual[:, i, :], dim=1).numpy()) 
                         for i in range(4)])
    acc_audio = np.mean([accuracy_score(all_labels[:, i].numpy(), 
                                        torch.argmax(all_preds_audio[:, i, :], dim=1).numpy()) 
                        for i in range(4)])
    
    return acc, f1_weighted, f1_macro, acc_text, acc_visual, acc_audio

def evaluate_all_missing_patterns(model, test_data, batch_size, device):
    pattern_names = {0: "A-only", 1: "T-only", 2: "V-only", 3: "AV", 4: "AT", 5: "TV", 6: "Complete"}
    results = {}
    print(f"{'Pattern':<10} | {'Fus':<5} | {'T':<5} | {'V':<5} | {'A':<5} | {'F1-W':<5} | {'F1-M':<5} | {'Gap':<6}")
    print("-" * 85)

    for pattern_id in range(6):
        acc, f1_w, f1_m, acc_text, acc_visual, acc_audio = evaluate_single_missing_pattern(
            model, test_data, batch_size, device, pattern_id
        )
        best_single = max(acc_text, acc_visual, acc_audio)
        gap = acc - best_single
        results[pattern_id] = {
            'name': pattern_names[pattern_id], 'acc': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m,
            'acc_text': acc_text, 'acc_visual': acc_visual, 'acc_audio': acc_audio, 'gap': gap
        }
        print(f"{pattern_names[pattern_id]:<10} | {acc*100:<5.1f} | {acc_text*100:<5.1f} | "
              f"{acc_visual*100:<5.1f} | {acc_audio*100:<5.1f} | {f1_w*100:<5.1f} | {f1_m*100:<5.1f} | {gap*100:+6.1f}")

    acc, f1_w, f1_m, acc_text, acc_visual, acc_audio = evaluate_complete_modality(
        model, test_data, batch_size, device
    )
    best_single = max(acc_text, acc_visual, acc_audio)
    gap = acc - best_single
    results[6] = {
        'name': pattern_names[6], 'acc': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m,
        'acc_text': acc_text, 'acc_visual': acc_visual, 'acc_audio': acc_audio, 'gap': gap
    }
    print(f"{pattern_names[6]:<10} | {acc*100:<5.1f} | {acc_text*100:<5.1f} | "
          f"{acc_visual*100:<5.1f} | {acc_audio*100:<5.1f} | {f1_w*100:<5.1f} | {f1_m*100:<5.1f} | {gap*100:+6.1f}")

    avg_gap = np.mean([results[i]['gap'] for i in range(7)])
    avg_acc = np.mean([results[i]['acc'] for i in range(7)])
    avg_f1_w = np.mean([results[i]['f1_weighted'] for i in range(7)])
    print("-" * 85)
    print(f"{'AVG':<10} | {avg_acc*100:<5.1f} | {'--':<5} | {'--':<5} | {'--':<5} | {avg_f1_w*100:<5.1f} | {'--':<5} | {avg_gap*100:+6.1f}")
    return results

class MultimodalFusion(nn.Module):
    def __init__(self, common_dim, num_heads=4, num_layers=2, pivot_len=4):
        super().__init__()
        self.common_dim = common_dim
        self.num_layers = num_layers
        self.pivot_len = pivot_len

        self.text_seq_gen = nn.ModuleList([nn.Linear(common_dim, common_dim) for _ in range(pivot_len)])
        self.visual_seq_gen = nn.ModuleList([nn.Linear(common_dim, common_dim) for _ in range(pivot_len)])
        self.audio_seq_gen = nn.ModuleList([nn.Linear(common_dim, common_dim) for _ in range(pivot_len)])

        self.transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=common_dim, nhead=num_heads, dim_feedforward=common_dim * 4,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(num_layers * 3)
        ])

        self.alpha_gen_v = nn.Sequential(nn.Linear(common_dim, common_dim // 2), nn.GELU(), nn.Linear(common_dim // 2, 1), nn.Sigmoid())
        self.alpha_gen_a = nn.Sequential(nn.Linear(common_dim, common_dim // 2), nn.GELU(), nn.Linear(common_dim // 2, 1), nn.Sigmoid())
        self.alpha_gen_t = nn.Sequential(nn.Linear(common_dim, common_dim // 2), nn.GELU(), nn.Linear(common_dim // 2, 1), nn.Sigmoid())

        self.output_mlp = nn.Sequential(nn.Linear(common_dim * pivot_len, common_dim), nn.GELU(), nn.Dropout(0.1))
        self.gate_net = nn.Sequential(nn.Linear(common_dim, common_dim // 2), nn.GELU(), nn.Linear(common_dim // 2, common_dim), nn.Sigmoid())
        self.input_fusion = nn.Linear(common_dim * 3, common_dim)
        self.ln_text = nn.LayerNorm(common_dim)
        self.ln_visual = nn.LayerNorm(common_dim)
        self.ln_audio = nn.LayerNorm(common_dim)
        self.ln_pivot = nn.LayerNorm(common_dim)

    def forward(self, text, visual, audio, mask=None):
        batch_size = text.size(0)
        text = self.ln_text(text)
        visual = self.ln_visual(visual)
        audio = self.ln_audio(audio)

        raw_input = torch.cat([text, visual, audio], dim=-1)
        raw_fused = self.input_fusion(raw_input)

        text_seq = torch.stack([fn(text) for fn in self.text_seq_gen], dim=1)
        visual_seq = torch.stack([fn(visual) for fn in self.visual_seq_gen], dim=1)
        audio_seq = torch.stack([fn(audio) for fn in self.audio_seq_gen], dim=1)
        pivot = (text_seq + visual_seq + audio_seq) / 3.0

        for layer_idx in range(self.num_layers):
            base = layer_idx * 3

            cat_v = torch.cat([visual_seq, pivot], dim=1)
            out_v = self.transformers[base](cat_v)
            visual_seq = out_v[:, :self.pivot_len, :]
            pivot_v = out_v[:, self.pivot_len:, :]
            alpha_v = self.alpha_gen_v(pivot_v.mean(dim=1))
            pivot = alpha_v.unsqueeze(-1) * pivot + (1 - alpha_v.unsqueeze(-1)) * pivot_v
            pivot = self.ln_pivot(pivot)

            cat_a = torch.cat([audio_seq, pivot], dim=1)
            out_a = self.transformers[base + 1](cat_a)
            audio_seq = out_a[:, :self.pivot_len, :]
            pivot_a = out_a[:, self.pivot_len:, :]
            alpha_a = self.alpha_gen_a(pivot_a.mean(dim=1))
            pivot = alpha_a.unsqueeze(-1) * pivot + (1 - alpha_a.unsqueeze(-1)) * pivot_a
            pivot = self.ln_pivot(pivot)

            cat_t = torch.cat([text_seq, pivot], dim=1)
            out_t = self.transformers[base + 2](cat_t)
            text_seq = out_t[:, :self.pivot_len, :]
            pivot_t = out_t[:, self.pivot_len:, :]
            alpha_t = self.alpha_gen_t(pivot_t.mean(dim=1))
            pivot = alpha_t.unsqueeze(-1) * pivot + (1 - alpha_t.unsqueeze(-1)) * pivot_t
            pivot = self.ln_pivot(pivot)

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
        is_zero = (norm < 1e-6)
        normalized = x / (norm + 1e-8) * torch.abs(self.target_norm)
        result = torch.where(
            is_zero.expand_as(x),
            torch.zeros_like(x),
            normalized
        )
        return result

class PNFCModule(nn.Module):
    def __init__(self, text_dim, image_dim, audio_dim, common_dim, k=10, temperature=0.1):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.common_dim = common_dim
        
        self.text_proj_completion = nn.Sequential(nn.Linear(text_dim, common_dim), nn.LayerNorm(common_dim), nn.ReLU())
        self.image_proj_completion = nn.Sequential(nn.Linear(image_dim, common_dim), nn.LayerNorm(common_dim), nn.ReLU())
        self.audio_proj_completion = nn.Sequential(nn.Linear(audio_dim, common_dim), nn.LayerNorm(common_dim), nn.ReLU())
        
        self.attention = nn.MultiheadAttention(embed_dim=common_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(common_dim, common_dim * 4), nn.ReLU(), nn.Linear(common_dim * 4, common_dim))
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
    def __init__(self, text_dim, image_dim, audio_dim, common_dim, num_classes=4, k_neighbors=5, drop_prob=0.1):
        super().__init__()
        self.common_dim = common_dim
        self.num_prototypes = num_classes
        self.num_classes = num_classes
        self.k = k_neighbors
        
        self.text_proj = nn.Sequential(nn.Linear(text_dim, common_dim), nn.LayerNorm(common_dim), nn.ReLU(), nn.Dropout(drop_prob))
        self.image_proj = nn.Sequential(nn.Linear(image_dim, common_dim), nn.LayerNorm(common_dim), nn.ReLU(), nn.Dropout(drop_prob))
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, common_dim), nn.LayerNorm(common_dim), nn.ReLU(), nn.Dropout(drop_prob))
        
        self.text_proj_proto = nn.Sequential(
            nn.Linear(text_dim, common_dim * 4), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(common_dim * 4, common_dim * 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(common_dim * 2, common_dim)
        )
        self.image_proj_proto = nn.Sequential(
            nn.Linear(image_dim, common_dim * 4), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(common_dim * 4, common_dim * 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(common_dim * 2, common_dim)
        )
        self.audio_proj_proto = nn.Sequential(
            nn.Linear(audio_dim, common_dim * 4), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(common_dim * 4, common_dim * 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(common_dim * 2, common_dim)
        )
        
        self.norm_text = ModalityNormalization(target_norm=60.0)
        self.norm_image = ModalityNormalization(target_norm=40.0)
        self.norm_audio = ModalityNormalization(target_norm=15.0)
        
        self.prototypes = nn.Parameter(torch.randn(num_classes, common_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        self.prototype_class_predictor = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(common_dim // 2, num_classes)
        )
        
        self.pnfc = PNFCModule(text_dim, image_dim, audio_dim, common_dim, k_neighbors)
        self.fusion = MultimodalFusion(common_dim, num_heads=4, num_layers=2, pivot_len=4)
        
        self.classifier = nn.Linear(common_dim, num_classes * 2)
        self.text_classifier = nn.Linear(common_dim, num_classes * 2)
        self.image_classifier = nn.Linear(common_dim, num_classes * 2)
        self.audio_classifier = nn.Linear(common_dim, num_classes * 2)
        
        self.register_buffer('train_text_feats', None)
        self.register_buffer('train_image_feats', None)
        self.register_buffer('train_audio_feats', None)

    def build_feature_bank(self, text, image, audio, device):
        with torch.no_grad():
            temp_mask = get_mask(3, len(text), missing_rate=0.0)
            complete_mask = (temp_mask.sum(dim=0) == 3)
            if complete_mask.sum() < 100:
                complete_mask = torch.ones(len(text), dtype=torch.bool)
            
            chunk_size = 1024
            def extract_feats(x):
                feats_list = []
                for i in range(0, complete_mask.sum(), chunk_size):
                    chunk = x[complete_mask][i:i+chunk_size].to(device)
                    feats_list.append(chunk.cpu())
                return torch.cat(feats_list)
            
            self.train_text_feats = extract_feats(text).to(device)
            self.train_image_feats = extract_feats(image).to(device)
            self.train_audio_feats = extract_feats(audio).to(device)
    
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
    
    def complete_modalities(self, text_raw, image_raw, audio_raw, mask, compute_loss=True):
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
                    available_modalities = [m for k, m in enumerate(modality_names) if sample_mask[k] == 1]
                    
                    if len(available_modalities) == 0:
                        proto_feat = self.prototypes.mean(dim=0).detach()
                        completed_feats[mod_name].append(proto_feat)
                        continue
                    
                    neighbor_recons = []
                    
                    for m_avail in available_modalities:
                        avail_feat_raw = raw_feats[m_avail][i]
                        proto_feat = self.get_prototype_for_sample(avail_feat_raw, modality=m_avail, temperature=0.1)
                        
                        with torch.no_grad():
                            avail_feats_bank = getattr(self, f"train_{m_avail}_feats")
                            missing_feats_bank = getattr(self, f"train_{mod_name}_feats")
                            idxs, similarities = self.pnfc.search_neighbors(avail_feat_raw, avail_feats_bank, k=self.k)
                            avail_neighbors_raw = avail_feats_bank[idxs]
                            missing_neighbors_raw = missing_feats_bank[idxs]
                        
                        recon_k = self.pnfc.complete_feature(
                            avail_feat_raw, avail_neighbors_raw.detach(), missing_neighbors_raw.detach(), 
                            proto_feat, avail_modality=m_avail, missing_modality=mod_name
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

    def forward(self, text, image, audio, labels, mask, is_train=True, current_epoch=0):
        completed_map, L_completion = self.complete_modalities(text, image, audio, mask, compute_loss=is_train)
        
        completed_map['text'] = self.norm_text(completed_map['text'])
        completed_map['image'] = self.norm_image(completed_map['image'])
        completed_map['audio'] = self.norm_audio(completed_map['audio'])
        
        L_pa = torch.tensor(0.0, device=text.device)
        
        pred_text = self.text_classifier(completed_map['text']).view(-1, 4, 2)
        pred_image = self.image_classifier(completed_map['image']).view(-1, 4, 2)
        pred_audio = self.audio_classifier(completed_map['audio']).view(-1, 4, 2)
        
        weight = 1.0 if current_epoch < 10 else 1.5
        
        class_weights = [
            torch.tensor([1.0, weight], device=text.device),  # Happy
            torch.tensor([1.0, 1.0], device=text.device),               # Sad
            torch.tensor([1.0, 1.0], device=text.device),                # Angry  
            torch.tensor([1.0, 1.0], device=text.device),               # Neutral
        ]
        
        L_unimodal = 0
        for emo_idx in range(4):
            L_unimodal += F.cross_entropy(pred_text[:, emo_idx, :], labels[:, emo_idx], 
                                          weight=class_weights[emo_idx])
            L_unimodal += F.cross_entropy(pred_image[:, emo_idx, :], labels[:, emo_idx],
                                          weight=class_weights[emo_idx])
            L_unimodal += F.cross_entropy(pred_audio[:, emo_idx, :], labels[:, emo_idx],
                                          weight=class_weights[emo_idx])
       
        fused_feats = self.fusion(
            completed_map['text'].detach(), 
            completed_map['image'].detach(), 
            completed_map['audio'].detach(), 
            mask
        )
        predictions = self.classifier(fused_feats).view(-1, 4, 2)
        
        L_cls = 0
        for emo_idx in range(4):
            L_cls += F.cross_entropy(predictions[:, emo_idx, :], labels[:, emo_idx],
                                    weight=class_weights[emo_idx])
        
        return predictions, L_unimodal, L_cls, L_completion, L_pa

def compute_prototype_class_loss(model, labels, device):
    proto_logits = model.prototype_class_predictor(model.prototypes)
    total_loss = 0.0
    valid_prototypes = 0
    
    for k in range(model.num_prototypes):
        mask_k = (labels[:, k] == 1)
        if mask_k.sum() > 0:
            target = torch.full((1,), k, dtype=torch.long, device=device)
            loss_k = F.cross_entropy(proto_logits[k:k+1], target)
            total_loss += loss_k
            valid_prototypes += 1
    
    if valid_prototypes > 0:
        return total_loss / valid_prototypes
    else:
        return torch.tensor(0.0, device=device)

def prototype_separation_loss(prototypes, margin=0.5):
    proto_norm = F.normalize(prototypes, dim=1)
    sim_matrix = torch.mm(proto_norm, proto_norm.t())
    mask = ~torch.eye(len(prototypes), dtype=torch.bool, device=prototypes.device)
    off_diag = sim_matrix[mask]
    loss = F.relu(off_diag - margin).mean()
    return loss

def pretrain_prototypes_and_projections(model, train_data, batch_size, device, epochs=40):
    text_data, image_data, audio_data, label_data = train_data
    model.prototypes.requires_grad = True
    
    optimizer = torch.optim.AdamW([
        {'params': model.text_proj_proto.parameters(), 'lr': 5e-4},
        {'params': model.image_proj_proto.parameters(), 'lr': 5e-4},
        {'params': model.audio_proj_proto.parameters(), 'lr': 1e-3},
        {'params': [model.prototypes], 'lr': 1e-3},
        {'params': model.prototype_class_predictor.parameters(), 'lr': 1e-3},
        {'params': model.text_classifier.parameters(), 'lr': 5e-4},
        {'params': model.image_classifier.parameters(), 'lr': 5e-4},
        {'params': model.audio_classifier.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-4)
    
    stage1_epochs = 20
    
    for epoch in range(epochs):
        model.train()
        num_samples = len(label_data)
        indices = torch.randperm(num_samples)
        
        if epoch < stage1_epochs:
            stage = 1
            ce_weight, cls_weight, class_weight, sep_weight = 2.0, 0.5, 0.1, 0.3
            for param_group in optimizer.param_groups:
                if len(param_group['params']) == 1 and param_group['params'][0].shape == model.prototypes.shape:
                    param_group['lr'] = 1e-4
        else:
            stage = 2
            ce_weight, cls_weight, class_weight, sep_weight = 0.5, 2.0, 1.5, 1.0
            if epoch == stage1_epochs:
                for i, param_group in enumerate(optimizer.param_groups):
                    if i < 3:
                        param_group['lr'] *= 0.5
                    elif len(param_group['params']) == 1 and param_group['params'][0].shape == model.prototypes.shape:
                        param_group['lr'] = 1e-3
        
        total_loss = 0
        num_batches = 0
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            text = text_data[batch_indices].to(device)
            image = image_data[batch_indices].to(device)
            audio = audio_data[batch_indices].to(device)
            labels = label_data[batch_indices].long().to(device)
            
            text_feat = model.text_proj_proto(text)
            image_feat = model.image_proj_proto(image)
            audio_feat = model.audio_proj_proto(audio)
            
            text_feat_norm = F.normalize(text_feat, dim=1)
            image_feat_norm = F.normalize(image_feat, dim=1)
            audio_feat_norm = F.normalize(audio_feat, dim=1)
            proto_norm = F.normalize(model.prototypes, dim=1)
            
            text_sim = torch.mm(text_feat_norm, proto_norm.t())
            image_sim = torch.mm(image_feat_norm, proto_norm.t())
            audio_sim = torch.mm(audio_feat_norm, proto_norm.t())
            
            cls_loss = 0
            for emo_idx in range(4):
                cls_loss += F.cross_entropy(text_sim / 0.05, labels[:, emo_idx])
                cls_loss += F.cross_entropy(image_sim / 0.10, labels[:, emo_idx])
                cls_loss += F.cross_entropy(audio_sim / 0.15, labels[:, emo_idx])
            
            pred_text = model.text_classifier(text_feat).view(-1, 4, 2)
            pred_image = model.image_classifier(image_feat).view(-1, 4, 2)
            pred_audio = model.audio_classifier(audio_feat).view(-1, 4, 2)
            
            ce_loss = 0
            for emo_idx in range(4):
                ce_loss += F.cross_entropy(pred_text[:, emo_idx, :], labels[:, emo_idx])
                ce_loss += F.cross_entropy(pred_image[:, emo_idx, :], labels[:, emo_idx])
                ce_loss += F.cross_entropy(pred_audio[:, emo_idx, :], labels[:, emo_idx])
            
            class_loss = compute_prototype_class_loss(model, labels, device)
            margin = 0.6 if stage == 1 else 0.4
            sep_loss = prototype_separation_loss(model.prototypes, margin=margin)
            
            loss = cls_weight * cls_loss + class_weight * class_loss + sep_weight * sep_loss + ce_weight * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        print(f"[Stage {stage}] Epoch {epoch+1}/{epochs}: Loss={total_loss/num_batches:.4f}")

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
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def set_decay(self, decay):
        self.decay = decay
        
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def add_adaptive_noise(x, base_noise_level, progress_factor):
    if base_noise_level <= 0:
        return x
    
    actual_noise = base_noise_level * (0.5 + 0.5 * progress_factor)
    
    std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
    noise = torch.randn_like(x) * std * actual_noise
    
    return x + noise
    
def train_epoch(model, train_data, batch_size, missing_rate, optimizer, device, epoch, total_epochs, ema=None):
    model.train()
    text_data, image_data, audio_data, label_data = train_data
    num_samples = len(label_data)
    indices = torch.randperm(num_samples)
    total_loss = 0
    num_batches = 0
    alpha = 0.5
    pa_weight = 0.0
    
    progress = epoch / total_epochs
    if epoch < 15:
        noise_text, noise_visual, noise_audio = 0.03, 0.05, 0.08
    elif epoch < 35:
        noise_text, noise_visual, noise_audio = 0.05, 0.08, 0.12
    else:
        noise_text, noise_visual, noise_audio = 0.07, 0.10, 0.15
    
    if epoch >= 20:
        mc_weight = max(0.0, 0.4 * (1 - (epoch - 20) / (total_epochs - 20)))
    else:
        mc_weight = 0.4
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        text = text_data[batch_indices].to(device)
        image = image_data[batch_indices].to(device)
        audio = audio_data[batch_indices].to(device)
        labels = label_data[batch_indices].long().to(device)
        
        text = add_adaptive_noise(text, noise_text, progress)
        image = add_adaptive_noise(image, noise_visual, progress)
        audio = add_adaptive_noise(audio, noise_audio, progress)
        # ================================
        
        mask = get_mask(3, len(batch_indices), missing_rate).to(device)
        
        optimizer.zero_grad()
        preds, L_unimodal, l_cls, l_mc, l_pa = model(text, image, audio, labels, mask, 
                                                     is_train=True, current_epoch=epoch)
        
        L_multimodal = l_cls + mc_weight * l_mc + pa_weight * l_pa
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
        num_batches += 1
        
    return total_loss/num_batches

def load_iemocap_data(pkl_path):
    import pickle
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    def process_split(split_data):
        text = pool(torch.from_numpy(split_data['text']).float())
        audio = pool(torch.from_numpy(split_data['audio']).float())
        vision = pool(torch.from_numpy(split_data['vision']).float())
        
        labels_onehot = split_data['labels']  # [N, 4, 2]
        labels = labels_onehot[:, :, 1].astype(np.int64)  # [N, 4]
        labels_tensor = torch.from_numpy(labels).long()
        
        return text, vision, audio, labels_tensor
    
    train_text, train_vision, train_audio, train_labels = process_split(data['train'])
    val_text, val_vision, val_audio, val_labels = process_split(data['valid'])
    test_text, test_vision, test_audio, test_labels = process_split(data['test'])
    
    print(f"\nIEMOCAP Data Loaded:")
    print(f"Train: {len(train_labels)} | Valid: {len(val_labels)} | Test: {len(test_labels)}")
    print(f"Dims: Text={train_text.shape[1]}, Vision={train_vision.shape[1]}, Audio={train_audio.shape[1]}\n")
    
    return (train_text, train_vision, train_audio, train_labels,
            val_text, val_vision, val_audio, val_labels,
            test_text, test_vision, test_audio, test_labels)

if __name__ == '__main__':
    seed_everything(1)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    pkl_path = '/home/shantao/data/iemocap_data.pkl'
    
    (feature_text_train, feature_visual_train, feature_audio_train, label_train,
     feature_text_val, feature_visual_val, feature_audio_val, label_val,
     feature_text_test, feature_visual_test, feature_audio_test, label_test) = load_iemocap_data(pkl_path)
    
    model = PrototypeEmotionModel(
        text_dim=300, image_dim=35, audio_dim=74,
        common_dim=256, num_classes=4, k_neighbors=15, drop_prob=0.2
    ).to(device)
    
    model.build_feature_bank(feature_text_train, feature_visual_train, feature_audio_train, device)
    
    train_data = (feature_text_train, feature_visual_train, feature_audio_train, label_train)
    print("\n=== Phase 1: Pretraining (30 epochs) ===")
    pretrain_prototypes_and_projections(model, train_data, batch_size=128, device=device, epochs=30)
    
    print("\n=== Phase 2: Training with frozen prototypes (50 epochs) ===")
    model.prototypes.requires_grad = False
    for param in model.prototype_class_predictor.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() 
         if 'prototypes' not in n and 'prototype_class_predictor' not in n], 
        lr=1.5e-4, weight_decay=1e-5
    )
    scheduler = get_warmup_cosine_scheduler(optimizer, 10, 150, min_lr=1e-6)
    
    ema = EMA(model, decay=0.999)
    ema.register()
    
    batch_size = 32
    missing_rate = 0.5
    best_avg_acc = 0
    best_avg_f1w = 0
    best_avg_acc_epoch = 0
    best_avg_f1w_epoch = 0
    patience = 20
    patience_counter = 0
    test_data = (feature_text_test, feature_visual_test, feature_audio_test, label_test)
    
    best_results_acc = {i: {'acc': 0, 'epoch': 0, 'f1_w': 0, 'f1_m': 0} for i in range(7)}
    best_results_f1w = {i: {'f1_w': 0, 'epoch': 0, 'acc': 0, 'f1_m': 0} for i in range(7)}
    
    pattern_names = {
        0: "Audio Only (A)", 1: "Text Only (T)", 2: "Visual Only (V)",
        3: "Audio+Visual (AV)", 4: "Audio+Text (AT)", 5: "Text+Visual (TV)",
        6: "Complete (TAV)"
    }
    
    for epoch in range(150):
        ema_decay = 0.995 + 0.004 * (epoch / 150)
        ema.set_decay(ema_decay)
        
        happy_weight = 1.0 if epoch < 10 else 1.5
        sad_weight = 1.0
        
        train_loss = train_epoch(model, train_data, batch_size, missing_rate, optimizer, device, epoch, 150, ema=ema)
        scheduler.step()
    
        print(f"\n{'='*90}")
        print(f"Epoch {epoch+1}/100: Loss={train_loss:.4f} | Happy Weight={happy_weight} | Sad Weight={sad_weight}")
        
        ema.apply_shadow()
        test_results = evaluate_all_missing_patterns(model, test_data, batch_size, device)
        ema.restore()
        
        current_accs = []
        current_f1ws = []
        
        for pattern_id in range(7):
            result = test_results[pattern_id]
            current_accs.append(result['acc'])
            current_f1ws.append(result['f1_weighted'])
            
            if result['acc'] > best_results_acc[pattern_id]['acc']:
                best_results_acc[pattern_id] = {
                    'acc': result['acc'], 
                    'epoch': epoch + 1,
                    'f1_w': result['f1_weighted'], 
                    'f1_m': result['f1_macro']
                }
                
                ema.apply_shadow()
                torch.save(model.state_dict(), f"best_acc_pattern_{pattern_id}.pth")
                ema.restore()
            
            if result['f1_weighted'] > best_results_f1w[pattern_id]['f1_w']:
                best_results_f1w[pattern_id] = {
                    'f1_w': result['f1_weighted'],
                    'epoch': epoch + 1,
                    'acc': result['acc'],
                    'f1_m': result['f1_macro']
                }
                
                ema.apply_shadow()
                torch.save(model.state_dict(), f"best_f1w_pattern_{pattern_id}.pth")
                ema.restore()
            
            acc_best_marker = "[BEST-ACC]" if result['acc'] == best_results_acc[pattern_id]['acc'] else ""
            f1w_best_marker = "[BEST-F1W]" if result['f1_weighted'] == best_results_f1w[pattern_id]['f1_w'] else ""
            
            print(f"{result['name']:20s} | Cur: {result['acc']*100:.2f}%/{result['f1_weighted']*100:.2f}% | "
                  f"Best-ACC: {best_results_acc[pattern_id]['acc']*100:.2f}%(E{best_results_acc[pattern_id]['epoch']}) | "
                  f"Best-F1W: {best_results_f1w[pattern_id]['f1_w']*100:.2f}%(E{best_results_f1w[pattern_id]['epoch']}) "
                  f"{acc_best_marker}{f1w_best_marker}")
        
        current_avg_acc = np.mean(current_accs)
        current_avg_f1w = np.mean(current_f1ws)
        print(f"\n{'Current Avg ACC':20s}: {current_avg_acc*100:.2f}%")
        print(f"{'Current Avg F1-W':20s}: {current_avg_f1w*100:.2f}%")
        
        if current_avg_acc > best_avg_acc:
            best_avg_acc = current_avg_acc
            best_avg_acc_epoch = epoch + 1
            patience_counter = 0
            ema.apply_shadow()
            torch.save(model.state_dict(), "best_avg_acc_model.pth")
            ema.restore()
            print(f"✓ Saved best avg ACC model (Avg Acc: {best_avg_acc*100:.2f}%)")
        else:
            patience_counter += 1
        
        if current_avg_f1w > best_avg_f1w:
            best_avg_f1w = current_avg_f1w
            best_avg_f1w_epoch = epoch + 1
            ema.apply_shadow()
            torch.save(model.state_dict(), "best_avg_f1w_model.pth")
            ema.restore()
            print(f"✓ Saved best avg F1-W model (Avg F1-W: {best_avg_f1w*100:.2f}%)")
        
        print(f"\n--- Best Results So Far (by ACC) ---")
        for pattern_id in range(7):
            best = best_results_acc[pattern_id]
            print(f"{pattern_names[pattern_id]:20s} | ACC: {best['acc']*100:.2f}% (Epoch {best['epoch']})")
        
        print(f"\n--- Best Results So Far (by F1-W) ---")
        for pattern_id in range(7):
            best = best_results_f1w[pattern_id]
            print(f"{pattern_names[pattern_id]:20s} | F1-W: {best['f1_w']*100:.2f}% (Epoch {best['epoch']})")
        
        overall_best_avg_acc = np.mean([best_results_acc[i]['acc'] for i in range(7)])
        overall_best_avg_f1w = np.mean([best_results_f1w[i]['f1_w'] for i in range(7)])
        
        print(f"\n{'Overall Best Avg ACC':20s}: {overall_best_avg_acc*100:.2f}%")
        print(f"{'Overall Best Avg F1-W':20s}: {overall_best_avg_f1w*100:.2f}%")
        print(f"{'Best Avg ACC (ES)':20s}: {best_avg_acc*100:.2f}% (Epoch {best_avg_acc_epoch})")
        print(f"{'Best Avg F1-W (ES)':20s}: {best_avg_f1w*100:.2f}% (Epoch {best_avg_f1w_epoch})")
        print(f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("\n" + "="*90)
    print("Training Complete!")
    print("="*90)
    
    print("\n" + "="*90)
    print("FINAL SUMMARY - BEST ACCURACY")
    print("="*90)
    for pattern_id in range(7):
        best = best_results_acc[pattern_id]
        print(f"\n{pattern_names[pattern_id]:20s} (Best ACC at Epoch {best['epoch']})")
        print(f"  ACC: {best['acc']*100:.2f}% | F1-W: {best['f1_w']*100:.2f}% | F1-M: {best['f1_m']*100:.2f}%")
    
    overall_best_avg_acc = np.mean([best_results_acc[i]['acc'] for i in range(7)])
    print(f"\n{'Overall Best Avg ACC':20s}: {overall_best_avg_acc*100:.2f}%")
    
    print("\n" + "="*90)
    print("FINAL SUMMARY - BEST F1-WEIGHTED")
    print("="*90)
    for pattern_id in range(7):
        best = best_results_f1w[pattern_id]
        print(f"\n{pattern_names[pattern_id]:20s} (Best F1-W at Epoch {best['epoch']})")
        print(f"  F1-W: {best['f1_w']*100:.2f}% | ACC: {best['acc']*100:.2f}% | F1-M: {best['f1_m']*100:.2f}%")
    
    overall_best_avg_f1w = np.mean([best_results_f1w[i]['f1_w'] for i in range(7)])
    print(f"\n{'Overall Best Avg F1-W':20s}: {overall_best_avg_f1w*100:.2f}%")
    
    print("\n" + "="*90)
    print("EARLY STOPPING RESULTS")
    print("="*90)
    print(f"Best Avg ACC: {best_avg_acc*100:.2f}% (Epoch {best_avg_acc_epoch})")
    print(f"Best Avg F1-W: {best_avg_f1w*100:.2f}% (Epoch {best_avg_f1w_epoch})")
    
    print("\n" + "="*90)
    print("SAVED MODELS")
    print("="*90)
    print(f"\n[Overall Best Models]")
    print(f"  best_avg_acc_model.pth  - Avg ACC: {best_avg_acc*100:.2f}% (Epoch {best_avg_acc_epoch})")
    print(f"  best_avg_f1w_model.pth  - Avg F1-W: {best_avg_f1w*100:.2f}% (Epoch {best_avg_f1w_epoch})")
    
    print(f"\n[Pattern-Specific Best ACC Models]")
    for pattern_id in range(7):
        best = best_results_acc[pattern_id]
        print(f"  best_acc_pattern_{pattern_id}.pth - {pattern_names[pattern_id]:20s}: {best['acc']*100:.2f}% (Epoch {best['epoch']})")
   
    print("="*90)