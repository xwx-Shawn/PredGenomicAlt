
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, average_precision_score
from public_config import load_data_config
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


CONFIG = {'seed': 42,
 'gpu_id': 0,
 'batch_size': 1,
 'accum_steps': 32,
 'lr': 1e-05,
 'weight_decay': 0.01,
 'max_epochs': 150,
 'patience': 15,
 'feature_dim': 1536,
 'n_classes': 1,
 'use_amp': True,
 'num_workers': 16,
 'target_gene': 'PTEN_mutdel',
 'log_dir': './results/pten_mutdel/pooled'}
CONFIG.update(load_data_config(CONFIG['target_gene']))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_metrics(y_true, y_prob, threshold):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}
    try:
        metrics['AUC'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['AUC'] = 0.5
    try:
        metrics['PRAUC'] = average_precision_score(y_true, y_prob)
    except:
        metrics['PRAUC'] = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['Acc'] = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    metrics['Sens'] = tp / (tp + fn + 1e-8)
    metrics['Spec'] = tn / (tn + fp + 1e-8)
    metrics['PPV'] = tp / (tp + fp + 1e-8)
    metrics['NPV'] = tn / (tn + fn + 1e-8)
    metrics['F1'] = 2 * tp / (2 * tp + fp + fn + 1e-8)
    metrics['Threshold'] = threshold
    return metrics


def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr - fpr
    idx = np.argmax(J)
    return np.clip(thresholds[idx], 0.01, 0.99)


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256):
        super().__init__()
        self.att_a = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.att_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.att_c = nn.Linear(D, 1)

    def forward(self, x):
        A = self.att_c(self.att_a(x) * self.att_b(x))
        return A, x


class AttentionNet(nn.Module):
    def __init__(self, input_feature_size=1024):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_feature_size, 256), nn.ReLU(), nn.Dropout(0.5)
        )
        self.attn = Attn_Net_Gated(L=256, D=128)
        self.classifier = nn.Linear(256, 1)

    def forward(self, h):
        h = self.fc(h)
        A_raw, h = self.attn(h)
        A = F.softmax(A_raw.T, dim=1)
        M = torch.mm(A, h)
        logit = self.classifier(M)
        return logit.squeeze(0), A


class FeatureBagsDataset(Dataset):
    def __init__(self, df, base_h5_dir, is_train=False):
        self.df = df.reset_index(drop=True)
        self.base_h5_dir = base_h5_dir
        self.is_train = is_train
        self.target_gene = CONFIG['target_gene']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]


        gene_status = row[self.target_gene]
        label = 1.0 if gene_status == 'MT' else 0.0

        sample_name = str(row['IMAGE_ID_clean'])
        fname = sample_name if sample_name.endswith('.pt') else sample_name + '.pt'
        pt_path = os.path.join(self.base_h5_dir, str(row['image_dir']), 'tiles_feature', fname)

        try:
            data = torch.load(pt_path, map_location='cpu')
            feats = data['tile_embeds'].float()
            if self.is_train:
                feats = feats + torch.randn_like(feats) * 0.01
            return feats, label
        except Exception as e:
            return torch.zeros((1, CONFIG['feature_dim'])), label


def collate_bag(batch):
    feats, label = batch[0]
    return feats, torch.tensor([label]).float()


def load_global_data():
    print(f">>> Loading predefined split for {CONFIG['target_gene']}...")

    global_df = pd.read_csv(CONFIG['global_split_file'])


    original_len = len(global_df)
    global_df = global_df[global_df[CONFIG['target_gene']] != 'Not available']
    print(f"  -> Dropped {original_len - len(global_df)} slices with missing {CONFIG['target_gene']} labels.")


    img_dir_map = {}
    for _, file_path in CONFIG['cohort_files'].items():
        if not os.path.exists(file_path): continue
        temp_df = pd.read_excel(file_path)
        id_col = 'GIGA_feats_name' if 'GIGA_feats_name' in temp_df.columns else (
            'IMAGE_ID' if 'IMAGE_ID' in temp_df.columns else 'Sample_ID')
        if 'image_dir' in temp_df.columns:
            for _, row in temp_df.iterrows():
                img_id = str(row[id_col]).replace('.pt', '').strip()
                img_dir_map[img_id] = row['image_dir']


    global_df['image_dir'] = global_df['IMAGE_ID_clean'].map(img_dir_map)

    global_df = global_df.dropna(subset=['image_dir'])

    return global_df.reset_index(drop=True)


def main():
    set_seed(CONFIG['seed'])
    device = torch.device(f"cuda:{CONFIG['gpu_id']}" if torch.cuda.is_available() else "cpu")


    df = load_global_data()

    print(f"\nStarting pooled training with predefined splits...")

    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    if os.path.exists(os.path.join(CONFIG['log_dir'], "best_model.pt")):
        print("Model already trained. Exiting.")
        return


    train_df = df[df['Dataset_Split'] == 'Training'].reset_index(drop=True)
    val_df = df[df['Dataset_Split'] == 'Validation'].reset_index(drop=True)

    print(f"Training Slices: {len(train_df)} | Validation Slices: {len(val_df)}")

    train_ds = FeatureBagsDataset(train_df, CONFIG['h5_base_dir'], is_train=True)
    val_ds = FeatureBagsDataset(val_df, CONFIG['h5_base_dir'], is_train=False)


    y_train = (train_df[CONFIG['target_gene']] == 'MT').astype(int).values
    count_pos = (y_train == 1).sum()
    count_neg = (y_train == 0).sum()
    w_pos = 0.5 / count_pos if count_pos > 0 else 0
    w_neg = 0.5 / count_neg if count_neg > 0 else 0
    weights = torch.DoubleTensor([w_pos if t == 1 else w_neg for t in y_train])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                              num_workers=CONFIG['num_workers'], collate_fn=collate_bag)
    val_loader = DataLoader(val_ds, batch_size=1,
                            num_workers=CONFIG['num_workers'], collate_fn=collate_bag)

    model = AttentionNet(CONFIG['feature_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=CONFIG['use_amp'])

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=5),
            CosineAnnealingLR(optimizer, T_max=CONFIG['max_epochs'] - 5, eta_min=1e-6)
        ],
        milestones=[5]
    )

    log_f = open(os.path.join(CONFIG['log_dir'], "training_log.txt"), "w")
    log_f.write("Epoch\tSet\tLoss\tAUC\tAcc\tF1\tThresh\n")

    best_auc = 0
    patience_counter = 0

    for epoch in range(CONFIG['max_epochs']):

        model.train()
        train_loss = 0.0
        y_true, y_prob = [], []
        optimizer.zero_grad()

        for i, (feats, label) in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1} Train")):
            feats, label = feats.to(device), label.to(device)
            with autocast(enabled=CONFIG['use_amp']):
                logit, _ = model(feats)
                loss_val = criterion(logit, label)
                loss = loss_val / CONFIG['accum_steps']

            scaler.scale(loss).backward()
            train_loss += loss_val.item()

            if (i + 1) % CONFIG['accum_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            y_true.append(label.item())
            y_prob.append(torch.sigmoid(logit).item())

        scheduler.step()
        train_loss /= len(train_loader)
        train_thresh = find_optimal_threshold(y_true, y_prob)
        train_metrics = calculate_metrics(y_true, y_prob, train_thresh)


        model.eval()
        val_loss = 0.0
        y_true_v, y_prob_v = [], []

        with torch.no_grad():
            for feats, label in val_loader:
                feats, label = feats.to(device), label.to(device)
                logit, _ = model(feats)
                loss_v = criterion(logit, label)
                val_loss += loss_v.item()

                y_true_v.append(label.item())
                y_prob_v.append(torch.sigmoid(logit).item())

        val_loss /= len(val_loader)
        val_thresh = find_optimal_threshold(y_true_v, y_prob_v)
        val_metrics = calculate_metrics(y_true_v, y_prob_v, val_thresh)

        print(
            f"Ep {epoch + 1:03d} | Train Loss:{train_loss:.3f} AUC:{train_metrics['AUC']:.3f} | Val Loss:{val_loss:.3f} AUC:{val_metrics['AUC']:.3f}")
        log_f.write(
            f"{epoch + 1}\tTrain\t{train_loss:.4f}\t{train_metrics['AUC']:.4f}\t{train_metrics['Acc']:.4f}\t{train_metrics['F1']:.4f}\t{train_thresh:.3f}\n")
        log_f.write(
            f"{epoch + 1}\tVal\t{val_loss:.4f}\t{val_metrics['AUC']:.4f}\t{val_metrics['Acc']:.4f}\t{val_metrics['F1']:.4f}\t{val_thresh:.3f}\n")
        log_f.flush()

        if val_metrics['AUC'] > best_auc:
            best_auc = val_metrics['AUC']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['log_dir'], "best_model.pt"))

            print(f"  >>> Best AUC Saved: {best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print("Early Stopping.")
                break

    log_f.close()
    print("\nPooled training completed.")


if __name__ == "__main__":
    main()
