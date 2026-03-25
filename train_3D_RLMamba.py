import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from scipy.io import loadmat, savemat
from torch import optim
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os
from utils import *
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from model._3D_RLMamba import _3D_RLMamba
# CUDA_VISIBLE_DEVICES=0 python train_3D_RLMamba.py --dataset='SAR' --epoches=1 --patches=11 --sess MambaHSI_promote --dropout 0.4 --lr 1e-3
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'WHU_Hi_LongKou', 'SAR'], default='Indian', help='dataset to use')
parser.add_argument('--flag', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--sess', default='mamba')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--test_freq', type=int, default=5, help='evaluation frequency')
parser.add_argument('--patches', type=int, default=11, help='patch size')
parser.add_argument('--epoches', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
args = parser.parse_args()

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

setup_seed(args)
input_normalize, label, num_classes, TR, TE, color_matrix, color_matrix_pred = load_HSI(args)
num_classes = int(num_classes)
height, width, band = input_normalize.shape

total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)

train_pos = np.argwhere(TR > 0)
train_labels = label[train_pos[:, 0], train_pos[:, 1]]

train_pos_new, val_pos, train_label_new, val_label = train_test_split(
    train_pos, train_labels, test_size=0.25, random_state=args.seed, stratify=train_labels
)

TR_new = np.zeros_like(TR)
VAL = np.zeros_like(TR)
TR_new[train_pos_new[:, 0], train_pos_new[:, 1]] = train_label_new
VAL[val_pos[:, 0], val_pos[:, 1]] = val_label

print("="*60)
print(f"Final Split: Train:Val:Test = 6:2:2")
print(f"Train samples: {len(train_pos_new)}")
print(f"Val samples:   {len(val_pos)}")
print(f"Test samples:  {len(total_pos_test)}")
print("="*60)

class PatchSet(Dataset):
    def __init__(self, data, gt, patch_size, num_classes, is_pred=False):
        self.is_pred = is_pred
        self.patch_size = patch_size
        self.num_classes = num_classes
        p = patch_size // 2
        self.data = np.pad(data, ((p, p), (p, p), (0, 0)), 'constant', constant_values=0)

        if is_pred:
            gt_proc = np.ones_like(gt)
        else:
            gt_proc = np.where(gt == 0, -1, gt - 1)

        self.label = np.pad(gt_proc, ((p, p), (p, p)), 'constant', constant_values=-1)
        x_pos, y_pos = np.nonzero(self.label >= 0)
        self.indices = np.array(list(zip(x_pos, y_pos)))

        if not is_pred:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        patch = self.data[x1:x2, y1:y2]
        patch = patch.astype('float32').transpose((2, 0, 1))
        patch = np.expand_dims(patch, axis=1)
        patch = torch.from_numpy(patch)

        if self.is_pred:
            return patch
        else:
            label = int(self.label[x, y])
            assert 0 <= label < self.num_classes
            label = torch.tensor(label, dtype=torch.long)
            return patch, label

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    correct = total = 0
    total_loss = 0.0
    tar_all, pre_all = [], []

    loop = tqdm(dataloader, desc='Train', leave=False)
    for inputs, targets in loop:
        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        total_loss += loss.item() * targets.size(0)

        tar_all.extend(targets.cpu().numpy())
        pre_all.extend(preds.cpu().numpy())
        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    acc = 100. * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss, tar_all, pre_all

def valid_epoch(model, dataloader, criterion):
    model.eval()
    tar_all, pre_all = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            _, preds = outputs.max(1)
            tar_all.extend(targets.cpu().numpy())
            pre_all.extend(preds.cpu().numpy())

    return tar_all, pre_all

train_dataset = PatchSet(input_normalize, TR_new, args.patches, num_classes)
val_dataset = PatchSet(input_normalize, VAL, args.patches, num_classes)
test_dataset = PatchSet(input_normalize, TE, args.patches, num_classes)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model = _3D_RLMamba(num_classes=num_classes, hidden_dim=128)

model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

class_name = ['road', 'sugarcane', 'rice_normal', 'rice_loding', 'weed',
              'tree', 'building', 'water', 'rice_booting']

if args.flag == 'test':
    model_path = os.path.join("checkpoints", args.dataset, 'best.pt')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    tar_v, pre_v = valid_epoch(model, test_loader, criterion)
    OA, AA_mean, Kappa, _ = output_metric(tar_v, pre_v)
    print("Test Result: OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA_mean, Kappa))
    print(classification_report(tar_v, pre_v, target_names=class_name, digits=4))

elif args.flag == 'train':
    save_dir = os.path.join("checkpoints", args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    best_OA = 0.0

    for epoch in range(args.epoches):
        t0 = time.time()
        train_acc, train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        train_time = time.time() - t0

        val_OA = None
        if epoch % args.test_freq == 0 or epoch == args.epoches - 1:
            tar_v, pre_v = valid_epoch(model, val_loader, criterion)
            val_OA, AA_mean, Kappa, _ = output_metric(tar_v, pre_v)
            print(f"\n[Val] Epoch {epoch+1} OA: {val_OA:.4f} | AA: {AA_mean:.4f} | Kappa: {Kappa:.4f}")
            print(classification_report(tar_v, pre_v, target_names=class_name, digits=4))

            if val_OA > best_OA:
                best_OA = val_OA
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))

        print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.6f} | Acc: {train_acc:.2f}% | Time: {train_time:.2f}s")

    torch.save(model.state_dict(), os.path.join(save_dir, "last.pt"))
    print(f"\nTraining Finished | Best Val OA: {best_OA:.4f}")

    print("\n" + "="*50)
    print("Final Test Result")
    print("="*50)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pt")))
    tar_test, pre_test = valid_epoch(model, test_loader, criterion)
    test_OA, test_AA, test_Kappa, _ = output_metric(tar_test, pre_test)
    print(f"OA: {test_OA:.4f} | AA: {test_AA:.4f} | Kappa: {test_Kappa:.4f}")
    print(classification_report(tar_test, pre_test, target_names=class_name, digits=4))