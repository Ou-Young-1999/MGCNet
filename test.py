import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from data.dataset import AFDataset
from data.ecg_augment import ECGAugment
import json

def load_config(config_path="config/resnet18_1d_afdb.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_model(config):
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    in_channels = config['model']['in_channels']

    if model_name == 'resnet18_1d':
        model = resnet18_1d(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'resnet34_1d':
        model = resnet34_1d(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'sccnn':
        from model.sccnn import SCCNN
        model = SCCNN(num_classes=num_classes)
    elif model_name == 'imcresnet':
        from model.imcresnet import IMCResNet
        model = IMCResNet(num_classes=num_classes)
    elif model_name == 'moetrans':
        from model.moetrans import MoEAF
        model = MoEAF()
    elif model_name == 'seqafnet':
        from model.seqafnet import SeqAFNet
        model = SeqAFNet()
    elif model_name == 'mfegnet':
        from model.mfegnet import MFEGNet
        model = MFEGNet()
    elif model_name == 'mscgn':
        from model.mscgn import MSCGN
        model = MSCGN()
    elif model_name == 'bgm':
        from model.bgm import GMCNet
        model = GMCNet()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def get_dataloader(config, is_train=False):
    """创建测试集 DataLoader"""
    if is_train:
        data_cfg = config['dataset']['train']  # 使用 test 分割
        aug_cfg = config['augmentation']['train']
        split = data_cfg['split']
    else:
        data_cfg = config['dataset']['test']
        aug_cfg = config['augmentation']['test']
        split = data_cfg['split']

    transform = ECGAugment(**aug_cfg)

    dataset = AFDataset(
        path1=data_cfg['split_dir'],
        path2=data_cfg['data_dir'],
        split=split,
        fold=data_cfg['fold'],
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        drop_last=False
    )
    return loader, dataset

def load_model_weights(model, checkpoint_path, device):
    """加载模型权重"""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model weights loaded from {checkpoint_path}")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="ECG Classification Training")
    parser.add_argument(
        '--config',
        type=str,
        default='config/resnet18_1d_afdb.yaml',
        help='Path to the config file (YAML)'
    )
    return parser.parse_args()

def main():
    # -------------------------------
    # 1. 加载配置
    # -------------------------------
    args = parse_args()
    # 使用命令行传入的 config 路径，如果没传则用默认值
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # -------------------------------
    # 2. 设置设备
    # -------------------------------
    device_config = config['training']['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_config == 'auto' else torch.device(device_config)
    print(f'Using device: {device}')

    # -------------------------------
    # 3. 加载测试集
    # -------------------------------
    test_loader, test_dataset = get_dataloader(config, is_train=False)
    print(f"Test set size: {len(test_dataset)}")

    # -------------------------------
    # 4. 创建模型并加载权重
    # -------------------------------
    model = get_model(config)
    model.to(device)

    # 模型权重路径（根据你的保存逻辑）
    checkpoint_dir = config['logging']['checkpoint_dir']
    title = config['title']
    checkpoint_path = os.path.join(checkpoint_dir, f"{title}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model_weights(model, checkpoint_path, device)

    # -------------------------------
    # 5. 推理：收集真实标签、预测概率、预测标签
    # -------------------------------
    all_labels = []
    all_probs = []
    all_preds = []

    all_t = []
    all_f = []

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as t:
            for ecgs,imgs, labels in t:
                ecgs,imgs, labels = ecgs.to(device), imgs.to(device), labels.to(device)

                outputs,t,f = model(ecgs,imgs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # 预测概率（softmax）
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
                all_preds.extend(preds)

                all_t.extend(t.cpu().numpy())
                all_f.extend(f.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    # 转为 numpy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    t_arr = np.array(all_t)
    f_arr = np.array(all_f)

    # -------------------------------
    # 6. 计算各项指标
    # -------------------------------
    num_classes = config['model']['num_classes']

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary')
    precision = precision_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary')
    recall = recall_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary')  # 敏感度

    # AUC（多分类用 OvR）
    if num_classes == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')

    # 特异性（Specificity）：对于每个类，计算 TN / (TN + FP)
    specificity_list = []
    for cls in range(num_classes):
        tn = np.sum((y_true != cls) & (y_pred != cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)
    avg_specificity = np.mean(specificity_list)

    # -------------------------------
    # 7. 打印结果
    # -------------------------------
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity (avg): {avg_specificity:.4f}")
    print("-"*50)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("="*50)

    # -------------------------------
    # 8. 保存结果
    # -------------------------------
    result_dir = os.path.join(config['logging']['result_dir'], config['title'])
    os.makedirs(result_dir, exist_ok=True)

    # 8.1 保存预测概率和标签
    results_df = pd.DataFrame({
        'label': y_true,
        'prediction': y_pred,
    })
    # 添加每一类的概率
    for i in range(y_prob.shape[1]):
        results_df[f'prob_class_{i}'] = y_prob[:, i]

    results_df.to_csv(os.path.join(result_dir, 'test_predictions.csv'), index=False)
    print(f"Predictions saved to {os.path.join(result_dir, 'test_predictions.csv')}")

    # 8.2 保存指标
    metrics = {
        'loss': float(avg_loss),
        'accuracy': float(accuracy),
        'auc': float(auc),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(recall),  # 同 recall
        'specificity_avg': float(avg_specificity),
        'specificity_per_class': [float(s) for s in specificity_list]
    }

    with open(os.path.join(result_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # 8.3 保存 classification report
    with open(os.path.join(result_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Results\n")
        f.write(f"Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Specificity (avg): {avg_specificity:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, digits=4))

    print(f"Metrics and report saved to {result_dir}")

    np.savez_compressed(
        os.path.join(result_dir, 'predictions.npz'),
        labels=y_true,
        t=t_arr,
        f=f_arr
    )

if __name__ == '__main__':
    main()