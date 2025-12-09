import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import time

from data.dataset import AFDataset
from data.ecg_augment import ECGAugment


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path="config.yaml"):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_model(config):
    """根据配置创建模型"""
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


def get_optimizer(model, config):
    """根据配置创建优化器"""
    optim_type = config['optimizer']['type']
    params = config['optimizer']['params']

    if optim_type == 'Adam':
        return optim.Adam(model.parameters(), **params)
    elif optim_type == 'SGD':
        return optim.SGD(model.parameters(), **params)
    elif optim_type == 'AdamW':
        return optim.AdamW(model.parameters(), **params)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_type}")


def get_scheduler(optimizer, config):
    """根据配置创建学习率调度器"""
    sched_type = config['scheduler']['type']
    params = config['scheduler']['params']

    if sched_type == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, **params)
    elif sched_type == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    elif sched_type == 'CosineAnnealingLR':
        T_max = params.get('T_max', 25)  # 默认 0.0001
        eta_min = params.get('eta_min', 0.000001)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
    elif sched_type == 'InverseTimeLR':
        # 实现: lr = initial_lr / (1 + decay_rate * epoch)
        # 注意：initial_lr 来自 optimizer，decay_rate 从 params 传入
        decay_rate = params.get('decay_rate', 0.0001)  # 默认 0.0001
        last_epoch = params.get('last_epoch', -1)

        def lr_lambda(epoch):
            return 1.0 / (1 + decay_rate * epoch)  # 缩放因子

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    else:
        raise ValueError(f"Unsupported scheduler: {sched_type}")


def get_dataloader(config, is_train=True):
    """创建 DataLoader"""
    if is_train:
        data_cfg = config['dataset']['train']
        aug_cfg = config['augmentation']['train']
        split = data_cfg['split']
    else:
        data_cfg = config['dataset']['val']
        aug_cfg = config['augmentation']['val']
        split = data_cfg['split']

    # 数据增强
    transform = ECGAugment(**aug_cfg)

    # 数据集
    dataset = AFDataset(
        path1=data_cfg['split_dir'],
        path2=data_cfg['data_dir'],
        split=split,
        fold=data_cfg['fold'],
        transform=transform
    )

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=is_train,
        drop_last=is_train
    )
    return loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, config):
    """训练主循环"""
    best_accuracy = 0.0
    num_epochs = config['training']['num_epochs']
    save_dir = config['logging']['checkpoint_dir']
    os.makedirs(save_dir, exist_ok=True)

    patience = config['training'].get('patience', 10)  # 等待多少个 epoch 没提升就停止
    early_stop_counter = 0
    early_stopping = False

    for epoch in range(num_epochs):

        if early_stopping:
            break

        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t:
            for ecgs,imgs, labels in t:
                ecgs,imgs, labels = ecgs.to(device), imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs,total_scl,scl_losses = model(ecgs,imgs,labels,True,True,True)
                loss = criterion(outputs, labels) + 0.01*total_scl
                # loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

                accuracy = 100 * correct_preds / total_preds
                t.set_postfix(loss=running_loss / (t.n + 1), accuracy=accuracy)

        # 训练指标
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_preds / total_preds

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for ecgs,imgs, labels in val_loader:
                ecgs,imgs, labels = ecgs.to(device), imgs.to(device), labels.to(device)
                outputs,total_scl,scl_losses = model(ecgs,imgs,labels,True,True,True)
                loss = criterion(outputs, labels) + 0.01*total_scl
                # loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        # 更新学习率
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # 打印日志
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%')
        tqdm.write(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # TensorBoard 记录
        writer.add_scalar('Loss/train', epoch_loss, epoch + 1)
        writer.add_scalar('Accuracy/train', epoch_accuracy, epoch + 1)
        writer.add_scalar('Loss/valid', val_loss, epoch + 1)
        writer.add_scalar('Accuracy/valid', val_accuracy, epoch + 1)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch + 1)

        # 保存最优模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_path = os.path.join(save_dir, f"{config['title']}.pth")
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"Best model saved: {save_path}, Accuracy: {best_accuracy:.2f}%")
            early_stop_counter = 0  # 重置计数器
        else:
            early_stop_counter += 1
            tqdm.write(f"Early stopping counter: {early_stop_counter}/{patience}")

            # if early_stop_counter >= patience and epoch+1 >= 20:
            if early_stop_counter >= patience:
                tqdm.write(f"Early stopping triggered after {epoch + 1} epochs.")
                early_stopping = True

    print(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")

# 创建损失函数前先计算类别权重
def get_class_weights(loader, device):
    """遍历 DataLoader 统计每个类别的样本数，并返回类别权重"""
    class_counts = {}
    total_samples = 0

    # 遍历一次数据集统计标签分布
    for _, labels in loader:
        for label in labels.numpy():
            class_counts[label] = class_counts.get(label, 0) + 1
            total_samples += 1

    num_classes = len(class_counts)
    weights = []
    for cls in range(num_classes):
        count = class_counts.get(cls, 1)  # 防止缺失类
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float).to(device)

def parse_args():
    parser = argparse.ArgumentParser(description="ECG Classification Training")
    parser.add_argument(
        '--config',
        type=str,
        default='config/resnet18_1d_afdb.yaml',
        help='Path to the config file (YAML)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # 使用命令行传入的 config 路径，如果没传则用默认值
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # 设置随机种子
    seed = config['training']['seed']
    set_seed(seed)
    print(f'Random seed set to {seed}')

    # 设置设备
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    print(f'Using device: {device}')

    # 创建数据加载器
    train_loader = get_dataloader(config, is_train=True)
    val_loader = get_dataloader(config, is_train=False)

    print(f'Train set size: {len(train_loader.dataset)}')
    print(f'Validation set size: {len(val_loader.dataset)}')

    # 创建模型
    model = get_model(config)
    model = model.to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # class_weights = get_class_weights(train_loader, device)
    # print(f'Class weights: {class_weights}')
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 优化器和调度器
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # TensorBoard
    log_dir = os.path.join(config['logging']['tensorboard_dir'], config['title'])
    writer = SummaryWriter(log_dir)

    # 开始训练
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, config)


    writer.close()
