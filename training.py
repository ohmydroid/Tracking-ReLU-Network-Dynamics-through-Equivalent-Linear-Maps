import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
import pickle
import glob
from collections import defaultdict
import time
import gc

warnings.filterwarnings('ignore')

# 自动选择设备，优先使用CPU以节省GPU内存
device = torch.device('cpu')  # 强制使用CPU，避免GPU内存问题

def save_data_compatibly(data, filepath):
    """使用兼容格式保存数据"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=2)
        return True
    except Exception as e:
        print(f"保存文件 {filepath} 失败: {e}")
        return False

# 简化版网络，减少参数数量
class LightweightDualModeNetwork(nn.Module):
    def __init__(self, img_input_dim=784, num_classes=10, hidden1=128, hidden2=64, 
                 embed_dim=32, mode='classifier'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.mode = mode
        
        # 大幅减少网络大小
        self.img_fc1 = nn.Linear(img_input_dim, hidden1, bias=False)
        self.img_fc2 = nn.Linear(hidden1, hidden2, bias=False)
        self.img_fc3 = nn.Linear(hidden2, embed_dim, bias=False)
        self.relu = nn.ReLU()
        
        if mode == 'dual_stream':
            self.label_embed = nn.Embedding(num_classes, embed_dim)
        else:
            self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, img, label_indices=None):
        x = self.relu(self.img_fc1(img))
        x = self.relu(self.img_fc2(x))
        img_embed = self.img_fc3(x)
        
        if self.mode == 'dual_stream':
            label_embed = self.label_embed(label_indices)
            return img_embed, label_embed
        else:
            output = self.classifier(img_embed)
            return output
    
    def forward_with_masks(self, img):
        """前向传播并返回激活mask"""
        # 第一层
        x1 = self.img_fc1(img)
        mask1 = (x1 > 0).float()  # 在ReLU之前获取mask
        x1 = self.relu(x1)
        
        # 第二层  
        x2 = self.img_fc2(x1)
        mask2 = (x2 > 0).float()  # 在ReLU之前获取mask
        x2 = self.relu(x2)
        
        # 第三层（无激活函数）
        img_embed = self.img_fc3(x2)
        
        return img_embed, mask1, mask2
    
    def compute_effective_weight(self, mask1, mask2):
        """
        正确计算等效权重矩阵
        W_eq = W3 @ diag(mask2) @ W2 @ diag(mask1) @ W1
        使用逐样本计算确保正确性
        """
        W1 = self.img_fc1.weight  # [h1, input_dim]
        W2 = self.img_fc2.weight  # [h2, h1]  
        W3 = self.img_fc3.weight  # [embed_dim, h2]
        
        batch_size = mask1.size(0)
        input_dim = W1.size(1)
        embed_dim = W3.size(0)
        
        W_eff_batch = torch.zeros(batch_size, embed_dim, input_dim, device=mask1.device)
        
        for i in range(batch_size):
            # 为每个样本构建对角矩阵
            D1 = torch.diag(mask1[i])  # [h1, h1]
            D2 = torch.diag(mask2[i])  # [h2, h2]
            
            # 按照公式计算: W3 @ D2 @ W2 @ D1 @ W1
            W_eff = W3 @ D2 @ W2 @ D1 @ W1
            W_eff_batch[i] = W_eff
        
        return W_eff_batch
    
    def compute_effective_weight_fast(self, mask1, mask2):
        """
        快速批量计算等效权重（使用einsum）
        """
        W1 = self.img_fc1.weight  # [h1, input_dim]
        W2 = self.img_fc2.weight  # [h2, h1]  
        W3 = self.img_fc3.weight  # [embed_dim, h2]
        
        # 使用einsum进行批量矩阵乘法
        # W_eq = W3 @ diag(mask2) @ W2 @ diag(mask1) @ W1
        W_eff = torch.einsum('ij,bjk,kl,blm,mn->bin', 
                            W3, 
                            torch.diag_embed(mask2),  # diag(mask2)
                            W2,
                            torch.diag_embed(mask1),  # diag(mask1) 
                            W1)
        
        return W_eff
    
    def get_classifier_weights(self):
        """获取分类器权重（类别向量）"""
        if hasattr(self, 'classifier'):
            return self.classifier.weight.data.cpu().numpy()
        return None

# 增强版增量分析器，支持更高保存频率
class EnhancedIncrementalAnalyzer:
    def __init__(self, model, device, base_save_dir="incremental_weights"):
        self.model = model
        self.device = device
        self.base_save_dir = base_save_dir
        os.makedirs(base_save_dir, exist_ok=True)
        
        # 保存配置
        self.save_count = 0
        
    def save_sampled_weights(self, test_loader, epoch, iteration, sample_fraction=0.1, force_save=False):
        """保存部分样本的权重，支持强制保存"""
        self.model.eval()
        
        save_dir = os.path.join(self.base_save_dir, f"epoch_{epoch:02d}_iter_{iteration:03d}")
        os.makedirs(save_dir, exist_ok=True)
        
        total_saved = 0
        total_failed = 0
        
        print(f"采样保存测试数据 (采样比例: {sample_fraction})...")
        
        # 保存类别向量
        classifier_weights = self.model.get_classifier_weights()
        if classifier_weights is not None:
            class_vectors_file = os.path.join(save_dir, "class_vectors.npy")
            np.save(class_vectors_file, classifier_weights.astype(np.float16))
            print(f"保存类别向量到: {class_vectors_file}")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images_flat = images.view(images.size(0), -1).to(self.device)
                
                # 随机采样，减少处理量
                batch_size = len(images_flat)
                sample_size = max(1, int(batch_size * sample_fraction))
                sample_indices = np.random.choice(batch_size, sample_size, replace=False)
                
                sample_images = images_flat[sample_indices]
                sample_labels = labels[sample_indices]
                
                try:
                    img_embed, mask1, mask2 = self.model.forward_with_masks(sample_images)
                    
                    # 使用修正的等效权重计算
                    W_eff_batch = self.model.compute_effective_weight(mask1, mask2)
                    
                    for j in range(len(sample_images)):
                        sample_id = batch_idx * batch_size + sample_indices[j]
                        label = sample_labels[j].item()
                        
                        # 获取对应样本的等效权重
                        W_eff_sample = W_eff_batch[j]  # shape: [embed_dim, input_dim]
                        
                        sample_data = {
                            'sample_id': int(sample_id),
                            'label': int(label),
                            'effective_weight': W_eff_sample.cpu().numpy().astype(np.float16),
                            'embedding': img_embed[j].cpu().numpy().astype(np.float16),
                            'mask1': mask1[j].cpu().numpy().astype(np.bool_),  # 保存mask用于验证
                            'mask2': mask2[j].cpu().numpy().astype(np.bool_),
                            'epoch': int(epoch),
                            'iteration': int(iteration),
                            'save_count': int(self.save_count)
                        }
                        
                        filename = f"sample_{sample_id:05d}.pkl"
                        filepath = os.path.join(save_dir, filename)
                        
                        if save_data_compatibly(sample_data, filepath):
                            total_saved += 1
                        else:
                            total_failed += 1
                    
                except Exception as e:
                    print(f"处理批次 {batch_idx} 时出错: {e}")
                    total_failed += len(sample_images)
                    continue
                
                # 每处理一个批次就清理内存
                del images_flat, sample_images, img_embed, mask1, mask2, W_eff_batch
                gc.collect()
        
        # 保存元数据
        metadata = {
            'epoch': epoch,
            'iteration': iteration,
            'total_saved': total_saved,
            'total_failed': total_failed,
            'sample_fraction': sample_fraction,
            'timestamp': time.time(),
            'save_count': self.save_count
        }
        
        metadata_file = os.path.join(save_dir, "metadata.pkl")
        save_data_compatibly(metadata, metadata_file)
        
        self.save_count += 1
        print(f"采样保存完成: 成功 {total_saved}, 失败 {total_failed} (总保存次数: {self.save_count})")
        return total_saved, total_failed

# 增强版训练函数，支持不同epoch的不同保存频率
def enhanced_training(model, train_loader, test_loader, epochs=2, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 初始化分析器
    analyzer = EnhancedIncrementalAnalyzer(model, device)
    
    total_iterations = len(train_loader)
    
    # 动态保存频率：第一个epoch保存更多次
    save_frequencies = {
        1: max(1, total_iterations // 8),  # 第一个epoch保存8次
        2: max(1, total_iterations // 4),  # 第二个epoch保存4次
    }
    
    best_acc = 0
    training_history = []
    
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        
        # 当前epoch的保存频率
        current_save_freq = save_frequencies.get(epoch + 1, total_iterations // 4)
        
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        print(f"保存频率: 每 {current_save_freq} 个iteration保存一次")
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪，避免训练不稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            current_acc = 100 * train_correct / train_total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })
            
            # 增量保存点 - 根据epoch动态调整频率
            should_save = (
                batch_idx % current_save_freq == 0 or  # 定期保存
                batch_idx == 0 or  # 第一个iteration
                batch_idx == total_iterations - 1 or  # 最后一个iteration
                current_acc > 80 and epoch == 0 and batch_idx % (current_save_freq // 2) == 0  # 第一个epoch准确率快速上升时保存更频繁
            )
            
            if should_save:
                print(f"\n执行增量保存: Epoch {epoch+1}, Iteration {batch_idx}, 准确率: {current_acc:.2f}%")
                
                # 保存10%的测试样本
                saved, failed = analyzer.save_sampled_weights(
                    test_loader, epoch+1, batch_idx, sample_fraction=0.1
                )
                
                # 立即清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # epoch结束时的测试
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images_flat = images.view(images.size(0), -1).to(device)
                outputs = model(images_flat)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)
        
        test_accuracy = 100 * test_correct / test_total
        print(f'Epoch {epoch+1} 完成: 测试准确率: {test_accuracy:.2f}%')
        
        training_history.append({
            'epoch': epoch+1,
            'train_accuracy': 100*train_correct/train_total,
            'test_accuracy': test_accuracy,
            'loss': loss.item()
        })
        
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(model.state_dict(), 'best_classifier.pth')
            print(f"新的最佳模型已保存，准确率: {best_acc:.2f}%")
    
    # 训练结束时的最终保存
    print("\n训练完成，执行最终保存...")
    analyzer.save_sampled_weights(test_loader, epochs, total_iterations, sample_fraction=0.1, force_save=True)
    
    print(f"最佳测试准确率: {best_acc:.2f}%")
    
    # 保存训练历史
    history_file = os.path.join(analyzer.base_save_dir, "training_history.pkl")
    save_data_compatibly(training_history, history_file)
    
    # 保存保存点的统计信息
    save_stats = {
        'total_saves': analyzer.save_count,
        'save_frequencies': save_frequencies,
        'final_accuracy': best_acc
    }
    stats_file = os.path.join(analyzer.base_save_dir, "save_statistics.pkl")
    save_data_compatibly(save_stats, stats_file)
    
    return best_acc, training_history

# 数据加载函数
def get_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=500, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset

# 验证等效权重正确性的函数
def verify_effective_weight_calculation(model, test_loader):
    """验证等效权重计算是否正确"""
    print("验证等效权重计算...")
    model.eval()
    
    with torch.no_grad():
        # 取一个批次进行验证
        images, labels = next(iter(test_loader))
        images_flat = images.view(images.size(0), -1).to(device)
        
        # 取前5个样本验证
        test_samples = images_flat[:5]
        
        # 方法1: 正常前向传播
        output_normal = model(test_samples)
        
        # 方法2: 通过等效权重计算
        embedding, mask1, mask2 = model.forward_with_masks(test_samples)
        W_eff_batch = model.compute_effective_weight(mask1, mask2)
        
        # 通过等效权重计算输出
        output_via_weff = []
        for i in range(len(test_samples)):
            # W_eff @ x 然后通过分类器
            linear_output = W_eff_batch[i] @ test_samples[i]  # [embed_dim]
            classifier_output = model.classifier(linear_output.unsqueeze(0))  # [1, num_classes]
            output_via_weff.append(classifier_output)
        
        output_via_weff = torch.cat(output_via_weff, dim=0)
        
        # 比较两种方法的输出
        diff = torch.abs(output_normal - output_via_weff).max()
        print(f"等效权重验证 - 最大输出差异: {diff.item():.6f}")
        
        if diff < 1e-4:
            print("✓ 等效权重计算正确！")
            return True
        else:
            print("✗ 等效权重计算可能有误！")
            return False

# 内存监控装饰器
def memory_monitor(func):
    def wrapper(*args, **kwargs):
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        print(f"函数 {func.__name__} 内存使用: {memory_used:.2f} MB")
        
        return result
    return wrapper

@memory_monitor
def main_enhanced_training(mode='classifier'):
    print(f"=== 增强版增量训练 - {mode.upper()} 模式 ===")
    print("优化特性:")
    print("  • 修正的等效权重计算")
    print("  • 第一个epoch更高保存频率")
    print("  • 动态保存策略")
    print("  • 内存优化管理")
    
    # 1. 加载数据
    print("1. 加载MNIST数据...")
    train_loader, test_loader, train_dataset, test_dataset = get_mnist_data(batch_size=64)
    
    # 2. 创建轻量模型
    print("2. 初始化网络...")
    model = LightweightDualModeNetwork(
        hidden1=128,
        hidden2=64,
        embed_dim=32,
        mode=mode
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 3. 验证等效权重计算
    print("3. 验证等效权重计算...")
    verification_passed = verify_effective_weight_calculation(model, test_loader)
    if not verification_passed:
        print("警告: 等效权重验证未通过，但继续训练...")
    
    # 4. 训练模型
    print("4. 开始增强训练...")
    accuracy, history = enhanced_training(
        model, train_loader, test_loader,
        epochs=2, 
        lr=0.01
    )
    
    print("训练完成！")
    
    # 显示磁盘使用情况
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk("incremental_weights"):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
            file_count += 1
    
    print(f"总保存点数: {file_count} 个文件")
    print(f"总磁盘使用: {total_size / 1024 / 1024:.2f} MB")
    
    return model, accuracy, history

if __name__ == "__main__":
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        model, accuracy, history = main_enhanced_training(mode='classifier')
        print(f"\n=== 训练总结 ===")
        print(f"最终准确率: {accuracy:.2f}%")
        print(f"所有数据保存在: incremental_weights/ 目录")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        print("尝试清理内存...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
