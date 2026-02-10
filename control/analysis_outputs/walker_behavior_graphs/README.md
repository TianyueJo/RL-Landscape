# 行为空间图可视化结果

## 目录说明

本目录包含使用不同阈值生成的 Walker2d-v4 策略行为空间图可视化结果。

## 数据来源

- **策略**: task 0-15 (共16个策略)
- **环境**: Walker2d-v4
- **数据文件**: `../behavior_space_pca_3d.npz`

## 文件组织

### PCA2 维度 (6个图)
- `behavior_graph_pca2_threshold_5_7260.png` - 10%分位数阈值
- `behavior_graph_pca2_threshold_9_6406.png` - 25%分位数阈值
- `behavior_graph_pca2_threshold_14_3736.png` - 50%分位数阈值（中位数）
- `behavior_graph_pca2_threshold_22_7754.png` - 75%分位数阈值
- `behavior_graph_pca2_threshold_31_3402.png` - 90%分位数阈值
- `behavior_graph_pca2_threshold_27_0895.png` - 之前生成的中位数阈值

### PCA6 维度 (6个图)
- `behavior_graph_pca6_threshold_16_8326.png` - 10%分位数阈值
- `behavior_graph_pca6_threshold_20_2259.png` - 25%分位数阈值
- `behavior_graph_pca6_threshold_24_6846.png` - 50%分位数阈值（中位数）
- `behavior_graph_pca6_threshold_29_9993.png` - 75%分位数阈值
- `behavior_graph_pca6_threshold_34_5371.png` - 90%分位数阈值
- `behavior_graph_pca6_threshold_27_0895.png` - 之前生成的中位数阈值

### PCA10 维度 (6个图)
- `behavior_graph_pca10_threshold_19_3107.png` - 10%分位数阈值
- `behavior_graph_pca10_threshold_23_8616.png` - 25%分位数阈值
- `behavior_graph_pca10_threshold_27_0895.png` - 50%分位数阈值（中位数）
- `behavior_graph_pca10_threshold_31_8387.png` - 75%分位数阈值
- `behavior_graph_pca10_threshold_36_0688.png` - 90%分位数阈值
- `behavior_graph_pca10_threshold_27_0895.png` - 之前生成的中位数阈值

## 观察结果

### 关键发现

1. **T4 策略的特殊性**:
   - 在大多数阈值下，T4 都是孤立节点
   - 只有在较高阈值（90%分位数）时，T4 才会与其他策略连接
   - 这表明 T4 的行为与其他策略有显著差异

2. **连通分量变化**:
   - **低阈值** (10-25%分位数): 多个小连通分量，只有最相似的策略连接
   - **中阈值** (50%分位数): 通常形成2个连通分量（15个策略 + T4孤立）
   - **高阈值** (75-90%分位数): 所有策略连接成一个大的连通分量

3. **不同 PCA 维度的差异**:
   - **PCA2**: 距离范围较小（2.38-42.38），更容易形成连接
   - **PCA6**: 距离范围中等（11.02-42.87）
   - **PCA10**: 距离范围较大（14.65-43.38），需要更高阈值才能连接

## 距离统计

### PCA2
- 最小值: 2.38
- 最大值: 42.38
- 平均值: 16.83
- 中位数: 14.37

### PCA6
- 最小值: 11.02
- 最大值: 42.87
- 平均值: 25.28
- 中位数: 24.68

### PCA10
- 最小值: 14.65
- 最大值: 43.38
- 平均值: 27.71
- 中位数: 27.09

## 使用方法

### 重新生成图

```bash
python3 plot_behavior_graph_multiple_thresholds.py \
    --data-file analysis_outputs/behavior_space_pca_3d.npz \
    --output-dir analysis_outputs/behavior_graphs \
    --env-name Walker2d-v4 \
    --pca-dims 2 6 10
```

### 使用自定义阈值

```bash
python3 plot_behavior_graph_multiple_thresholds.py \
    --data-file analysis_outputs/behavior_space_pca_3d.npz \
    --output-dir analysis_outputs/behavior_graphs \
    --env-name Walker2d-v4 \
    --pca-dims 2 \
    --thresholds 10.0 15.0 20.0 25.0 30.0
```

### 生成分离视图（每个连通分量单独子图）

```bash
python3 plot_behavior_graph_multiple_thresholds.py \
    --data-file analysis_outputs/behavior_space_pca_3d.npz \
    --output-dir analysis_outputs/behavior_graphs \
    --env-name Walker2d-v4 \
    --pca-dims 2 \
    --separate-components
```

## 文件大小

- 总大小: ~16MB
- 每个图文件: 300KB - 1.7MB（取决于连通分量数量和边的数量）

