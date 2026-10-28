# 动态模型轨迹预测

这个文件夹包含轨迹预测脚本，用于通过训练好的 GRU 动态代理模型进行 free-run 轨迹预测。

## 目录结构

- `scripts/`：预测入口脚本和 GRU checkpoint 加载逻辑。
- `datasets/`：用于预测输入的轨迹 CSV 文件。
- `model_weights/`：训练好的模型权重和训练配置 JSON。
- `prediction_results/`：预测输出结果。每条轨迹会保存到单独的子文件夹中。

## 模型信息

- 默认模型：`B_exogenous_u_only`
- 输入变量：`Tano_in, RHano_in, Pano_in, STC_ano, XH2_dry, Tcat_in, RHcat_in, Pcat_in, STC_cat, TCL_in, FR_CL, I`
- 输出变量：`sa, ua, lambda_airin`
- 预测方式：先对轨迹做 1 秒重采样，然后从轨迹起点开始进行 free-run 预测。启动阶段使用 `u(0)` 对历史输入做左填充，与 notebook 中的外部轨迹预测逻辑保持一致。

## Python 环境配置

推荐使用独立的 Python 环境运行，避免和系统 Python 或其他项目的依赖冲突。建议 Python 版本使用 `3.10` 或 `3.11`。

下面两种方式任选一种即可：

- 方案 A：使用 conda 管理环境。
- 方案 B：使用 Python 自带的 `venv` 管理环境。

### 1A. 使用 conda 创建并激活环境

在 PowerShell 或 Anaconda Prompt 中运行：

```powershell
conda create -n pemfc-dynamic python=3.10 -y
conda activate pemfc-dynamic
```

如果已经有自己的环境，也可以直接激活已有环境：

```powershell
conda activate <你的环境名>
```

### 1B. 使用 venv 创建并激活环境

如果不使用 conda，也可以用 Python 自带的 `venv`：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 提示禁止执行脚本，可以在当前终端临时放开执行策略后再激活：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

以后每次重新打开终端，只需要在本目录重新激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. 安装 Python 包

本目录脚本依赖以下 Python 包：

```powershell
python -m pip install numpy pandas matplotlib scikit-learn scipy torch
```

SciPy 用于 CSV/TSV 的压力平滑，以及原始 TSV 的 `STC_ano` 插值：

```powershell
python -m pip install scipy
```

如果需要使用 GPU，请根据本机 CUDA 版本到 PyTorch 官网选择对应安装命令；如果不确定，先使用上面的 CPU 版本即可。

### 3. 验证环境

安装完成后，可以运行下面的命令检查依赖是否可正常导入：

```powershell
python -c "import numpy, pandas, matplotlib, sklearn, scipy, torch; print('Python env OK'); print('torch cuda:', torch.cuda.is_available())"
```

如果使用 conda，以后每次重新打开终端运行脚本前，先进入项目目录并激活环境：

```powershell
conda activate pemfc-dynamic
```

## 运行预测

在项目根目录 `dynamic_trajectory_prediction` 中运行：

```powershell
python .\scripts\predict_trajectories.py
```

默认会预测以下目录中的所有 CSV 文件：

```text
datasets/
```

只预测某一个 CSV：

```powershell
python .\scripts\predict_trajectories.py --csv .\datasets\20240625_777_2D_F95.csv
```

直接预测原始导出的 TSV 文件：

```powershell
python .\scripts\predict_trajectories.py --tsv .\datasets\your_export.tsv
```

脚本会先调用 `scripts/interpolate_stc.py`，把 TSV 转换为代理模型需要的轨迹 CSV 格式，然后再预测。转换后的 CSV 默认保存到：

```text
prediction_results/_converted_inputs/
```

如需指定转换后 CSV 的保存位置：

```powershell
python .\scripts\predict_trajectories.py --tsv .\datasets\your_export.tsv --converted-dir .\datasets\converted
```

指定自己的数据集文件夹：

```powershell
python .\scripts\predict_trajectories.py --dataset-dir .\path\to\your_csv_folder
```


## 输入预处理

CSV（包括自动发现的数据集）和 TSV 均在预测读取阶段、重采样之前执行同一套预处理：

- `Pano_in`、`Pcat_in`：Savitzky–Golay 平滑，窗口 60 个采样点、三阶多项式；短轨迹自动缩短窗口。
- `I`：负值截断为 0。
- `STC_ano`、`STC_cat`：电流为 0 时设为 10；电流非 0 且 STC 为 0 时，用前一有效 STC 填充，无前值时保留 0。
- 原始 TSV 转换后保留 `Com_nP1ActSpd` 列；存在该列时，ARB 转速为 0 的 `STC_ano` 最终仍置为 0。标准 CSV 缺少转速时仅执行通用规则。

公共规则位于 `scripts/trajectory_preprocessing.py`。预测入口的 TSV 转换仅转换格式和插值，随后统一预处理，避免一次运行中重复平滑。单独运行 `interpolate_stc.py` 仍默认输出预处理后的数据；预测时应输入未平滑的轨迹，避免再次平滑。

这些操作不会改写源数据集；已有预测结果和片段图需要重新运行才能更新。冷却流量 `FR_CL` 不做额外单位换算。

## 输出结果

脚本不会在 `prediction_results/` 根目录下保存汇总 CSV。预测结果按轨迹名称分别保存：

```text
prediction_results/
    <trajectory_id>/
    <trajectory_id>_free_run_predictions.csv
    <trajectory_id>_free_run_metrics.csv
    <trajectory_id>_free_run_prediction.png
    <trajectory_id>_free_run_prediction_with_current.png
    <trajectory_id>_input_trajectories.png
    <trajectory_id>_free_run_ua_sa_criteria.png
```

其中：

- `<trajectory_id>_free_run_predictions.csv`：该轨迹的逐时刻预测结果。
- `<trajectory_id>_free_run_metrics.csv`：该轨迹的预测误差指标；只有输入 CSV 中包含真实 `sa/ua/lambda_airin` 时才有有效指标。
- `<trajectory_id>_free_run_prediction.png`：该轨迹预测图。
- `<trajectory_id>_free_run_prediction_with_current.png`：三个预测变量和电流 `I` 的轨迹图。
- `<trajectory_id>_input_trajectories.png`：12 个输入变量的轨迹图。
- `<trajectory_id>_free_run_ua_sa_criteria.png`：`ua_pred–sa_pred` 阈值分区散点图；存在成对真实值时叠加空心圆对照。阈值点为 `(0.5, 0.032)`、`(1.5, 0.062)`、`(2.5, 0.092)`，第二条阈值线上移 `0.015`，依次填充绿、蓝、红三个区域。分区仅覆盖 `ua=0.5～2.5`，显示范围为 `ua=0～6`、`sa=0～0.16`，范围外的点不显示。

预测图使用与 notebook 外部 free-run 预测图一致的 matplotlib 风格：

- 黑色曲线：真实值。
- 橙色曲线：预测值。
- 灰色阴影：startup padded 区间。
- 三个共享时间轴的子图：`sa`、`ua`、`lambda_airin`。

如果输入 CSV 中没有真实的 `sa`、`ua` 或 `lambda_airin`，脚本仍会生成预测值；缺失真值对应的评价指标会自动跳过。
