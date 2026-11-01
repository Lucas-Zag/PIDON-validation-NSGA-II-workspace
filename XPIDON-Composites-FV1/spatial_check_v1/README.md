# XPIDON 厚度场诊断
基于 XPIDON-Composites-FV1 main (160ac3fe1839c6eae255c9816f940ff95e624ff9)。不训练、不优化。使用你自己的可信本地 pickle 权重。

解压到项目根目录中的 spatial_check_v1 文件夹。环境使用现有 xpidon-test-jax，勿升级。

```bash
cd spatial_check_v1
python -m unittest discover -s tests
python spatial_check.py --demo --out outputs_demo
python spatial_check.py --repo .. --case case_example.json --tags 03333 06667 10000 --nx 81 --dt 10 --out outputs_real_case
```

标签是示例！请按同一次训练实际文件填写，如 xpidon_class_03333_T.pkl 和对应 a.pkl；按时间递增，结束为10000。默认模型目录为 ../outputs/models，可用 --models 指定。多个历史权重不得混用。输出目录必须为空。

示例工况来自仓库绘图脚本，非历史 baseline：两段升温均2.2°C/min；110°C保温58min；180°C保温105min；底部/顶部HTC为90/75 W/m²K；模具厚度25mm。零件厚度从 phy_params.json 的 part_len 读取（仓库30mm）。工况参数可修改 case_example.json，但必须处于训练范围。

输出：field.csv/npz 是 [time, position] 完整场；uniformity.csv 是每时刻温差、DoC差和最低/最高DoC；spatial_profiles.csv 是关键时刻厚度分布；summary.json 是汇总、JAX后端、逐点核查及时间子域交界跳变；uniformity.png 是三位置曲线与空间差；fields.png 是时间-厚度热图。

x_norm=0..1，z_mm=x_norm×零件厚度×1000；没有三维空间或模具场。网格先逐时间、再逐位置排列。每子域选一个时刻，底/中/顶单独调用模型，与批量预测核对，不一致中止。共享时间边界采用前一子域结果；两个模型在边界的预测跳变单独报告，不能当空间温差。保温结束与完整周期结束的指标分开。

demo 使用合成函数，空间温差恒为10°C、DoC差恒为0.02，绝不是模型预测或物理固化模拟。已通过6项单元测试和demo导出；真实权重未在此运行。不预设合格阈值，不据此声称缺陷减少。采样不能保证捕获连续场极值；有必要时用161空间点/5秒步长核对。若本地训练结构改变，需要调整脚本里的网络层。

运行后发送 summary.json、uniformity.csv、uniformity.png、fields.png，通常不需要发送完整场。
