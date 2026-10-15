# N/D/Q 数据诊断

B6–B8 为“基于真实数据校准的半合成补充实验”，不是直接训练实验。

所有主键均经 Decimal 精确数值规范化：0.07 与 0.070 是同一工作点；未作近似匹配、填补、去重或质量反向。

## B6

```json
{
  "dataset": "B6",
  "source_file": "real_attachments/B_scaling_laws/supplementary_NQ_experiment.csv",
  "rows": 360,
  "columns": [
    "experiment_id",
    "N_params_B",
    "D_tokens_B",
    "Q_score",
    "val_loss"
  ],
  "missing_or_nan_cells": 0,
  "nan_required_cells": 0,
  "inf_required_cells": 0,
  "exact_duplicate_groups": 0,
  "exact_duplicate_excess_rows": 0,
  "duplicate_candidate_key_groups": 0,
  "unique_N": 9,
  "unique_D": 5,
  "unique_Q": 8,
  "unique_ND": 45,
  "expected_grid_points": 360,
  "actual_unique_grid_points": 360,
  "missing_grid_points": 0,
  "grid_coverage": 1.0,
  "complete_cartesian_grid": true,
  "N_params_B_min": 0.07,
  "N_params_B_max": 11.97,
  "D_tokens_B_min": 10.0,
  "D_tokens_B_max": 600.0,
  "Q_score_min": 0.1,
  "Q_score_max": 1.0,
  "val_loss_min": 2.0161,
  "val_loss_max": 3.7216
}
```

## B7

```json
{
  "dataset": "B7",
  "source_file": "real_attachments/B_scaling_laws/supplementary_NQ_experiment_expanded.csv",
  "rows": 450,
  "columns": [
    "experiment_id",
    "N_params_B",
    "D_tokens_B",
    "Q_score",
    "val_loss"
  ],
  "missing_or_nan_cells": 0,
  "nan_required_cells": 0,
  "inf_required_cells": 0,
  "exact_duplicate_groups": 0,
  "exact_duplicate_excess_rows": 0,
  "duplicate_candidate_key_groups": 0,
  "unique_N": 9,
  "unique_D": 5,
  "unique_Q": 10,
  "unique_ND": 45,
  "expected_grid_points": 450,
  "actual_unique_grid_points": 450,
  "missing_grid_points": 0,
  "grid_coverage": 1.0,
  "complete_cartesian_grid": true,
  "N_params_B_min": 0.07,
  "N_params_B_max": 11.97,
  "D_tokens_B_min": 10.0,
  "D_tokens_B_max": 600.0,
  "Q_score_min": 0.1,
  "Q_score_max": 1.0,
  "val_loss_min": 2.0161,
  "val_loss_max": 3.7216
}
```

## B8

```json
{
  "dataset": "B8",
  "source_file": "real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv",
  "rows": 1704,
  "columns": [
    "experiment_id",
    "N_params_B",
    "D_tokens_B",
    "Q_score",
    "val_loss",
    "data_type"
  ],
  "missing_or_nan_cells": 0,
  "nan_required_cells": 0,
  "inf_required_cells": 0,
  "exact_duplicate_groups": 0,
  "exact_duplicate_excess_rows": 0,
  "duplicate_candidate_key_groups": 0,
  "unique_N": 15,
  "unique_D": 10,
  "unique_Q": 12,
  "unique_ND": 150,
  "expected_grid_points": 1800,
  "actual_unique_grid_points": 1704,
  "missing_grid_points": 96,
  "grid_coverage": 0.9466666666666667,
  "complete_cartesian_grid": false,
  "N_params_B_min": 0.07,
  "N_params_B_max": 700.0,
  "D_tokens_B_min": 5.0,
  "D_tokens_B_max": 2000.0,
  "Q_score_min": 0.05,
  "Q_score_max": 1.0,
  "val_loss_min": 0.5,
  "val_loss_max": 3.3659
}
```

候选主键 (N,D,Q) 无重复，B7_nonoverlap 排除 B6 主键；B8_nonoverlap 排除 B6∪B7 主键。
Q 单调性按精确固定 (N,D) 分组；相邻 Loss 增加超过 1e-12 为违背，相等不是违背。
该容差仅防浮点误差；不是人为调整数据、噪声阈值或统计显著性阈值。
B8 的 Q–Loss 方向与主假设相反。已触发用户指定的拟合停止条件。
