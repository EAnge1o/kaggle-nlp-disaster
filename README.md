# Kaggle Competition: Natural Language Processing with Disaster Tweets

> Kaggle 入门赛 [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)：
> 根据推文判断是否在描述真实灾难（二分类，评价指标 F1）。
>
> 本仓库是完整解题过程的存档：本地 CPU 做特征工程与线性模型，Kaggle 免费 Notebook 算力训练
> Transformer，最后 OOF 权重网格搜索融合。**比赛已停止推进，仓库仅作记录。**

## 最终结果

**5 模型融合 OOF F1 = 0.81637**（最好的单模型 0.80540）

| 模型 | OOF F1 @0.5 | 阈值调优后 |
|---|---|---|
| TF-IDF 线性集成（LR+SGD+NB 概率平均） | 0.77487 | — |
| RoBERTa-base（Kaggle GPU，5-fold） | 0.80032 | thr 0.57 → 0.80321 |
| DistilBERT v1（Kaggle CPU） | 0.80482 | thr 0.42 → 0.80745 |
| DistilBERT v2（Kaggle CPU） | 0.80451 | thr 0.51 → 0.80503 |
| RoBERTa-large（Kaggle GPU，5-fold） | 0.80509 | thr 0.46 → 0.80540 |
| **融合（blend5，OOF 权重网格搜索）** | **0.81637** | — |

融合最优权重：distilbert-v2 `0.40` / roberta-base `0.35` / distilbert-v1 `0.15` /
roberta-large `0.10` / tfidf `0.00`。

## 流水线

```
本地 CPU（特征工程 + 线性模型）          Kaggle Notebook（免费算力）
baseline.py → v2.py → v3.py            distilbert v1/v2（CPU kernel，各 ~6h）
    └─ cache_tfidf_proba.py ─┐              └─ roberta-base / roberta-large（GPU kernel）
                             │                        │
              OOF/test 概率（.npy，kaggle API 拉回）
                             └────────┬───────────────┘
                          blend5.py：权重网格搜索 + 阈值搜索 + 重复文本 leak
                                      → submissions/submission_final_blend5.csv
```

## 关键经验

- **不要过度清洗文本**：清洗（替换 URL/@mention/HTML）对线性模型有小幅提升，但会明显伤害
  Transformer —— 后者直接用原始文本效果最好（top 公开方案的共识，实测一致）。
- **重复文本 leak**：test 中有 67 条与 train 完全相同、365 条归一化后相同的推文，标签可直接
  映射。所有生成提交的脚本最后都会用这层覆盖模型预测。
- **单模型天花板 ~0.805**：RoBERTa-large 与 DistilBERT 的 OOF 几乎持平，瓶颈在数据标注噪声
  而非模型容量；但**不同架构间的融合**仍带来约 +0.011 的提升。
- **无本地 GPU 也能完整参赛**：DistilBERT 在 Kaggle CPU Notebook 即可跑完 5-fold（约 6 小时）；
  RoBERTa 用 GPU kernel（30h/周配额足够）。工作流：本地写脚本 → `kaggle kernels push` →
  跑完 `kaggle kernels output` 拉回 OOF/test 概率 → 本地融合。

## 文件结构

```
├── baseline.py             # v1：TF-IDF(word 1-2gram, 50k) + LR/SVC/NB，5-fold CV
├── v2.py                   # +word/char 双路 TF-IDF、keyword/location 元特征、精确重复 leak
├── v3.py                   # +OOF keyword target encoding、SVD300、三模型概率平均、归一化近重复 leak
├── cache_tfidf_proba.py    # 缓存 TF-IDF 集成的 OOF/test 概率，供与 Transformer 融合
├── bert_training.py        # DistilBERT 5-fold 微调（Kaggle CPU kernel，即 distilbert-v1）
├── train_distilbert_cpu.py # DistilBERT v2（改进版，即 distilbert-v2）
├── deb_gpu/
│   ├── deberta_train.py    # RoBERTa 5-fold 微调（MODEL_NAME 切换 base/large）
│   └── kernel-metadata.json
├── blend.py / blend4.py / blend5.py   # 多模型 OOF 权重网格搜索 + 阈值搜索 + leak → 最终提交
├── submit_diag.py          # kaggle API 分步提交诊断脚本
├── kernel-metadata.json    # CPU kernel 配置（kaggle kernels push）
├── logs/                   # 4 次云端训练的完整日志（分数见各文件末尾）
└── submissions/
    └── submission_final_blend5.csv    # 最终提交文件
```

## 复现

1. 从比赛页面下载 `train.csv` / `test.csv` 放入仓库根目录（按 Kaggle 规则比赛数据不入库）。
2. 本地线性模型：`python baseline.py` → `python v3.py` → `python cache_tfidf_proba.py`
3. 云端 Transformer：配置好 Kaggle API 后 `kaggle kernels push`（GPU 版改
   `deb_gpu/deberta_train.py` 中的 `MODEL_NAME` 切换 roberta-base/large），跑完后
   `kaggle kernels output <kernel> -p downloads_xxx` 拉回概率文件。
4. 融合出提交：`python blend5.py`

依赖见 `requirements.txt`；各次云端训练的完整过程日志在 `logs/`。
