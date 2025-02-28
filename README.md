# Can Multimodal LLMs Perform Time Series Anomaly Detection?
This repo includes the official code and datasets for paper ["Can Multimodal LLMs Perform Time Series Anomaly Detection?"](https://arxiv.org/abs/2502.17812)

## 🕵️‍♂️ VisualTimeAnomaly
<div align="center">
<img src="teaser.png" style="width: 100%;height: 100%">
</div>

<p align="center"><b><font size="70">Left: the workflow of VisualTimeAnomaly. Right: the performance comparison across various setting.</font></b></p>

**The VisualTimeAnomaly code is coming soon (within 3 weeks)!**

## 🏆 Contributions
- The first comprehensive benchmark for multimodal LLMs (MLLMs) in time series anomaly detection (TSAD), covering diverse scenarios (univariate, multivariate, irregular) and varying anomaly granularities (point-, range-, variate-wise).
- Several critical insights significantly advance the understanding of both MLLMs and TSAD.
- We construct a large-scale dataset including 12.4k time series images, and release the dateset and code to foster future research.

## 🔎 Findings
- MLLMs detect range- and variate-wise anomalies more effectively than point-wise anomalies;
- MLLMs are highly robust to irregular time series, even with 25% of the data missing;
- Open-source MLLMs perform comparably to proprietary models in TSAD. While open-source MLLMs excel on univariate time series, proprietary MLLMs demonstrate superior effectiveness on multivariate time series.

## 📝 Citation  
If you find our work useful, please cite the below paper:
```
@article{xu2025can,
  title={Can Multimodal LLMs Perform Time Series Anomaly Detection?},
  author={Xu, Xiongxiao and Wang, Haoran and Liang, Yueqing and Yu, Philip S and Zhao, Yue and Shu, Kai},
  journal={arXiv preprint arXiv:2502.17812},
  year={2025}
}
```
