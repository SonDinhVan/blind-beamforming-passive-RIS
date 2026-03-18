This repo provides code implementation for the journal paper:

Paper title: Near-optimal Reconfigurable Intelligent Surface Configuration: Blind Beamforming with Sensing.

Abstract: Blind beamforming has emerged as a promising approach to configure reconfigurable intelligent surfaces (RISs) without relying on channel state information (CSI) or geometric models, making it directly compatible with commodity hardware. In this paper, we propose a new blind beamforming algorithm, so-called Blind Optimal RIS Beamforming with Sensing (\textsc{BORN}), that operates using only received signal strength (RSS). In contrast to existing methods that rely on majority-voting mechanisms, \textsc{BORN} exploits the intrinsic quadratic structure of the received signal-to-noise ratio (SNR). The algorithm proceeds in two stages: \emph{sensing}, where a quadratic model is estimated from RSS measurements, and \emph{optimization}, where the RIS configuration is obtained using the estimated quadratic model. Our novelties are twofold. Firstly, we show for the first time, that \textsc{BORN} can achieve provable near-optimal performance using only $O(N \log_2(N))$ samples, where $N$ is the number of RIS elements. Secondly, as a by-product of our analysis, we show that quadratic models are learnable under Rademacher feature distributions when the second-order coefficient matrix is low-rank. This result, to our knowledge, has not been established in prior matrix sensing literature. Extensive simulations and real-world field tests demonstrate that \textsc{BORN} achieves near-optimal performance, substantially outperforming state-of-the-art blind beamforming algorithms, particularly in scenarios with a weak background channel such as non-line-of-sight (NLOS).


Install:
- From the repo root, run: `pip install -e.`
- I recommend to install environment from `environment.yml` via:
  - `conda env create -f environment.yml`
  - `conda activate <env-name>`

To play around with the algorithm:
- Refer to `notebooks/analysis.ipynb`.

To reproduce the simulation results:
- Refer to `notebooks/simulation results/`.

Copyright: © 2025 Son Dinh-Van. All rights reserved.
