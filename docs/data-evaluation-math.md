# 数据评估数学公式说明

本文从数学公式角度说明机器人采集数据的质量评估方式。它对应一条从“能不能用”到“值不值得用于训练”的评估链路：

$$
\text{硬门槛准入}\rightarrow\text{模态质量评分}\rightarrow\text{轨迹语义评估}\rightarrow\text{训练价值决策}
$$

需要注意：自动质检里的 `overall_score` 更接近“检查项通过率”，不能直接等价为“任务成功率”或“训练价值”。本文因此把分数、通过条件、决策标签分开定义。

---

## 1. 基本符号

| 符号 | 含义 |
|---|---|
| \(D\) | 数据集 |
| \(N\) | 数据集中的 episode 数量 |
| \(e\) | 单个 episode |
| \(K\) | 单个 episode 中运行的验证器数量 |
| \(v_k\) | 第 \(k\) 个验证器 |
| \(n_k\) | 第 \(k\) 个验证器产生的检查项数量 |
| \(i_{k,j}\) | 第 \(k\) 个验证器的第 \(j\) 个检查项 |
| \(x_{e,k,j}\) | 检查项通过结果 |
| \(l_{e,k,j}\) | 检查项等级，取值为 `critical`、`major`、`minor`、`info` |
| \(w_{k,j}\) | 检查项权重 |
| \(c\) | 质量类别，例如 metadata、timing、action、visual、depth、trajectory |

$$
D=\{e_1,e_2,\dots,e_N\}
$$

$$
V_e=\{v_1,v_2,\dots,v_K\}
$$

$$
I_{e,k}=\{i_{k,1},i_{k,2},\dots,i_{k,n_k}\}
$$

$$
x_{e,k,j}=\begin{cases}
1, & \text{检查通过}\\
0, & \text{检查失败}
\end{cases}
$$

---

## 2. 评估对象分层

| 层级 | 目标 | 输出 |
|---|---|---|
| 检查项 | 判断一个具体条件是否满足 | \(x_{e,k,j}\in\{0,1\}\) |
| 验证器 | 汇总同一类检查项 | \(S_{e,k}\)、\(P_{e,k}\) |
| 质量类别 | 汇总同一模态或同一质量维度 | \(S_{e,c}\)、\(P_{e,c}\) |
| Episode | 汇总单条采集样本的质量 | \(S_e\)、\(P_e\)、\(d_e\) |
| 数据集 | 汇总批量数据质量与可用性 | \(S_D\)、\(R_{\text{pass}}\)、分档统计 |

其中：

| 符号 | 含义 |
|---|---|
| \(S\) | 分数 |
| \(P\) | 是否通过 |
| \(d_e\) | episode 决策标签 |

---

## 3. 检查项分数与验证器分数

### 3.1 当前工程里的简单通过率分数

当前自动质检的基础分数可以抽象为：

$$
S_{e,k}^{\text{plain}}
=
100\cdot\frac{1}{n_k}\sum_{j=1}^{n_k}x_{e,k,j}
$$

它的含义是“检查项通过率”，优点是直观，缺点是默认所有检查项等权。

### 3.2 推荐的加权验证器分数

为了避免“缺一个关键文件”和“轻微色偏”被同等对待，推荐使用加权形式：

$$
S_{e,k}
=
100\cdot
\frac{\sum_{j=1}^{n_k}w_{k,j}x_{e,k,j}}{\sum_{j=1}^{n_k}w_{k,j}}
$$

默认权重可按等级初始化：

| 等级 | 建议权重 | 业务含义 |
|---|---:|---|
| `critical` | \(1.00\) | 失败后数据通常不可用 |
| `major` | \(0.70\) | 失败后显著影响训练或复核成本 |
| `minor` | \(0.30\) | 失败后可降权使用或提示优化 |
| `info` | \(0.05\) | 仅提供上下文，不应主导分数 |

---

## 4. 通过条件

通过条件不应该由平均分单独决定，而应该由阻塞检查项决定。

定义阻塞检查项集合：

$$
B_{e,k}=\{j\mid l_{e,k,j}\in\{\text{critical},\text{major}\}\}
$$

验证器通过：

$$
P_{e,k}=\prod_{j\in B_{e,k}}x_{e,k,j}
$$

类别通过：

$$
P_{e,c}=\prod_{k\in c}P_{e,k}
$$

Episode 通过：

$$
P_e=\prod_{k=1}^{K}P_{e,k}
$$

数据集通过数量、失败数量和通过率：

$$
N_{\text{pass}}=\sum_{e=1}^{N}P_e
$$

$$
N_{\text{fail}}=N-N_{\text{pass}}
$$

$$
R_{\text{pass}}=\frac{N_{\text{pass}}}{N}
$$

核心准入条件：

$$
\boxed{
P_e=1
\iff
\forall k,\forall j\in B_{e,k},\ x_{e,k,j}=1
}
$$

---

## 5. Episode 总分与类别总分

### 5.1 当前简单总分

工程实现中常见的整体分数可以写成：

$$
S_e^{\text{plain}}
=
100\cdot
\frac{\sum_{k=1}^{K}\sum_{j=1}^{n_k}x_{e,k,j}}{\sum_{k=1}^{K}n_k}
$$

这表示“所有检查项的整体通过率”。

### 5.2 推荐类别加权总分

为了让不同模态有清晰语义，先计算类别分数：

$$
S_{e,c}
=
100\cdot
\frac{\sum_{(k,j)\in c}w_{k,j}x_{e,k,j}}{\sum_{(k,j)\in c}w_{k,j}}
$$

再计算 episode 分数：

$$
S_e
=
\sum_{c\in C}\lambda_c S_{e,c},
\quad
\sum_{c\in C}\lambda_c=1
$$

建议初始类别权重：

| 类别 | 权重 \(\lambda_c\) | 说明 |
|---|---:|---|
| Metadata / 文件完整性 | 0.20 | 决定数据能否解析 |
| Timing / 时间同步 | 0.15 | 决定多模态对齐可靠性 |
| Action / 动作质量 | 0.20 | 决定策略学习信号质量 |
| Visual / 视觉质量 | 0.15 | 决定视觉策略或回放可用性 |
| Depth / 深度质量 | 0.10 | 决定 3D 感知与空间推理可靠性 |
| Trajectory / 轨迹语义 | 0.20 | 决定行为是否接近高质量示范 |

这些权重不是常数真理，应按任务类型、机器人型号、训练目标重新标定。

---

## 6. 决策标签

分数用于排序，通过条件用于准入，最终业务动作建议用决策标签表示。

$$
d_e\in\{\text{accept},\text{review},\text{low\_weight},\text{reject}\}
$$

### 6.1 accept

$$
P_e=1\land S_e\ge85\land C_{\text{task}}\ge0.75
$$

含义：可直接进入高质量训练集。

### 6.2 review

$$
P_e=1\land 60\le S_e<85
$$

含义：自动检查没阻塞，但建议人工复核。关键语义置信度不足时也应进入复核。

### 6.3 low_weight

$$
P_e=1\land S_e<60
$$

含义：没有 `critical` 失败时可保留，但训练时降低采样权重。

### 6.4 reject

$$
P_e=0
$$

含义：存在不可恢复 `critical` 失败时，不进入训练集。

对应训练采样权重可定义为：

$$
\omega_e=
\begin{cases}
1.0, & d_e=\text{accept}\\
0.5, & d_e=\text{review}\\
0.2, & d_e=\text{low\_weight}\\
0, & d_e=\text{reject}
\end{cases}
$$

---

# 具体指标公式

## 7. Metadata / 文件完整性评估

定义必需文件集合：

$$
F_{\text{required}}=\{f_1,f_2,\dots,f_R\}
$$

文件存在性：

$$
m_{e,r}=\mathbf{1}[f_r\in e]
$$

### 7.1 必需文件完整率

$$
R_{\text{file}}=\frac{1}{R}\sum_{r=1}^{R}m_{e,r}
$$

通过条件：所有 `critical` 文件必须存在。

### 7.2 元数据字段完整率

设关键元数据字段数量为 \(Q\)，第 \(q\) 个字段是否存在为 \(h_q\)：

$$
h_q=\mathbf{1}[q\in\text{metadata}]
$$

$$
R_{\text{meta}}=\frac{1}{Q}\sum_{q=1}^{Q}h_q
$$

通过条件：关键字段必须存在。

### 7.3 格式可解析

$$
P_{\text{parse}}=\mathbf{1}[\text{metadata/state/action 可解析}]
$$

通过条件：

$$
P_{\text{parse}}=1
$$

Metadata 是硬门槛：如果关键文件缺失或格式不可解析，后续视觉、动作、轨迹分数不应掩盖这个失败。

---

## 8. 时间戳评估

设时间戳序列为：

$$
t_1,t_2,\dots,t_M
$$

时间间隔为：

$$
\Delta t_i=t_i-t_{i-1},\quad i=2,3,\dots,M
$$

正时间间隔集合为：

$$
\Delta T^+=\{\Delta t_i\mid \Delta t_i>0\}
$$

### 8.1 单调性

$$
R_{\text{mono}}=1-\frac{\#\{i:\Delta t_i\le 0\}}{M-1}
$$

默认阈值：\(0.99\)。通过条件：

$$
R_{\text{mono}}\ge0.99
$$

### 8.2 采样间隔变异系数

$$
CV_{\Delta t}=\frac{\sigma(\Delta T^+)}{\mu(\Delta T^+)}
$$

默认阈值：\(0.05\)。通过条件：

$$
CV_{\Delta t}<0.05
$$

### 8.3 估计采样频率

$$
f=\frac{1}{\operatorname{median}(\Delta T^+)}
$$

默认阈值：\(20\ \text{Hz}\)。通过条件：

$$
f\ge20
$$

### 8.4 大间隔比例

$$
R_{\text{gap}}=\frac{\#\{i:\Delta t_i>1.0\}}{|\Delta T^+|}
$$

默认阈值：\(0.01\)。通过条件：

$$
R_{\text{gap}}<0.01
$$

### 8.5 频率一致性

先去掉明显异常的采样间隔，得到裁剪后的集合 \(\Delta T^{\text{trim}}\)：

$$
C_f=1-\frac{\sigma(\Delta T^{\text{trim}})}{\mu(\Delta T^{\text{trim}})}
$$

默认阈值：\(0.98\)。通过条件：

$$
C_f\ge0.98
$$

---

## 9. 动作序列评估

设动作序列为：

$$
a_i=(a_{i,1},a_{i,2},\dots,a_{i,m})
$$

### 9.1 相邻动作最大变化量

$$
d_i=\max_r |a_{i,r}-a_{i-1,r}|
$$

默认阈值：\(0.001\)。当满足：

$$
d_i<0.001
$$

该时间步可认为处于静止状态。

### 9.2 最长连续静止时长

设 \(C\) 是连续满足静止条件的时间片集合：

$$
T_{\text{static}}=\max_C\sum_{i\in C}\Delta t_i
$$

通过条件：

$$
T_{\text{static-all}}\le3.0
$$

$$
T_{\text{static-key}}\le5.0
$$

含义：全关节异常静止不能超过 3 秒，关键关节异常静止不能超过 5 秒。

### 9.3 动作时长

$$
T_{\text{action}}=t_M-t_1
$$

默认阈值：\(1.0\ \text{s}\)。通过条件：

$$
T_{\text{action}}\ge1.0
$$

### 9.4 缺失值比例

$$
R_{\text{nan}}=
\frac{\#\{a_{i,r}:a_{i,r}\text{ 是 NaN 或 None}\}}{M\cdot m}
$$

默认阈值：\(0.01\)。通过条件：

$$
R_{\text{nan}}<0.01
$$

### 9.5 最大关节速度

$$
v_{\max}=\max_{i,r}\left|\frac{a_{i,r}-a_{i-1,r}}{\Delta t_i}\right|
$$

默认阈值：\(3.14\ \text{rad/s}\)。通过条件：

$$
v_{\max}<3.14
$$

---

## 10. 末端执行器轨迹评估

设末端位姿和夹爪序列为：

$$
p_t\in\mathbb{R}^3,
\quad
R_t\in SO(3),
\quad
g_t\in\mathbb{R}
$$

### 10.1 夹爪运动幅度

$$
A_g=\max_t g_t-\min_t g_t
$$

默认阈值：\(0.05\)。通过条件：

$$
A_g\ge0.05
$$

### 10.2 抓取 / 放置事件数量

$$
C_{\text{event}}=\#\{\text{detected grasp/place events}\}
$$

默认阈值：\(1\)。通过条件：

$$
C_{\text{event}}\ge1
$$

### 10.3 末端路径长度

$$
L_{\text{ee}}=\sum_{t=2}^{T}\|p_t-p_{t-1}\|_2
$$

阈值与任务相关。路径不应过短，也不应异常过长。

### 10.4 末端速度峰值

$$
v_{\text{ee,max}}=
\max_t\frac{\|p_t-p_{t-1}\|_2}{\Delta t_t}
$$

阈值与机器人型号相关。通过条件是不能超过物理上限。

---

## 11. 视觉数据评估

设图像像素为 \(p\)，RGB 通道为 \((R_p,G_p,B_p)\)。

灰度值：

$$
g_p=\frac{R_p+G_p+B_p}{3}
$$

### 11.1 过曝比例

$$
R_{\text{over}}=\frac{\#\{p:g_p>250\}}{\#\{p\}}
$$

默认阈值：\(0.05\)。通过条件：

$$
R_{\text{over}}\le0.05
$$

### 11.2 欠曝比例

$$
R_{\text{under}}=\frac{\#\{p:g_p<5\}}{\#\{p\}}
$$

默认阈值：\(0.10\)。通过条件：

$$
R_{\text{under}}\le0.10
$$

### 11.3 黑屏比例

$$
R_{\text{black}}=\frac{\#\{p:g_p<2\}}{\#\{p\}}
$$

默认阈值：\(0.95\)。通过条件：

$$
R_{\text{black}}<0.95
$$

### 11.4 白屏比例

$$
R_{\text{white}}=\frac{\#\{p:g_p>253\}}{\#\{p\}}
$$

默认阈值：\(0.95\)。通过条件：

$$
R_{\text{white}}<0.95
$$

### 11.5 色偏

令 \(\mu_R,\mu_G,\mu_B\) 分别为 RGB 三个通道的均值：

$$
C_{\text{shift}}=\frac{\sigma(\mu_R,\mu_G,\mu_B)}{255}
$$

默认阈值：\(0.10\)。通过条件：

$$
C_{\text{shift}}\le0.10
$$

---

## 12. 深度数据评估

设深度图像像素深度为 \(d_p\)。

### 12.1 无效深度比例

$$
R_{\text{invalid}}=
\frac{\#\{p:d_p=0\lor d_p=\text{NaN}\}}{\#\{p\}}
$$

默认阈值：\(0.10\)。通过条件：

$$
R_{\text{invalid}}\le0.10
$$

### 12.2 有效深度像素集合

$$
M_t=\{p:d_{t,p}>0\land d_{t,p}\ne\text{NaN}\}
$$

### 12.3 深度连续性

$$
C_{\text{depth}}=\frac{|M_t\cap M_{t-1}|}{|M_t\cup M_{t-1}|}
$$

默认阈值：\(0.90\)。通过条件：

$$
C_{\text{depth}}\ge0.90
$$

---


# Reference Tube DTW 轨迹异常检测

这一节对应 RoboClaw 当前质检链路里更接近生产使用的 DTW：不是 K-medoids，也不是 ProSemA，而是用同任务、同机器人、同轨迹表示下的其他高质量样本构建一个 reference tube，再把候选 episode 对齐到这个标准动作通道上，诊断 `Deviation`、`Hesitate`、`Stall` 等异常。候选 episode 不能进入自己的 reference tube；如果没有外部参考，应标记为未验证，而不是通过。

对应实现：

- `roboclaw/data/curation/trajectory_quality.py`
- `roboclaw/data/curation/reference_tube.py`

---

## R1. Reference Tube 基本符号

设同一个任务下有一组历史高质量参考 episode：

$$
\mathcal{R}=\{R^{(1)},R^{(2)},\dots,R^{(n)}\}
$$

第 \(r\) 条参考轨迹是多关节时间序列：

$$
R^{(r)}=(q^{(r)}_1,q^{(r)}_2,\dots,q^{(r)}_{T_r})
$$

其中每一帧：

$$
q^{(r)}_t\in\mathbb{R}^{J}
$$

\(J\) 是参与检测的关节数量。

---

## R2. 参考锚点选择

工程实现里会选择时长最接近中位数的参考轨迹作为时间轴锚点。设参考轨迹长度为：

$$
L_r=T_r
$$

锚点索引：

$$
r^*=\arg\min_r |L_r-\operatorname{median}(L_1,L_2,\dots,L_n)|
$$

锚点轨迹：

$$
A=R^{(r^*)}
$$

作用：把所有参考轨迹对齐到同一个标准时间轴，避免不同采集速度直接平均造成语义错位。

---

## R3. Windowed exact DTW 对齐

对每条参考轨迹 \(R^{(r)}\)，使用 DTW 将其对齐到锚点 \(A\)：

$$
\pi^{(r)}=\operatorname{DTWPath}(A,R^{(r)})
$$

路径元素为：

$$
(i,j)\in\pi^{(r)}
$$

其中 \(i\) 是锚点轨迹帧索引，\(j\) 是参考轨迹帧索引。

工程里使用的是带窗口约束的精确 DTW：

$$
\operatorname{DTW}(A,R^{(r)};\ \rho)
$$

局部距离使用多关节欧氏距离：

$$
d(q_i,q_j)=\|q_i-q_j\|_2
$$

---

## R4. 对齐后的参考轨迹库

对锚点时间步 \(i\)，收集所有和它对齐的参考帧：

$$
J_i^{(r)}=\{j\mid(i,j)\in\pi^{(r)}\}
$$

对齐后的位置：

$$
\tilde q_i^{(r)}=
\frac{1}{|J_i^{(r)}|}
\sum_{j\in J_i^{(r)}}q_j^{(r)}
$$

如果某个锚点帧没有匹配帧，工程里会沿用前一帧作为填充值。这是工程上的连续性补偿，不应被解释为真实观测。

速度定义为一阶差分：

$$
v_t^{(r)}=q_t^{(r)}-q_{t-1}^{(r)}
$$

对齐后的速度也按同样路径求均值：

$$
\tilde v_i^{(r)}=
\frac{1}{|J_i^{(r)}|}
\sum_{j\in J_i^{(r)}}v_j^{(r)}
$$

---

## R5. Reference Tube 统计量

对齐后得到标准轨迹库：

$$
\tilde{\mathcal{R}}=\{\tilde R^{(1)},\tilde R^{(2)},\dots,\tilde R^{(n)}\}
$$

每个锚点时间步的均值轨迹：

$$
\mu_i=\frac{1}{n}\sum_{r=1}^{n}\tilde q_i^{(r)}
$$

位置标准差：

$$
\sigma_i=
\max\left(
\operatorname{Std}_r(\tilde q_i^{(r)}),
\sigma_{\min}
\right)
$$

工程里位置标准差有容错地板：

$$
\sigma_{\min}=0.05
$$

速度均值：

$$
\bar v_i=\frac{1}{n}\sum_{r=1}^{n}\tilde v_i^{(r)}
$$

速度标准差：

$$
\sigma^v_i=
\max\left(
\operatorname{Std}_r(\tilde v_i^{(r)}),
\sigma^v_{\min}
\right)
$$

工程里速度标准差有容错地板：

$$
\sigma^v_{\min}=0.01
$$

---

## R6. 自适应异常阈值

Reference Tube 的阈值来自参考样本内部的自然波动，而不是固定全局阈值。

工程默认至少需要 6 条外部参考轨迹。低于该数量时，`trajectory_dtw` 应输出 `insufficient_references`，属于未验证/需复核状态，不能当作高质量通过。

### R6.1 位置偏差分布

参考样本 \(r\) 在时间步 \(i\) 的几何偏差：

$$
D_i^{(r)}=\|\tilde q_i^{(r)}-\mu_i\|_2
$$

距离阈值取内部偏差的高分位数：

$$
\tau_d=Q_{99.9}\left(\{D_i^{(r)}\}_{r,i}\right)
$$

### R6.2 速度能量分布

参考样本 \(r\) 在时间步 \(i\) 的 RMS 速度能量：

$$
E_i^{(r)}
=
\sqrt{\frac{1}{J}\sum_{k=1}^{J}(\tilde v_{i,k}^{(r)})^2}
$$

速度能量阈值：

$$
\tau_v=Q_{99.9}\left(\{E_i^{(r)}\}_{r,i}\right)
$$

### R6.3 停滞帧数阈值

工程默认：

$$
\tau_s=30
$$

含义：如果太多候选帧被 DTW 对齐到同一个参考帧，就认为候选动作在该语义位置发生停滞。

---

## R7. 候选轨迹对齐

候选 episode 的关节轨迹为：

$$
C=(c_1,c_2,\dots,c_T)
$$

先将候选轨迹对齐到 reference tube 的均值轨迹：

$$
\pi_C=\operatorname{DTWPath}(\mu,C)
$$

其中路径元素：

$$
(i,j)\in\pi_C
$$

\(i\) 是 reference tube 时间步，\(j\) 是候选轨迹帧。

---

## R8. Deviation：动作走形

候选帧 \(c_j\) 对齐到 reference tube 的时间步 \(i\) 后，计算它到参考轨迹库的最近距离：

$$
d_j^{\min}=\min_r\|\tilde q_i^{(r)}-c_j\|_2
$$

动作走形判定：

$$
\operatorname{Deviation}(j)=
\mathbf{1}\left[d_j^{\min}>1.2\tau_d\right]
$$

工程里还会找出误差最大的关节作为证据：

$$
k^*=\arg\max_k |c_{j,k}-\mu_{i,k}|
$$

证据文本类似：某个关节偏移多少弧度。

---

## R9. Hesitate：犹豫 / 颤抖

候选速度：

$$
v^C_j=c_j-c_{j-1}
$$

当前帧的 RMS 速度能量：

$$
H_j
=
\sqrt{\frac{1}{J}\sum_{k=1}^{J}(v^C_{j,k})^2}
$$

犹豫 / 颤抖判定：

$$
\operatorname{Hesitate}(j)=
\mathbf{1}\left[H_j>2.0\tau_v\right]
$$

含义：不是看位置是否偏离，而是看局部动力学是否出现异常速度能量。使用 RMS 能量而不是跨关节方差，是为了避免所有关键关节同步异常加速时方差反而接近 0。

---

## R10. Stall：进度停滞

DTW 路径天然包含“一个参考帧对应多个候选帧”的情况。对 reference tube 时间步 \(i\)，候选匹配帧集合为：

$$
C_i=\{j\mid(i,j)\in\pi_C\}
$$

如果：

$$
|C_i|>\tau_s
$$

则认为候选轨迹在参考动作的第 \(i\) 个语义位置发生停滞。

停滞持续时长：

$$
T_{\text{stall}}(i)=t_{\max C_i}-t_{\min C_i}
$$

---

## R11. 异常片段合并

逐帧异常会被合并成连续片段。设连续异常片段为：

$$
G=(j_a,j_{a+1},\dots,j_b)
$$

片段开始和结束时间：

$$
t_{\text{start}}=t_{j_a}
$$

$$
t_{\text{end}}=t_{j_b}
$$

如果片段时长过短，工程里会过滤掉，避免单帧噪声造成误报。

---

## R12. 质量分与质检集成

`AnomalyDetector.evaluate(...)` 的输出可以抽象为：

$$
( S_{\text{tube}},\ \mathcal{A})
$$

其中：

| 符号 | 含义 |
|---|---|
| \(S_{\text{tube}}\) | reference tube 轨迹质量分 |
| \(\mathcal{A}\) | 异常文本集合，例如 Deviation、Hesitate、Stall |

如果异常集合为空：

$$
\mathcal{A}=\varnothing
$$

则动作轨迹检查通过，并记录为 `INFO`。

如果异常集合非空：

$$
\mathcal{A}\ne\varnothing
$$

则每条异常会转成一个 `MAJOR` 级别的动作数据问题：

$$
P_{\text{tube}}=0
$$

这意味着它会影响 episode 的整体通过条件。

---

## R13. Reference Tube DTW 的定位

| 项 | 说明 |
|---|---|
| 算法角色 | 当前质检链路中的轨迹异常检测 |
| 对齐对象 | 候选轨迹对齐到历史高质量 reference tube |
| DTW 实现 | windowed exact DTW，局部距离按轨迹表示选择欧氏或 grouped Huber |
| 主要异常 | `Deviation`、`Hesitate`、`Stall` |
| 结果归属 | 合并进“动作数据”类别 |
| 通过影响 | 异常被标记为 `MAJOR`，会阻塞整体通过 |

---

# 轨迹语义与 ProSemA 原型对齐公式

ProSemA 可以理解为两步：

$$
\text{Prototype Discovery}\rightarrow\text{Semantic Propagation}
$$

第一步用轨迹相似度找原型 episode；第二步把原型上的语义标注传播到同簇的相似 episode。

---

## 13. 轨迹特征表示

第 \(e\) 个 episode 表示成多维时间序列：

$$
X_e=(x_{e,1},x_{e,2},\dots,x_{e,T_e})
$$

每个时间步为：

$$
x_{e,t}=\phi(p_t,R_t,g_t,\Delta p_t,\Delta R_t,\Delta g_t)
\in\mathbb{R}^{20}
$$

| 特征组 | 维度 | 默认权重 |
|---|---:|---:|
| `eef_pos` | 3 | 1.0 |
| `eef_rot6d` | 6 | 0.7 |
| `gripper` | 1 | 1.2 |
| `delta_pos` | 3 | 0.5 |
| `delta_rot6d` | 6 | 0.3 |
| `delta_gripper` | 1 | 0.8 |

---

## 14. Grouped Huber 距离

Huber 损失：

$$
L_\delta(z)=
\begin{cases}
\frac{1}{2}z^2, & |z|\le\delta\\
\delta(|z|-\frac{1}{2}\delta), & |z|>\delta
\end{cases}
$$

设特征维度被划分为若干组：

$$
G=\{g_1,g_2,\dots,g_q\}
$$

第 \(g\) 组的组内欧氏距离：

$$
r_g(x,y)=\sqrt{\sum_{r\in g}(x_r-y_r)^2}
$$

Grouped Huber 距离：

$$
d_{\text{group}}(x,y)
=
\sum_{g\in G}w_g L_\delta(r_g(x,y))
{}+\sum_{r\notin \cup G}L_\delta(x_r-y_r)
$$

---

## 15. DTW 轨迹距离

给定两个 episode 序列：

$$
X=(x_1,x_2,\dots,x_T),\quad Y=(y_1,y_2,\dots,y_U)
$$

DTW 动态规划递推：

$$
D_{0,0}=0
$$

$$
D_{i,0}=D_{0,j}=+\infty
$$

$$
D_{i,j}
=
c(x_i,y_j)
{}+\min
\begin{cases}
D_{i-1,j}\\
D_{i,j-1}\\
D_{i-1,j-1}
\end{cases}
$$

归一化 DTW 距离：

$$
d_{\text{DTW}}(X,Y)
=
\frac{D_{T,U}}{L_{T,U}}
$$

窗口约束：

$$
w
=
\max\left(|T-U|,\left\lceil \rho\cdot\max(T,U)\right\rceil\right),
\quad
\lvert i-j\rvert\le w
$$

| 场景 | \(\rho\) |
|---|---:|
| 原型发现 | 0.15 |
| 语义传播对齐 | 0.20 |

---

## 16. K-medoids 原型发现

对所有候选 episode 计算两两距离：

$$
M_{a,b}=d_{\text{DTW}}(X_a,X_b)
$$

设所有 episode 集合为：

$$
E=\{e_1,e_2,\dots,e_N\}
$$

选出 \(K\) 个 medoid：

$$
M=\{m_1,m_2,\dots,m_K\}
$$

### 16.1 分配步骤

$$
\operatorname{cluster}(e)=\arg\min_{m\in M}d(e,m)
$$

含义：每个 episode 分配给最近的 medoid。

### 16.2 更新步骤

$$
m_k^*=\arg\min_{z\in C_k}\sum_{e\in C_k}d(e,z)
$$

含义：簇内总距离最小者成为新的 medoid。

### 16.3 收敛条件

$$
M^{(r+1)}=M^{(r)}
$$

含义：medoid 不再变化时停止迭代。

---

## 17. DBA Barycenter Refinement

对于同一簇内的多条序列：

$$
X^{(1)},X^{(2)},\dots,X^{(n)}
$$

DBA 的目标是求中心序列：

$$
B^*=
\arg\min_B\sum_{r=1}^{n}d_{\text{DTW}}(B,X^{(r)})
$$

对齐后，第 \(l\) 个 barycenter 时间步收集到的样本向量集合为：

$$
A_l=\{x^{(r)}_t\mid (l,t)\in \operatorname{DTWPath}(B,X^{(r)})\}
$$

更新公式：

$$
b_l
=
\frac{1}{|A_l|}\sum_{x\in A_l}x
$$

---

## 18. 轨迹异常分数

轨迹异常不建议只用单一 DTW 距离。可以把多种异常信号融合。

### 18.1 原型距离异常

$$
z_d=\operatorname{norm}(d_{\text{DTW}}(X_e,B_{c(e)}))
$$

含义：与簇中心越远越异常。

### 18.2 阶段顺序异常

$$
z_o=1-C_{\text{order}}
$$

含义：语义阶段顺序越乱越异常。

### 18.3 阶段时长异常

$$
z_\tau=
\operatorname{norm}\left(
\sum_l |\tau_{e,l}-\bar\tau_{c,l}|
\right)
$$

含义：各阶段时长越偏离原型越异常。

### 18.4 末端运动异常

$$
z_m=\operatorname{norm}(v_{\text{ee,max}}+a_{\text{ee,max}})
$$

含义：末端速度或加速度峰值越大越异常。

### 18.5 融合异常分数

$$
A_e
=
\alpha z_d+
\beta z_o+
\gamma z_\tau+
\eta z_m
$$

默认可令：

$$
\alpha=1.0,
\quad
\beta=1.0,
\quad
\gamma=0.5,
\quad
\eta=0.3
$$

异常判定：

$$
P_{\text{traj}}=
\mathbf{1}\left[A_e\le Q_{p}(\{A_r:r\in C(e)\})\right]
$$

其中 \(Q_p\) 是簇内异常分数的第 \(p\) 百分位数，例如 \(p=90\)。

---

## 19. 语义标注传播

source episode 上的人工标注 span：

$$
\alpha=(t_s^{\text{start}},t_s^{\text{end}},\ell)
$$

传播目标是：

$$
t_s\mapsto t_t
$$

从而得到：

$$
(t_s^{\text{start}},t_s^{\text{end}},\ell)
\mapsto
(t_t^{\text{start}},t_t^{\text{end}},\ell)
$$

DTW 对齐路径为：

$$
\mathcal{P}=\{(i_1,j_1),(i_2,j_2),\dots,(i_L,j_L)\}
$$

### 19.1 最近 source 索引

$$
i^*=\arg\min_i |\tau_i^s-t_s|
$$

含义：找到离 source 标注时间最近的采样点。

### 19.2 target 匹配集合

$$
J(i^*)=\{j\mid(i^*,j)\in\mathcal{P}\}
$$

含义：DTW 路径中与该 source 点对齐的 target 点。

### 19.3 target 时间

$$
t_t=\frac{1}{|J(i^*)|}\sum_{j\in J(i^*)}\tau_j^t
$$

含义：多个 target 点时取平均时间。

如果没有 DTW 对齐路径，则使用 duration 比例缩放：

$$
t_t=\frac{T_t}{T_s}t_s
$$

---

## 20. 任务成功度与训练价值

基础质量分只能说明数据格式和模态质量，不等于任务成功。任务成功度建议单独定义：

$$
C_{\text{task}}
=
\theta_1 C_{\text{phase}}
{}+
\theta_2 C_{\text{event}}
{}+
\theta_3 C_{\text{goal}}
{}+
\theta_4 C_{\text{human}}
$$

且：

$$
\sum_i\theta_i=1
$$

| 信号 | 含义 |
|---|---|
| 阶段完整度 | 是否经过合理的语义阶段，例如 approach、grasp、lift、place、release |
| 事件完整度 | 抓取、放置、释放等关键事件是否发生 |
| 目标达成度 | 终态是否满足任务目标 |
| 人工置信度 | 人工标注或人工复核信号 |

训练价值分数：

$$
U_e
=
P_e\cdot
\left(
\rho_q\frac{S_e}{100}
{}+
\rho_t C_{\text{task}}
{}+
\rho_p C_{\text{prototype}}
{}+
\rho_n C_{\text{novelty}}
\right)
$$

其中：

$$
\rho_q+
\rho_t+
\rho_p+
\rho_n=1
$$

| 信号 | 含义 |
|---|---|
| 基础质量 | \(S_e/100\)，由文件、时间、动作、视觉、深度等质量检查得到 |
| 任务成功置信度 | \(C_{\text{task}}\)，描述 episode 是否真的完成任务 |
| 原型一致性 | \(C_{\text{prototype}}\)，描述是否接近高质量原型 |
| 多样性 | \(C_{\text{novelty}}\)，避免只保留重复轨迹 |

---

## 21. Annotation / Prototype 置信度

### 21.1 标注信号

$$
A=\min\left(\frac{n_{\text{annotations}}}{4},1\right)
$$

含义：标注数量越多越可靠，最多归一化到 1。

### 21.2 质量信号

$$
Q=\operatorname{clip}\left(\frac{S_e}{100},0,1\right)
$$

含义：episode 质量分归一化。

### 21.3 原型信号

$$
P=\operatorname{clip}(1-d_{\text{bar}}(e),0,1)
$$

含义：越接近 barycenter 越可靠。

### 21.4 整体置信度

$$
C_{\text{overall}}=\frac{A+Q+P}{3}
$$

---

## 22. 数据集级统计

### 22.1 平均质量分

$$
S_D=\frac{1}{N}\sum_{e=1}^{N}S_e
$$

### 22.2 加权训练价值

$$
U_D=\frac{\sum_e\omega_e U_e}{\sum_e\omega_e}
$$

### 22.3 自动质检通过率

$$
R_{\text{pass}}=\frac{1}{N}\sum_e P_e
$$

### 22.4 拒绝率

$$
R_{\text{reject}}=
\frac{\#\{e:d_e=\text{reject}\}}{N}
$$

### 22.5 人工复核率

$$
R_{\text{review}}=
\frac{\#\{e:d_e=\text{review}\}}{N}
$$

### 22.6 高质量比例

$$
R_{\text{high}}=
\frac{\#\{e:U_e\ge0.85\}}{N}
$$

---

## 23. 人工复核闭环

自动质检最终需要通过人工复核数据校准误杀率和漏检率。

设人工标签为：

$$
y_e\in\{0,1\}
$$

自动准入结果为：

$$
\hat y_e=P_e
$$

### 23.1 混淆矩阵计数

$$
TP=\#\{e:\hat y_e=1\land y_e=1\}
$$

$$
FP=\#\{e:\hat y_e=1\land y_e=0\}
$$

$$
FN=\#\{e:\hat y_e=0\land y_e=1\}
$$

### 23.2 精确率与召回率

$$
Precision=\frac{TP}{TP+FP}
$$

$$
Recall=\frac{TP}{TP+FN}
$$

### 23.3 误杀率与漏检率

$$
R_{\text{kill}}=\frac{FN}{TP+FN}
$$

$$
R_{\text{leak}}=\frac{FP}{TP+FP}
$$

阈值调参目标可以写成：

$$
\min_{\Theta}
\left(
\lambda_{\text{leak}}R_{\text{leak}}
{}+
\lambda_{\text{kill}}R_{\text{kill}}
{}+
\lambda_{\text{review}}R_{\text{review}}
\right)
$$

其中 \(\Theta\) 是所有阈值和权重，\(\lambda\) 是业务成本权重。

---

## 24. 总结

### 24.1 检查项

$$
x_{e,k,j}\in\{0,1\}
$$

### 24.2 验证器

$$
S_{e,k}=100\cdot
\frac{\sum_j w_{k,j}x_{e,k,j}}{\sum_j w_{k,j}}
$$

通过条件：所有 `critical` / `major` 检查项通过。

### 24.3 质量类别

$$
S_{e,c}=100\cdot
\frac{\sum_{(k,j)\in c}w_{k,j}x_{e,k,j}}{\sum_{(k,j)\in c}w_{k,j}}
$$

### 24.4 Episode

$$
S_e=\sum_c\lambda_c S_{e,c}
$$

通过条件：

$$
P_e=1
$$

### 24.5 训练价值

$$
U_e=
P_e
\left(
\rho_q\frac{S_e}{100}
{}+
\rho_tC_{\text{task}}
{}+
\rho_pC_{\text{prototype}}
{}+
\rho_nC_{\text{novelty}}
\right)
$$

### 24.6 数据集

$$
S_D=\frac{1}{N}\sum_e S_e
$$

$$
R_{\text{pass}}=\frac{1}{N}\sum_eP_e
$$

核心逻辑：

$$
\boxed{
\text{检查项}\rightarrow\text{验证器}\rightarrow\text{质量类别}\rightarrow\text{episode 决策}\rightarrow\text{数据集价值}
}
$$
