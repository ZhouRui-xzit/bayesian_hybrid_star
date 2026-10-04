# NJL + RMF 混合星：贝叶斯推断与后验传播

## 项目目标

先在广义相对论（GR）下构造、标定冷致密物质 EOS，再将 EOS 联合后验样本传播到恒星结构和振荡观测量。EGB 耦合只作为第四步的外部扫描量，不进入本项目的贝叶斯待拟合参数。

项目按以下四步实施，不跳过基础求解器而直接开始大规模参数采样：

1. 实现三味 NJL + 矢量相互作用 + 有效袋常数，跑通 EOS → GR 恒星结构 → M-R、Lambda、f-mode。数值方法采用自行实现的 RK4 处理初值问题（IVP），以谱配置法处理边值问题（BVP）与频域本征值问题；当前不实现时间演化模块。
2. 实现 RMF 强子相，与 NJL 通过 Gibbs 两相构造连接，得到完整混合星 EOS，并复用第一步的恒星求解器。
3. 使用 GR 下的观测似然，对 RMF 参数、gV 和 B_eff 做贝叶斯推断，保存完整联合后验样本。
4. 传播后验，给出 GR 下的 f-mode 等预测，以及固定 EGB 耦合下的 M-R、f-mode 条件预测和相对 GR 的变化。

第一步的 NJL 单相恒星是物质与数值模块的基线。最终混合星的物理结论以第二步的完整 EOS 为基础。

## 项目设计

### src

`src` 下不设子文件夹，直接放若干 `.py` 模块。按 EOS、恒星观测量、贝叶斯三组分工；分组只用于说明，不额外建包或抽象层。模型函数不直接读写文件，不在模型内部设置扫描范围，不依赖绘图脚本。

| 分工 | 文件 | 职责 |
| --- | --- | --- |
| 初始化 | `src/__init__.py` | 在创建 JAX 数组之前启用 `jax_enable_x64`；不启动计算 |
| 公共 | `src/common.py` | 物理常数、单位转换、少量结果数据结构、配置与 CSV 读写 |
| 公共 | `src/numerics.py` | Optimistix 求根、延拓、自行实现的 RK4、BVP 谱配置与数值诊断 |
| EOS | `src/njl.py` | NJL 热力学势、矢量平均场、gap 与相内热力学量 |
| EOS | `src/rmf.py` | RMF 场方程与非电中性强子相 |
| EOS | `src/leptons.py` | 电子、缪子及阈值处理 |
| EOS | `src/gibbs.py` | Gibbs 共存方程、体积分数与混合相边界 |
| EOS | `src/eos.py` | 各相装配、壳层连接、EOS 表、插值、声速与热力学诊断 |
| 观测量 | `src/structure.py` | GR/EGB 背景方程、中心压力扫描、M-R、质量定位与核心结构 |
| 观测量 | `src/tidal.py` | 潮汐扰动、Love 数与 Lambda |
| 观测量 | `src/fmode.py` | 极向 l=2 f-mode、频域谱本征值求解与本征函数 |
| 贝叶斯 | `src/likelihood.py` | 参数先验、物理可行域、观测似然与相关性处理 |
| 贝叶斯 | `src/bayes.py` | NumPyro 建模、采样、预热、断点与诊断 |
| 贝叶斯 | `src/posterior.py` | 样本传播、配对差值、可信带与可用样本权重 |

先实现这些文件中当前阶段需要的函数，不一次生成空文件。代理模型确有需要时再增加一个平铺的 `src/surrogate.py`。`structure.py` 内以明确的函数或配置区分 GR/EGB，不再建立 gravity 子包。

统一从项目根目录运行模块，导入写成 `from src.njl import ...`；`src` 内部可用相对导入。不修改 `sys.path`，不依赖机器级 `PYTHONPATH`，也不把模块命名为 `jax.py`、`numpy.py` 等第三方库名称。`scripts/__init__.py` 保持为空，使入口脚本可以用 `python -m` 运行。

模型主要使用 JAX 的 64 位浮点和自动微分。先构造 `Omega(orders, T, muB, muQ, params)`，再用前向 AD 得到场方程和 Jacobian，以 Optimistix 的 Newton 求解。平衡态导数通过隐函数关系求解。可使用 Equinox 表达参数 PyTree、Lineax 求线性系统；只引入实际需要的结构，不强迫每个函数变成类。

### 数值与贝叶斯依赖

| 库 | 本项目用途 |
| --- | --- |
| JAX | 64 位计算、AD、JIT、向量化和随机数 |
| Optimistix | gap、RMF/Gibbs 非线性方程，以及振荡边界残差求根 |
| 自行实现的 RK4 | 用 JAX 实现经典四阶 Runge–Kutta，求解 TOV、潮汐等径向初值问题 |
| 自行实现的谱配置 | Chebyshev 微分矩阵、区域映射、BVP/本征值残差与边界条件 |
| Equinox / Lineax | JAX PyTree 模型与线性系统求解 |
| NumPyro | 默认且唯一的贝叶斯建模与采样接口，调用项目自己的先验和观测似然 |
| ArviZ | 多链诊断、ESS、R-hat 与后验可视化 |
| NumPy / SciPy | 文件边界、独立基准或暂不能 JAX 化的数值回退 |

NumPyro 基于 JAX，不是 Optimistix 自带的贝叶斯模块。Optimistix 解物理方程，NumPyro 采样参数后验，二者通过 JAX 前向函数衔接。默认依赖不加入 BlackJAX 或 Diffrax。

JIT 区域不调用 pandas、文件操作或依赖 Python 异常的流程，不对 tracer 执行 `float()`、`np.asarray()` 或数据相关的 Python 分支。随机数 key 显式传递；多链和并行计算要拆分 key。物理分支、不等长 EOS 网格和相变边界需要设计固定形状/掩码等处理，不能为了 JIT 静默丢掉物理分支。

### uv 环境

项目根目录使用随附的 `pyproject.toml`。这是直接运行脚本的科研项目，设定 `[tool.uv] package = false`，不做发行包构建，也不自动安装 `src` 为第三方包。默认安装 CPU 版 JAX；GPU 安装按实际平台和 JAX 官方说明配置，不在基线中预设 CUDA 依赖。

采用标准 CPython 3.14，TOML 设置 `requires-python = ">=3.14,<3.15"`，Ruff 目标同步为 `py314`。JAX 官方支持策略已覆盖 Python 3.14，jaxlib 已发布相应二进制 wheel，NumPyro 也声明支持 3.14。完整依赖组合仍需以 uv 解析、安装和导入检查验证；不将发行元数据等同于项目已经测试通过。首次建立环境：

```bash
uv sync --python 3.14
uv run python -c "import jax, optimistix, numpyro; print(jax.devices())"
```

NumPyro 随默认环境安装；`uv sync --extra notebook` 可安装交互式 notebook 支持。pytest 和 Ruff 放在默认开发依赖组。

依赖声明不冒充已验证的精确版本集合。第一次成功解析与安装后，提交生成的 `uv.lock`，并记录 Python 小版本和后端；复现使用 `uv sync --locked`。本次提供 TOML 与设计文档，若尚未生成 lock 或完成依赖安装，交付说明必须明确。

### scripts

具体运行流程放入 `scripts`，脚本只读取配置、调用 `src`、保存数据和绘图。默认数据输出 CSV，默认图片 SVG。

| 脚本 | 流程 |
| --- | --- |
| `scripts/step01_njl.py` | NJL EOS、单相恒星、Lambda、f-mode |
| `scripts/step02_gibbs.py` | RMF + NJL Gibbs EOS、混合星序列与核心结构 |
| `scripts/step03_bayes.py` | GR 下的贝叶斯推断与采样诊断 |
| `scripts/step04_posterior.py` | GR 后验预测、固定 EGB 耦合的条件预测 |
| `scripts/plot_results.py` | 只从已保存数据绘图，避免为了重画图重新计算 |

每个主脚本支持 `--config`、`--output`、`--seed`；长任务支持恢复。入口实现完成后，从根目录运行，例如：

```bash
uv run python -m scripts.step01_njl --config configs/step01.yaml
uv run python -m scripts.step02_gibbs --config configs/step02.yaml
uv run python -m scripts.step03_bayes --config configs/step03.yaml
uv run python -m scripts.step04_posterior --config configs/step04.yaml
```

项目根目录包含 `pyproject.toml`、依赖锁文件、`README.md` 和本设计文档。

### configs

`configs` 保存可复现配置，先平铺为 `base.yaml`、`step01.yaml` 至 `step04.yaml`；模型、数值、观测、推断与传播设置作为配置字段分组，不提前建立多层目录。

- 参数文件保存 NJL/RMF 的明确来源、拉格朗日量约定、正规化、数值和单位。
- 数值文件保存算法、容差、节点数、积分区间、根选择与失败重试策略。
- 观测文件保存原始数据来源、对应的引力假设、原始分析先验、是否和其他数据共享事件。
- 后验传播文件保存固定的 `alpha_gb_km2` 网格，不为其设置贝叶斯先验。

### data

所有运行输出进入 `data/runs/<step>/<run_id>/`，不同运行不互相覆盖。每次保存解析后的完整配置、随机种子、代码版本、依赖版本、数据来源与哈希、求解状态和运行摘要。

`data/raw/` 保存原始观测或基准数据，原则上只读。每次运行至少有 `config_resolved.yaml`、`metadata.json`、`summary.json`、`failures.csv`。失败记录为空时也保存表头。大规模剖面和后验可以补充 NPZ/HDF5 等格式，但保留主要 CSV 汇总。

### docs 与 tests

`docs` 记录方程、约定、基准和实施决策；`tests` 只针对会影响科学结论的事项，例如单位、热力学恒等式、边界条件、已知基准和数值收敛。不要给每个低层函数机械堆砌测试。

## Agent 实施规则与公共接口

1. 先检查已有实现，复用正确的模块；每次只推进一个可验证的阶段，不预先建立大量空模块。
2. 物理公式、模型参数、运行配置分离。不能在似然中临时修改 EOS 或在绘图中偷偷筛样本。
3. 不虚构观测数据、拟合参数、基准值、文献结论或求解成功状态。尚未实现的物理能力显式返回 `unsupported`。
4. 每一步结束时更新 `docs/progress.md`：已完成接口、验证结果、未解决问题和下一项具体工作。
5. 模型选择未指定时，先在 `docs/decisions.md` 给出有来源的基线选择；只有会改变研究目标的歧义才向用户澄清。
6. 本文描述目标架构，不代表已经完成代码、运行或验证。

公共接口采用以下语义，具体类名可随现有代码调整：

| 接口 | 输入与输出约定 |
| --- | --- |
| `solve_phase(muB, muQ, params, initial)` | 返回场解、P、epsilon、nB、nQ、组分、残差、分支和状态；返回的强相不包含轻子 |
| `build_eos(model, params, grid, config)` | 返回冷电中性 EOS、相态边界和热力学诊断 |
| `eos.epsilon_of_pressure(P)` | 只在声明的压力区间内插值，越界不静默外推 |
| `eos.sound_speed_sq(P)` | 返回平衡 EOS 的 dP/d epsilon，并说明求导方法 |
| `solve_star(eos, pc, gravity, config)` | 返回 M、R、背景剖面、表面状态和分支标签 |
| `solve_tidal(star, eos, config)` | 返回 k2、Lambda、匹配残差与方法标签 |
| `solve_fmode(star, eos, config)` | 返回频率、本征函数、方法与收敛信息；仅完整辐射计算返回阻尼时间 |
| `log_likelihood(theta, data, config)` | 仅调用 GR 前向模型；记录物理拒绝与数值失败的不同原因 |

每个记录带 `eos_id`、`sample_id`、`branch_id` 和 `status`。多稳定分支时，同一质量可对应多个构型，不把 `R(M)` 强行视为单值函数。

## 第一步：NJL + gV + B_eff 与 GR 恒星求解

### 1.1 物理约定

基线为 T=0、三味 u/d/s、无中微子俘获、beta 平衡的均匀物质，加入电子和必要时的缪子。暂不默认加入颜色超导、磁场或旋转。

定义 `gV = G_V / G_S` 为无量纲扫描参数，`G_V` 为有量纲耦合。明确矢量道采用味单态还是逐味形式，明确有效化学势平移的系数；不同约定的 gV 数值不能直接互换。

采用守恒荷基底：

\[
\mu_u=\mu_B/3+2\mu_Q/3,\quad
\mu_d=\mu_s=\mu_B/3-\mu_Q/3,\quad
\mu_e=\mu_\mu=-\mu_Q.
\]

物理化学势与矢量相互作用修正后的有效化学势分开命名。单相基线施加总电中性；第二步 Gibbs 必须能调用不预先施加电中性的同一个相模块。

袋常数统一定义为

\[
\Omega_Q=\Omega_{\rm NJL}^{\rm normalized}+B_{\rm eff},\qquad
P_Q=P_{\rm NJL}^{\rm normalized}-B_{\rm eff},\qquad
\epsilon_Q=\epsilon_{\rm NJL}^{\rm normalized}+B_{\rm eff}.
\]

这里 B_eff 为密度无关常数，配置与输出使用 MeV/fm^3，进入自然单位计算时显式转换。真空减除固定一次，不在每个密度点重新归零压力。B_eff 的允许正负范围在先验中声明。

### 1.2 Gap 与热力学

- 构造同一套热力学势和场方程，使用 AD 生成 Jacobian，配合 Newton、多初值和连续延拓寻找候选解。
- 矢量辅助场一般是驻点问题，不能对所有场统一做无约束最小化；按相应场方程消去或联合求根后比较物理解的热力学势。
- 多解区保留分支，筛除不合物理的解，在适当的化学势条件下比较热力学势；不能把 Newton 首次收敛的根自动视为稳定相。
- 计算 `P, epsilon, nB, nQ, muB, muQ, M_u, M_d, M_s, cs2`，保存 gap 残差和选根依据。
- 对场方程 `F(x, mu)=0`，平滑分支上的响应满足 `dx/dmu = -(dF/dx)^(-1) dF/dmu`。通过线性求解实现，不显式求逆；在相变和奇异点附近不跨分支 AD。
- 零温热力学需满足 `epsilon = -P + muB*nB + muQ*nQ`。总电中性态约化为 `epsilon + P = muB*nB`。

### 1.3 EOS 与 M-R

生成支持恒星求解的 EOS 表，保留相态边界、非光滑点和允许范围。插值需保持必要的单调性与导数一致性；不能为获得平滑曲线而抹掉物理相变。

首选自行实现的经典固定步长 RK4 求解 GR TOV 方程，从中心级数展开启动。通过步长减半检查四阶收敛与全局误差，必要时再实现基于 step-doubling 的步长控制；不能把经典 RK4 称为嵌入式 RK4/5。

表面以 P=0 或明确的壳层终止条件定位。RK4 的中间级若越过 EOS 允许范围，要缩短/重试步长；不把负压传给 EOS 插值，也不把第一个负压网格点直接当作恒星半径。采用受控缩步、区间细分或合法稠密重构定位表面。扫描中心压力得到 M-R 序列，在极值和目标质量附近加密。

纯 NJL EOS 只有在允许范围内存在合适的零压端点时才构造相应表面。有限表面密度按自束缚表面处理，但不由此宣称夸克物质绝对稳定；不存在物理表面时返回原因，不人为补一段 EOS。壳层是明确的附加模型，不默认给单相夸克星套上普通中子星壳层。

### 1.4 Lambda 与 f-mode

Lambda 模块求解静态 l=2 潮汐扰动，并计算

\[
C=GM/(Rc^2),\qquad \Lambda=\frac{2}{3}k_2C^{-5}.
\]

对有限表面密度和内部密度跃迁使用正确的扰动匹配条件，不能直接套用表面密度为零的公式。

f-mode 模块计算非旋转星的极向 l=2 流体基频。第一步可用 Cowling 近似跑通流程，但输出必须标记 `cowling`；正式基线优先实现含度规扰动的 GR 计算。不能把经验普适关系的输出标记为“数值求得的 f-mode”。

- 默认只实现谱方法：采用 Chebyshev 多区域配置，将内部、表面和必要的外部区域分别离散，再求本征值或边界残差的根。当前不并行开发 RK4 射击后端。
- 中心满足正则性，表面满足拉格朗日压力扰动 `Delta P=0`。完整 GR 辐射模式还需外部匹配和无穷远出射条件。
- 统一时间约定，区分 `omega` 与 `f=Re(omega)/(2*pi)`，以无穷远观察者频率输出 kHz。阻尼时间只在相应复频率问题求解后输出；Cowling 不提供引力波阻尼时间。
- 跟踪本征函数和频率连续性识别 f-mode，不能只取数值谱中最小的正本征值。
- 初始基线使用冷平衡的 barotropic 扰动闭合并记录该近似；背景平衡声速不自动等于冻结组分条件下的绝热响应。更精细的组分响应作为独立扩展。

### BVP 与频域本征值的谱方法

统一在 `src/numerics.py` 实现 Chebyshev 节点、微分矩阵与坐标映射；具体方程和边界残差保留在 `src/structure.py`、`src/tidal.py`、`src/fmode.py`。不引入时间推进、method-of-lines 或时空演化依赖。

- 初值问题继续交给 RK4；不因为都属于径向 ODE 就统一按 BVP 求解。若指定 EGB 理论的静态背景确实需要两端边界条件，则直接使用谱 BVP。
- 对非线性 BVP，把节点场值和必要的未知参数作为未知量，用谱微分矩阵构造内部残差，并以边界条件替换相应方程，最后交给 Optimistix。未知量与独立方程数必须匹配。
- 对 f-mode，采用频域形式，时间导数被频率参数代替。根据具体方程构造线性/广义本征值问题，或含未知频率的非线性谱残差；不要假定所有完整 GR/EGB 模式都能写成简单的线性本征值问题。
- 齐次扰动方程必须设置合适的幅度归一化，排除所有本征函数为零的伪解。复频率求根可拆为实部与虚部方程，符号与阻尼约定保持一致。
- 处理中心正则性、表面条件、区域间匹配和外部出射条件；非光滑 EOS 边界采用多区域。无限外部区域采用有依据的映射或渐近匹配，不能把有限外边界的零值条件当作出射条件。
- 用增加谱节点数、调整区域划分和检查方程/边界残差验证收敛；非光滑解不强求单区域指数收敛。跟踪本征函数排除伪模态。
- 本阶段以谱节点收敛、边界残差与已知基准验证 BVP，不再要求与自编 RK4 射击作双后端比较。

### 1.5 阶段交付

输出 `eos_njl.csv`、`sequence_gr.csv`、`tidal_gr.csv`、`fmode_gr.csv` 和对应 SVG。记录 gV、B_eff、算法和误差。以单位一致性、热力学恒等式、GR 基准及步长/谱节点收敛为通过条件；数值误差预算在配置中预先给出，且低于目标观测误差。

## 第二步：RMF + NJL 的 Gibbs 两相模型

### 2.1 RMF 与非电中性相接口

先采用一套有来源的 RMF 参数作为基线，将模型跑通后再开放连续 RMF 参数用于第三步。`src/rmf.py` 必须在给定 `(muB, muQ)` 下返回非电中性相：例如 `mu_n=muB`、`mu_p=muB+muQ`。若采用密度依赖耦合，场方程、压力和化学势必须包含所需的重排项。

RMF 和 NJL 的真空、单位与电荷定义统一。两相接口均不包含轻子，轻子由混合相层统一加入一次。

### 2.2 Gibbs 方程

采用忽略表面能和库仑有限尺寸修正的 bulk Gibbs 构造。令 chi_Q 为夸克相体积分数，H、Q 分别表示强子与夸克强相，l 为轻子：

\[
P_H(\mu_B,\mu_Q)=P_Q(\mu_B,\mu_Q),
\]

\[
(1-\chi_Q)n_Q^H+\chi_Q n_Q^Q+n_Q^l=0,\qquad 0\leq\chi_Q\leq1.
\]

两相共享 muB、muQ 和 T=0。这里 `n_Q` 是带符号的电荷数密度，避免与“夸克数密度”混淆。混合量为

\[
n_B=(1-\chi_Q)n_B^H+\chi_Q n_B^Q,
\]

\[
\epsilon=(1-\chi_Q)\epsilon_H+\chi_Q\epsilon_Q+\epsilon_l,
\qquad P=P_H+P_l=P_Q+P_l.
\]

给定 muB 时求解 `(muQ, chi_Q)`；也可给定总 nB 联合求解 `(muB, muQ, chi_Q)`，但全项目保留统一的输出约定。用延拓追踪共存分支，在 chi_Q=0、1 定位起止点。

纯强子和纯夸克段各自满足总电中性；混合段只施加全局电中性。禁止先对两相分别电中性化，再把两张单变量 EOS 表拼成“Gibbs”。混合相一般不是恒压平台，也不能用随意平滑插值代替共存方程。

对于多个可行共存分支，检查热力学稳定性并选择适当的平衡包络。没有物理共存解时，记录“无转变”或具体模型失败，不强行制造混合相。

### 2.3 完整 EOS 与恒星结构

将壳层、纯强子段、混合相段、纯夸克段连接成完整冷 EOS。检查压强、化学势匹配及总热力学关系。沿平衡、电中性 Gibbs 路径计算声速，不能把两相声速直接按 chi_Q 线性加权。

复用第一步的 M-R、Lambda、f-mode 求解器；谱区域按非光滑边界划分。记录混合相外边界半径、纯夸克核半径、中心 chi_Q，以及明确采用体积、引力质量还是重子数定义的夸克含量。

稳定性判据与构型分支分别记录。转折点可用于常规 GR 序列的初步标记；不把局域绝热指数大于 4/3 作为相对论混合星或 EGB 恒星稳定性的充分证明。涉及多分支和界面转换时间尺度时，需增加相应稳定性分析。

### 2.4 阶段交付

输出 `eos_hybrid.csv`、`phase_boundaries.csv`、`composition.csv`、混合星 M-R/Lambda/f-mode 及 SVG。验证共存压强、化学平衡、全局电中性、端点极限与数值收敛。第三步只调用已经通过这些检查的统一前向接口。

## 第三步：GR 下的贝叶斯推断

### 3.1 参数与先验

待拟合参数为 `theta = {theta_RMF, gV, B_eff}`。NJL 真空标定参数在基线中固定；需要放开时另行声明。RMF 优先选择有物理解释且受核物质信息约束的少量参数，并给出其到耦合常数的映射，不一次开放所有耦合。

采用物理先验和独立的核物质约束控制低密度行为。明确哪些信息作为先验、哪些作为似然，不重复使用。检查热力学稳定性、适用密度范围内的因果性及恒星可用分支，不默认将 `cs2 <= 1/3` 设为全密度硬约束。

`alpha_gb_km2` 不属于 theta，不在此步采样。不把 f-mode 的理论预测区间当成独立观测拟合目标。

### 3.2 观测似然

纳入有明确来源的脉冲星质量、NICER 联合 M-R 信息和引力波潮汐信息。给定 EOS 时对每颗恒星的质量或中心参数进行边缘化，不能只判断模型曲线是否穿过误差椭圆。

- 使用公开后验重建似然时，处理原分析先验及变量变换；不能默认后验就是似然。
- 同一恒星、同一事件的不同结果检查数据重叠与相关性；例如不要重复计入已包含在 NICER 分析中的质量信息。
- 引力波优先使用质量与潮汐的联合信息，不把同一事件的边缘 Lambda、f-mode 等结果独立相乘。
- 群体质量分布、双星配对和需要的选择效应在似然说明中明确。多恒星分支的权重规则独立声明。
- 物理不允许样本可返回负无穷；数值失败先按既定策略重试并单独统计，不能静默把所有求解失败都当成物理排除。

目标分布为

\[
p(\theta\mid D,\mathrm{GR})\propto
\mathcal L(D\mid\theta,\mathrm{GR})\,\pi(\theta).
\]

### 3.3 采样与加速

先用小规模直接求解验证似然，默认在 `src/bayes.py` 使用 NumPyro。用 `numpyro.sample` 表达先验、`numpyro.factor` 接入自定义对数似然，或使用语义等价的势函数接口；同一先验只计入一次。若使用势函数，应明确它是负对数目标密度，并处理参数变换及 Jacobian。

- 相变切换、选根、硬约束与失效分支可能让完整后验不光滑，不能因为使用 JAX 或单个 gap 方程可微就默认 NUTS 适用。
- 初始可用 NumPyro 的无梯度 `SA`（Sample Adaptive）建立小规模基线，检查多链混合与 ESS；这不是对其在高维或多峰后验中效率的保证。
- 只有在端到端前向映射及参数变换梯度经检查、分支处理明确后才采用 HMC/NUTS。保留直接前向模型与必要的梯度检查。
- 自行实现的 RK4 用 JAX 运算；固定形状的扫描可以用 `lax.scan`。提前终止、表面定位、动态迭代和物理解切换的导数需专门处理，不能假设自动微分会自动给出正确的边界/分支响应。
- 若前向函数包含 SciPy/NumPy 回退，不直接放入 JIT 或 NumPyro 采样内核。即使 SA 不需要梯度，仍须满足 NumPyro/JAX 的追踪和执行要求；先解决兼容性，再运行正式采样。
- 对多峰或跨分支混合不良，先报告采样失效和诊断，再选择明确验证的策略。默认不并列维护 BlackJAX、nested sampling 或其他采样框架。

保存链号、迭代号、log_prior、log_likelihood 与诊断，支持恢复。预热样本单独保存或丢弃；普通 MCMC 后验样本采用等权，不能再按 likelihood 给样本加权。

若前向模型耗时过高，再增加代理模型。训练数据来自数值求解，覆盖相变边界与分支；按 EOS 参数组划分训练/验证集合，避免同一 EOS 序列泄漏。独立检查其在后验高概率区域的误差，超出训练域返回直接求解或显式拒绝；代理误差不可忽略时纳入似然。神经网络不承担物理方程验证。

### 3.4 阶段交付

输出 `posterior_samples.csv`、`posterior_summary.csv`、采样诊断及 GR 下的 EOS/M-R/Lambda 可信带。样本至少保留 `sample_id, theta_*, weight, log_prior, log_likelihood, status`。

报告先验敏感性、参数相关性和收敛性。MAP 只作代表点展示，不替代联合样本。禁止分别取各参数的 95% 上下界再拼成“95% EOS”。

## 第四步：GR 后验预测与固定 EGB 耦合的后验传播

### 4.1 统计定位

输入第三步的完整联合样本及其权重。GR 中计算未参与拟合的 f-mode 等量，属于 GR 后验预测。

EGB 中选择外部给定的 alpha_gb_km2 网格，对每个 EOS 样本重新求解恒星背景和所支持的扰动量。记 Y 为固定质量处的预测量，其条件分布为

\[
q_\alpha(Y\mid D_{\rm GR})=
\int d\theta\,p(\theta\mid D,\mathrm{GR})
\delta\!\left[Y-Y_{\rm EGB}(\theta,\alpha)\right].
\]

报告名称采用“GR 标定 EOS 在固定 EGB 耦合下的条件预测”。这不是 EGB 下重新拟合出的后验，也不构成 alpha 的观测约束。不同 alpha 的结果分开展示，不在未定义权重时混成一个可信区间。

沿用原样本权重，不用同一批观测再次筛选或重新加权。由此遗漏的“GR 下不被允许、EGB 下可能被允许”的 EOS 属于本研究范围的明确边界。

### 4.2 EGB 物理实现

实现前先在 `docs/equations_egb.md` 固定作用量、标量约定、耦合归一化、单位、球对称解分支、边界条件及文献来源。不能仅凭“EGB”名称混用不同理论的结构和扰动方程。

假设物质最小耦合，复用同一物质 EOS，改变引力背景；若所选理论需要额外物质耦合，则必须修改设计并明确说明。

- 验证 alpha→0 恢复 GR，分别检查背景和已实现的扰动模块。
- EGB 下的 f-mode 必须使用同一作用量对应的扰动方程，处理可能的标量与度规扰动耦合。
- 若只完成 EGB Cowling 近似，结果标记为近似，并与 GR Cowling 对照；不将近似差异全部解释为引力效应。
- 未完成 EGB 潮汐方程时，Lambda 只给出 GR 结果；EGB Lambda 返回 `unsupported`，不得套用 GR 公式冒充完整结果。
- 记录解存在性与已检查的稳定性条件。没有完成完整扰动稳定性分析时，明确标为待验证，不能因得到 M-R 曲线就声称构型完全稳定。

### 4.3 输出与比较

给出固定 alpha 下的 M-R、最大质量、核心结构、f-mode 及条件可信带。在相同 EOS 样本、相同质量和可匹配分支上计算

\[
\Delta R(M;\theta,\alpha)=R_{\rm EGB}-R_{\rm GR},\qquad
\Delta f_f(M;\theta,\alpha)=f_{f,\rm EGB}-f_{f,\rm GR}.
\]

配对比较用于隔离固定 EOS 时的引力响应；另报完整的 EOS 不确定性带。若某个目标质量在某种引力下没有可用构型，记录其后验权重占比。对剩余样本归一化得到的区间必须标记为条件于构型存在；多分支不强行配对。

可信带默认为逐质量点的加权 68% 和 95% 区间，不称为覆盖整条曲线的同时可信带。多峰分布同时展示分支或分布形状，避免只画一条中位数曲线。

重点讨论：EGB 变化相对于 EOS 不确定性的大小、变化与 gV/B_eff 的相关性、夸克核出现附近的响应，以及不同质量范围的敏感性。可信带分离是情景差异，不自动等于观测可探测性。

输出 `gr_predictions.csv`、`egb_predictions.csv`、`paired_differences.csv`、`credible_bands.csv`、`availability.csv` 和 SVG，保留原始 sample_id、权重、alpha、分支及方法标签。

## 完成标准

- 四个主脚本可按顺序用小配置跑通，所有结果能从保存的配置与代码版本复现。
- 相模型、Gibbs、EOS、引力背景、扰动、统计和绘图模块解耦。
- 参数与单位无歧义，失败不会悄悄变成成功，近似结果和未实现能力明确标记。
- GR 贝叶斯推断不含 alpha；第四步保存条件预测的统计含义与适用边界。
- 数值基准与收敛验证通过后再运行大采样；IVP 使用 RK4、BVP 使用谱方法，不要求实现重复后端。

## 起始参考

以下文献用于定位方法；实现具体方程时需核对原文的作用量、正规化、边界条件与符号，不能从摘要复制公式。

- [Hybrid Stars in the Framework of different NJL Models](https://arxiv.org/abs/1612.09485)：RMF/NJL 与 Gibbs/Maxwell 构造。
- [Tidal Love Numbers of Neutron and Self-Bound Quark Stars](https://arxiv.org/abs/1004.5098)：潮汐求解与有限表面密度处理。
- [Bayesian analysis of the properties of hybrid stars with the NJL model](https://arxiv.org/abs/2112.09595)：NJL 矢量道、有效袋常数与贝叶斯框架；其相变构造不能直接当作本项目的 Gibbs 实现。
- [Bayesian investigation of the neutron star equation of state vs gravity degeneracy](https://doi.org/10.1103/PhysRevD.109.064048)：EOS/引力退化的方法参考；本项目采用 GR 标定后固定耦合传播，不照搬其联合推断目标。

## 软件参考

- [Optimistix](https://docs.kidger.site/optimistix/)：非线性求解与隐式微分。
- [NumPyro MCMC](https://num.pyro.ai/en/latest/mcmc.html)：默认推断接口，包含 SA、HMC/NUTS；核对安装版本的 API。
- [uv project configuration](https://docs.astral.sh/uv/concepts/projects/config/)：非打包项目与环境管理。

- [JAX Python version support](https://docs.jax.dev/en/latest/deprecation.html)：Python 3.14 支持策略。
- [jaxlib 发布文件](https://pypi.org/project/jaxlib/)与 [NumPyro 元数据](https://pypi.org/project/numpyro/)：运行环境兼容性依据。
