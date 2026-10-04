# 零温 NJL EOS 示例

模型位于 `NJL_T0.py`，验收入口是 `check_njl_eos.py`。从项目根目录运行：

```bash
uv sync --locked
uv run python -m example.check_njl_eos --find-critical --benchmark
```

默认 `Rv = 0, 0.25, 0.5, 1`，`B_eff = 0 MeV/fm³`。
按照最终确认的扫描方向，密度网格是 `1e-8, 0.01, 0.02, ..., 6`，共 601 点。
先求手征破缺真空和稀薄物质初解，然后以每一点的解作为下一点初值；
**不进行多初值、多解扫描或全局相态择优**。

修改参数的例子：

```bash
uv run python -m example.check_njl_eos --rv 0.0 --b-eff 0.0 --max-rho 6 --step 0.01 --benchmark
uv run python -m example.check_njl_eos --rv 0 0.5 1 --b-eff 60 --max-rho 6 --step 0.01
uv run python -m example.check_njl_eos --csv-mode full
uv run python -m example.check_njl_eos --csv-mode compact --find-critical
```

默认目录为 `data/example__NJL0T/`，每组参数只输出一对 CSV/SVG：
例如 `eos_Rv0.5_Beff0.csv`、`eos_Rv0.5_Beff0.svg`。
同一组参数重复运行覆盖对应的两个文件；不再创建时间戳子目录、JSON/YAML、PNG 或额外 TOV 表。
运行诊断、临界耦合和计时打印到终端。`--output` 可指定其他输出目录。
`--find-critical` 会额外把临界 Rv 和推荐的临界值以上 Rv 分别输出对应的 EOS 表和图。

## 参考与物理约定

参考用户的私有仓库：

- [pnjl_vec.jl](https://github.com/ZhouRui-xzit/PNJLs/blob/08a32a6660ee1b2e0d9a4bf04236071bbc9734cd/src/axion_red_vec/pnjl_vec.jl)
- [constants.jl](https://github.com/ZhouRui-xzit/PNJLs/blob/08a32a6660ee1b2e0d9a4bf04236071bbc9734cd/src/axion_red_vec/constants.jl)

保留原代码的正凝聚约定 `σf = −〈q̄f qf〉`、三味标量项和逐味矢量项。
取 `T=0`，去掉 axion/η 和 Polyakov 场；没有对有限温指数/对数公式直接代入零温。
不移植参考文件中与本例无关的有限温涨落和重复求导代码。

| 参数 | 数值 |
| --- | --- |
| Λ | 630 MeV |
| Gs Λ² | 1.781 |
| K Λ⁵ | 9.29 |
| mu, md, ms | 5.5, 5.5, 135.7 MeV |
| me, mμ | 0.511, 105.658 MeV |
| ℏc | 197.33 MeV fm，与参考一致 |
| n0 | 0.16 fm⁻³，重子数密度 |

在自然单位下：

\[
M_i=m_i+4G_S\sigma_i+2K\sigma_j\sigma_k,\qquad
\sigma_i=\frac{3M_i}{\pi^2}\int_{k_{Fi}}^\Lambda
\frac{p^2\,dp}{\sqrt{p^2+M_i^2}}.
\]

\[
n_i=\frac{k_{Fi}^3}{\pi^2},\qquad
\mu_i=\sqrt{k_{Fi}^2+M_i^2}+4G_Vn_i,\qquad G_V=R_vG_S.
\]

这是逐味矢量相互作用，不能把 `Rv` 数值直接等同于采用总密度平方、
不同耦合归一化或 `2Gv n` 化学势修正的文献。

约束为 `nB=(nu+nd+ns)/3`、`μd=μs=μu+μe`、`μμ=μe` 和
`2nu/3−nd/3−ns/3−ne−nμ=0`。轻子为自旋简并度 2 的自由费米气体，
`μe≤ml` 时该轻子密度、压力和能量严格为零。s 海未出现时设置 `ns=0`，
并检查 `μd≤Ms`；超过阈值后在同一延拓过程中打开 s 海。

设 `χ=2Gs Σσf²+4K σuσdσs`，能量由

\[
\epsilon=\chi-\frac{3}{\pi^2}\sum_f\int_0^\Lambda p^2E_f\,dp
+\frac{3}{\pi^2}\sum_f\int_0^{k_{Ff}}p^2E_f\,dp
+2G_V\sum_fn_f^2+\epsilon_e+\epsilon_\mu-\epsilon_{\rm vac}+B_{\rm eff}
\]

计算，压力单独计算，并检查 `ε+P=μB nB+μQ nQ`。
真空只减除一次；`B_eff` 是额外的、密度无关的袋常数：
`P=Pnormalized−B_eff`，`ε=εnormalized+B_eff`。它不改变 gap 解或组分。

公开单位为 MeV、fm⁻³、MeV/fm³。内核以 Λ 为尺度无量纲化，输出时显式转换。
当前示例只接受所有夸克费米动量小于 Λ 的点；超过这一声明范围会报错。

## 性能与数值精度

- 模型启用 JAX float64，gap Jacobian 由自动微分得到，使用 Optimistix Newton。
- 固定长度 7 维状态，s 海阈值由 `lax.cond` 切换；密度延拓由 JIT 编译的 `lax.scan` 执行。
- 热力学量和隐函数响应用 `vmap` 批量计算。声速由 `dx/dnB` 的线性方程得到，
  不通过对粗密度表直接差分产生。
- 费米积分使用解析零温表达式；小费米动量使用级数，避免相近数相减。
  真空能差先代数消去线性项，剩余平滑差值使用固定 96 点 Gauss–Legendre 积分。
- 不隐藏求根失败，不回退到其他初值或 SciPy gap 求解器；失败信息包含首个失败密度。
- SciPy 只用于表外的 PCHIP 插值、有限密度零压端点定位，以及测试中的独立积分。
- `--benchmark` 用 `jax.block_until_ready` 测量热启动扫描内核；完整计算时间另列，
  不将异步派发时间冒充计算耗时。

## 输出与 TOV 接口

默认精简模式 `--csv-mode compact`，每个 CSV 仅保留 8 列：

`Rv, B_eff_MeV_fm3, rho_over_rho0, nB_fm3, P_MeV_fm3, epsilon_MeV_fm3, muB_MeV, cs2`

不同参数组合分开保存，文件名包含 `_Rvxx_Beffxx`；表中也保留 `Rv` 和 `B_eff_MeV_fm3`。
每组保留全部延拓点；不稳定点也保留，可由 `cs2<0` 识别。

完整模式 `--csv-mode full` 在同名 CSV 中保存全部组分、质量、化学势、
求根/热力学诊断，并加入 cgs 压强、能量密度、`ε/c²` 质量密度和 km⁻² 几何单位。
两种模式均按 `eos_Rv{Rv}_Beff{B_eff}` 命名，每组只输出 `.csv`、`.svg`。

每个 SVG 展示 P–ε 和声速曲线。TOV 插值对象仍可直接在 Python 中调用，
但默认脚本不另存 TOV 分支表：

```python
from example.NJL_T0 import NJLModel, Parameters, tov_segment

model = NJLModel(Parameters(Rv=0.5, B_eff=0.0))
rows = model.scan()
eos, info = tov_segment(model, rows)
if eos is None:
    raise RuntimeError(info)
epsilon = eos.epsilon_of_pressure(100.0)  # 输入/输出都是 MeV/fm^3
cs2_interpolated = eos.sound_speed_sq(100.0)
print(eos.pressure_bounds, info["surface_available"])
```

原始表中的 `cs2` 来自隐函数微分，插值对象的 `sound_speed_sq` 是其 PCHIP
插值的导数，两者需区分。压力越界会抛异常，不进行静默外推。

**默认 Rv=0 的低密度区存在机械不稳定段。** 输出表不替换为 Maxwell/Gibbs 混合相，
因此 `tov_segment` 返回的分支标记为 `core_only`，不能单凭它把 TOV 积分到零压。
其他默认 Rv 的连续分支可连接到 `B_eff=0` 的解析真空端点；
用于插值的低密度段额外加入对数网格，而原始 601 点扫描保持不变。
若 `B_eff>0` 且稳定分支跨过零压，则重新求解真实零压点，保留有限表面密度。
若扫描范围内没有合适端点则报告原因，不人为补壳层。

这些都是所追踪均匀分支的性质，不代表已证明全局基态或夸克物质绝对稳定。
`NJL_TOV.py` 将此 EOS 接入 `src` 的 GR/EGB TOV 积分器。

## 直接计算 NJL 恒星

```bash
# 默认 Rv=0.5 > Rv_crit，B_eff=10 MeV/fm³，连续 600 点，alpha=0/6 km²
uv run python -m example.NJL_TOV

# 加密中心压力网格或指定范围
uv run python -m example.NJL_TOV --rv 0.5 --b-eff 10 --points 1200 --alpha 0 6
uv run python -m example.NJL_TOV --pc-min 0.01 --pc-max 800 --points 600
# 径向步长减半检查
uv run python -m example.NJL_TOV --dr 0.01
```

脚本实际调用 `NJLModel`（含 u/d/s/e/μ），从稀薄态延拓到 nB/n0=12，
用 `tov_segment` 求出真实零压端点，再转成 JAX `tabulated_eos`。
默认从 Pc=0.001 MeV/fm³ 到 EOS 上限按对数网格连续扫描 600 点。
内部将 MeV/fm³ 转成 km⁻²，用 JIT `mass_radius_sequence` / `lax.map`
依次求解每个中心压力对应的恒星。EOS 使用单初值延拓；每颗星从各自中心初值积分。
原先 nB/n0≤6 未覆盖默认 GR/EGB 的质量峰值，因此这个 M–R 入口的密度上限
提高到 12，仍检查所有夸克费米动量小于 Λ。

只生成一对文件，不改已有 `eos_*.csv/.svg`：

- `data/example__NJL0T/mr_Rv0.5_Beff10.csv`
- `data/example__NJL0T/mr_Rv0.5_Beff10.svg`

CSV 合并所选 alpha 的序列，包含 Rv、B_eff、alpha、中心压力/能量密度、M、R、
红移与求解状态。SVG 左侧为 M–R，右侧为 M–Pc，展示峰值及高压侧质量下降段。
同参数运行覆盖同名 MR 文件；`--output` 可指定其他目录。
输出的峰值为网格采样峰值，可用 `--points` 加密；它不等同于径向稳定性证明。
默认范围是模型有效范围内覆盖峰值的有限序列，不进行 EOS 范围外的外推。

这个入口要求 `Rv>0.0670584907704`，并检查实际扫描分支的正压缩率和可用表面。
此临界值对应当前固定 NJL 参数和所追踪均匀分支，不自动重新搜索临界耦合。
`B_eff=10` 的表面为有限密度零压点；未附加壳层。
中心压力超过 EOS 表上限或缺少零压端点时会报错。TOV 失败点在 CSV 中保留状态，
图中留空且程序以非零状态退出。未包围峰值时终端明确报告 `peak NOT bracketed`，
图中用叉号而非星号标注边界处最大采样质量。

## 一阶转变消失的耦合

```bash
uv run python -m example.check_njl_eos --find-critical
# 单独搜索，只打印结果，不写数据文件
uv run python -m example.find_critical_rv
```

本模型的零温、局域电中性均匀分支，在 `1e-8≤nB/n0≤6` 范围内得到：

| 量 | 数值 |
| --- | --- |
| 临界 Rv=Gv/Gs | 0.06705849 |
| 临界 Gv | 0.01171717 fm² = 3.00910×10⁻⁷ MeV⁻² |
| 临界 nB/n0 | 0.8158153 |
| 临界 μB | 1004.7695 MeV |
| 临界压力（B_eff=0） | 1.2563188 MeV/fm³ |
| 实用取值 | Rv=0.075，有正压缩率余量 |

判据是 `min[dμB/d(nB/n0)]=0`，在内部极小点处两条 spinodal 边界合并。
利用零温第一定律，搜索量等于 `(dP/dnB)/(nB/n0)`。不采用全区间最小声速平方
作为根函数，因为普通稀薄物质的声速也会趋近零。
先用 JAX 延拓扫描找各内部极小区间，再以局部单初值场求解连续加密；
外层夹逼 Rv。密度步长 0.01 与 0.005 给出的临界 Rv 差小于 10⁻⁹。

另以能量下凸包估计共存区间，然后求等 μB、等 P 的两个局域电中性端点，
确认这是逐渐消失的一阶共存区，而不是仅凭粗网格看不到负声速：

| Rv | 共存低密度 nB/n0 | 共存高密度 nB/n0 | 密度跃变 fm⁻³ |
| --- | --- | --- | --- |
| 0.05705849 | 0.65155686 | 0.98640221 | 0.05357526 |
| 0.06605849 | 0.76297632 | 0.86928216 | 0.01700893 |

这些 Maxwell 端点仅用于验证，不替换 CSV 中的原始均匀分支。
`Rv=0.075` 的全区间最小 `dμB/d(nB/n0)≈0.87476 MeV>0`。
常数 B_eff 不改变这个内部手征转变的临界值，但会平移压力和改变零压表面；
不能据此排除真空边界、强子相或未搜索的不连通分支上的其他相变。

矢量排斥削弱均匀手征一阶转变的物理背景可参考
[Carignano et al., arXiv:1007.1397](https://arxiv.org/abs/1007.1397)；
这里的具体数值来自本代码及原仓库参数，不能套用其他矢量道约定的数值。

## 验证

```bash
uv run python -m pytest -q
uv run ruff check example tests
```

包括完整默认扫描、独立积分重建参考零温势、轻子阈值与 `dP/dμ=n`、
有限差分第一定律/声速、袋常数平移、单位转换、TOV 越界和密度步长减半。
验证的是参考方程的零温实现；未声称运行原 Julia 的有限温脚本作逐点对照。
