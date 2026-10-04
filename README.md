# NJL / RMF 混合星

项目设计见 [designing.md](designing.md)。目前完成零温 NJL EOS 示例，以及
JAX RK4 的 GR / regularized 4D EGB TOV 背景求解器。
尚未实现 RMF、Gibbs、潮汐或振荡求解器。

```bash
uv sync --locked
uv run python -m example.check_njl_eos --find-critical --benchmark
uv run python -m pytest -q
```

示例使用 JAX 64 位、Optimistix Newton 和 `jax.lax.scan`，从 `nB/n0=1e-8`
向上延拓至 `6`，保留 u/d/s、电子和 μ 子，支持 `Rv=Gv/Gs` 和 `B_eff`。
运行在 `data/example__NJL0T/` 下按参数输出 `eos_Rv{Rv}_Beff{B_eff}.csv` 和同名 `.svg`。
CSV 支持 `--csv-mode compact/full`，两种模式使用相同的参数文件名。
当前均匀分支一阶转变消失于 `Rv≈0.06705849`；取 `Rv=0.075` 可保留正压缩率余量。

方程、单位、使用方法和 TOV 适用范围见 [example/README.md](example/README.md)。
已完成工作与验证记录见 [docs/progress.md](docs/progress.md)。

## TOV 用法

实际 NJL EOS 调用入口：`uv run python -m example.NJL_TOV`。
默认 `Rv=0.5`（高于临界值）、`B_eff=10 MeV/fm³`，连续扫描 600 个中心压力，
计算 GR 和 `alpha=6 km²` 的 M–R 曲线，包含质量峰值及高压侧下降段。
输出 `data/example__NJL0T/mr_Rv0.5_Beff10.csv/.svg`；参数说明见
[example/README.md](example/README.md#直接计算-njl-恒星)。

从项目根目录导入。以下是自束缚线性 EOS 的最小示例，所有求解器输入使用
几何单位：压力/能量密度 km⁻²，长度 km，`alpha` 为 km²。
`pc` 表示中心**压力**，与参考 Julia 入口的中心能量密度不同。

```python
import jax.numpy as jnp

from src.common import MEV_FM3_TO_KM2
from src.eos import LinearEOS
from src.structure import Status, mass_radius_sequence, solve_star

eos = LinearEOS(epsilon_surface=240 * MEV_FM3_TO_KM2, cs2=1 / 3)
pc = 100 * MEV_FM3_TO_KM2
gr = solve_star(eos, pc, alpha=0.0)
egb = solve_star(eos, pc, alpha=6.0)
assert int(gr.status) == int(egb.status) == Status.SURFACE
print(float(gr.mass_msun), float(gr.radius_km), float(gr.redshift))

pcs = jnp.linspace(20, 400, 100) * MEV_FM3_TO_KM2
sequence = mass_radius_sequence(eos, pcs, alpha=6.0)
# 检查 sequence.status 后使用 sequence.mass_msun、sequence.radius_km。
```

表格通过 `src.eos.tabulated_eos(P, epsilon, units="MeV/fm3")` 接入；
输入必须已选好物理分支、按压力严格递增，不能直接重排含 spinodal 的扫描表。
NJL 示例先用 `tov_segment(model, rows)` 得到该分支，再从其 `.rows` 取
`P_MeV_fm3`、`epsilon_MeV_fm3` 两列。有限密度零压端点可直接使用；
真空 `(0,0)` 端点还需显式提供第一插值区间的低压指数 `surface_power`，
例如确定处于非相对论费米气体极限时 ε∝P³⁄⁵。不自动补零压端点或壳层。

`solve_star` 已 JIT，径向推进用 `lax.while_loop`；中心压力序列用 `lax.map`。
常规步长 `dr=0.02 km`，表面缩步定位尺度 `surface_tol_km=1e-8 km`；
后者不代表整体精度，需减半 dr 验证，插值节点/非光滑 EOS 也可能限制收敛阶。
`status=0` 为到达零压表面；1/2/3/4 分别为半径上限、步数上限、无效状态、
正压 EOS 下边界。失败时 M/R/z 为 NaN，`last_radius_km` 和 `last_state=[P,m]`
保留终止点用于诊断。当前返回端点，不存储径向剖面。

方程推导、理论分支和边界约定见 [docs/equations_egb.md](docs/equations_egb.md)。
独立验收：`uv run python -m pytest tests/test_structure.py -q`。
