# GR / regularized 4D EGB 背景方程

本阶段仅实现静态、球对称、各向同性单流体背景；电子、μ 子等贡献包含在
总 EOS 中。参考用户的 [tov_geo.jl](https://github.com/ZhouRui-xzit/PNJLs/blob/dc6706f38a75eee0a6742890e027379891ed1051/tov_geo.jl)。

理论约定采用 [Saavedra et al., arXiv:2412.15459v2](https://arxiv.org/html/2412.15459v2)
的式 (4)、(13)、(19)、(25)–(29)：

\[
S=\frac1{16\pi}\int d^4x\sqrt{-g}\left[R+\alpha\left(
\phi\mathcal G+4G_{\mu\nu}\nabla^\mu\phi\nabla^\nu\phi
-4(\nabla\phi)^2\Box\phi+2((\nabla\phi)^2)^2\right)\right]+S_m.
\]

取 Λ=0、度规号差 (-,+,+,+)、G=c=1、最小耦合物质；φ 无量纲，
α 的代码单位为 km²，不吸收 8π。选择与 GR 连续、渐近平坦的平方根负分支：
`ds²=-exp(χ) f dt²+dr²/f+r²dΩ²`，`φ'=(√f−1)/(r√f)`。
φ 的常数偏移无关背景。该分支的标量约束已消去，无须额外 shooting。

## 实际积分公式

公开 RHS 与结果的状态顺序为 **[P,m]**。r、m 单位 km，P、ε 单位 km⁻²；m 为由下式定义的
质量函数，其表面值与真空解的渐近引力质量匹配。

\[
q=m/r^3,\qquad \Gamma=\sqrt{1+8\alpha q},\qquad
A=\frac{4q}{1+\Gamma},\qquad f=1-r^2 A,
\]
\[
\frac{dm}{dr}=4\pi r^2\epsilon(P),\qquad
\frac{dP}{dr}=-\frac{(\epsilon+P)r(A-q+4\pi P)}{\Gamma f}.
\]

这是对参考 Julia 方程的代数有理化，避免计算 `(Γ−1)/α`。
**α=0 时 Γ=1、A=2q，直接得到 GR TOV**，无需人为设置小 α。
外部 `f(R)=1−4M/[R(1+Γ(R))]`，红移 `z=1/√f(R)−1`。
仅接受 Γ²>0、f>0、压力向外递减的正则静态构型。

中心用 `q_c=4πε_c/3`，`Γ_c=√(1+8αq_c)`，`A_c=4q_c/(1+Γ_c)`：

\[
P(r_0)=P_c-\frac{(\epsilon_c+P_c)(A_c-q_c+4\pi P_c)}{2\Gamma_c}r_0^2,
\quad m(r_0)=\frac{4\pi}3\epsilon_c r_0^3.
\]

实际 RK4 内部积分 `[P,q=m/r³]`，用 `q'=(4πε−3q)/r`，避免对 m 的 RK
中间级在中心附近带来二阶全局误差。初值进一步取
`q(r₀)=q_c+4πε'_c P₂ r₀²/5`，其中 P₂ 为上述压力的二次项系数。
默认 r₀≤10⁻³ km、r₀≤dr/4，中心附近限制 h≤r/2，再增长至 dr。
表面是 EOS 已定义的 P=0 端点；有限 ε(0) 允许。
RK4 越界中间级不求 EOS，拒绝整步并二分步长。在表面邻域将越界尝试缩至
`surface_tol_km` 后，返回最后一个内部合法点作为表面近似，同时保留残余压力。
表面附近还按 `h≤(P−P_min)/(4|P'|)` 解析压力变化尺度；此限制在
`8×surface_tol_km` 处截断，允许最终越界定位。这样 ε→0 的 EOS 也不会因
单步相对压力变化过大而积累明显半径误差。
此容差控制定位尺度，不是全局积分误差；全局误差用 dr 减半检查。
达到 rmax、迭代上限、EOS 下边界 P>0 或奇异几何均有独立状态，不能当完整恒星。

单位转换使用 `G=6.67430e−8 cm³ g⁻¹ s⁻²`、`c=2.99792458e10 cm/s`：
`1 MeV/fm³ = 1.602176634e33 erg/cm³`，乘 `G/c⁴ × 10¹⁰` 得 km⁻²。
太阳质量长度沿用参考文件的 1.47664 km。转换系数较参考文件的舍入值略有差异。

本实现不包含潮汐、f-mode、径向稳定性或内部密度跳变的匹配。
