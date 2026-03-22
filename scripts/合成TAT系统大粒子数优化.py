import os
import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor, jmat, qeye, sesolve, Qobj, Options
from scipy.stats import binom
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ================= 绘图全局设置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False 

# ================= 高效底层函数 =================
def custom_spin_coherent(j, theta, phi):
    """利用二项分布解析公式，极速生成高维相干自旋态"""
    N = int(2 * j)
    i = np.arange(N + 1)
    p = np.sin(theta / 2)**2
    prob_m = binom.pmf(i, N, p)
    amp = np.sqrt(prob_m)
    phase = np.exp(1j * i * phi)
    return Qobj(amp * phase).unit()

def calculate_Vmin_dynamics(exp_Jx, exp_Jy, exp_Jz, Cov_xx, Cov_yy, Cov_zz, Cov_xy, Cov_yz, Cov_zx):
    """完全向量化的协方差矩阵本征值求解（极小方差 V_min）"""
    T = len(exp_Jx)
    R = np.sqrt(exp_Jx**2 + exp_Jy**2 + exp_Jz**2)
    R_safe = np.where(R == 0, 1e-12, R)
    
    n0 = np.zeros((T, 3, 1))
    n0[:, 0, 0], n0[:, 1, 0], n0[:, 2, 0] = exp_Jx / R_safe, exp_Jy / R_safe, exp_Jz / R_safe
    
    I = np.eye(3).reshape(1, 3, 3)
    P = I - np.matmul(n0, np.transpose(n0, axes=(0, 2, 1)))
    
    Sigma = np.zeros((T, 3, 3))
    Sigma[:, 0, 0], Sigma[:, 1, 1], Sigma[:, 2, 2] = Cov_xx, Cov_yy, Cov_zz
    Sigma[:, 0, 1] = Sigma[:, 1, 0] = Cov_xy
    Sigma[:, 1, 2] = Sigma[:, 2, 1] = Cov_yz
    Sigma[:, 0, 2] = Sigma[:, 2, 0] = Cov_zx
    
    Sigma_tilde = np.matmul(P, np.matmul(Sigma, P))
    tr_Sigma = np.trace(Sigma_tilde, axis1=1, axis2=2)
    tr_Sigma_sq = np.trace(np.matmul(Sigma_tilde, Sigma_tilde), axis1=1, axis2=2)
    
    V_min = 0.5 * (tr_Sigma - np.sqrt(np.abs(2 * tr_Sigma_sq - tr_Sigma**2)))
    return V_min

# ================= TAT 核心并行计算任务 =================
def run_sim_task_TAT(args):
    N_s, N_j, tau, g = args
    j_j, j_s = N_j / 2.0, N_s / 2.0
    
    # 1. 理论 TAT 预估与 t_max 设定 (依据论文公式 13)
    t_min_est = (8 * np.log(4 * N_s)) / (g**2 * tau * N_j * N_s)
    # 对于 TAT 这种急速掉落再急速反弹的曲线，1.2 倍的时间足以捕捉完整波谷
    t_max = 1.2 * t_min_est 
    
    # 构建高密度时间网格（确保不漏掉极窄的脉冲）
    delta_t = tau / 100.0
    Omega = np.pi / (2 * delta_t)
    t_num = int(t_max / delta_t) * 2
    t_list = np.linspace(0, t_max, t_num)
    
    # ================= 理论 TAT 对比组模拟 =================
    s_z, s_x, s_y = jmat(j_s, 'z'), jmat(j_s, 'x'), jmat(j_s, 'y')
    psi_0_s = custom_spin_coherent(j_s, np.pi/2, 0)
    chi_TAT = (g**2 * N_j * tau) / 16.0
    H_TAT = chi_TAT * (s_z * s_z - s_y * s_y)
    
    e_ops_single = [s_x, s_y, s_z, s_x*s_x, s_y*s_y, s_z*s_z, s_x*s_y+s_y*s_x, s_y*s_z+s_z*s_y, s_z*s_x+s_x*s_z]
    res_TAT = sesolve(H_TAT, psi_0_s, t_list, e_ops=e_ops_single, options=Options(store_states=False))
    
    SxO, SyO, SzO, Sx2O, Sy2O, Sz2O, SxyO, SyzO, SzxO = res_TAT.expect
    V_min_array_TAT = calculate_Vmin_dynamics(SxO, SyO, SzO, Sx2O-SxO**2, Sy2O-SyO**2, Sz2O-SzO**2, 
                                              0.5*SxyO-SxO*SyO, 0.5*SyzO-SyO*SzO, 0.5*SzxO-SzO*SxO)
    idx_TAT = np.argmin(V_min_array_TAT)
    t_min_TAT = t_list[idx_TAT]
    xi2_min_TAT = 4 * V_min_array_TAT[idx_TAT] / N_s

    # ================= 真实 Pulse 方案模拟 =================
    psi_0_j = custom_spin_coherent(j_j, np.pi / 2, 0)
    psi_0 = tensor(psi_0_j, psi_0_s)
    
    J_z = tensor(jmat(j_j, 'z'), qeye(int(N_s + 1)))
    S_z = tensor(qeye(int(N_j + 1)), s_z)
    J_x = tensor(jmat(j_j, 'x'), qeye(int(N_s + 1)))
    S_x = tensor(qeye(int(N_j + 1)), s_x)
    S_y = tensor(qeye(int(N_j + 1)), s_y)
    
    # --- 超高速纯向量化脉冲阵列构建 ---
    pulse_J = np.zeros_like(t_list)
    pulse_S = np.zeros_like(t_list)
    
    t_mod = np.fmod(t_list, 8*tau)
    # 修复浮点数边界的缠绕问题
    t_mod[(t_mod < 1e-10) & (t_list > 1e-10)] = 8 * tau
    
    for k in range(1, 9):
        # 寻找处于脉冲窗口内的时间点索引
        mask = (t_mod >= k*tau - delta_t - 1e-10) & (t_mod <= k*tau + 1e-10)
        
        # 填充 J 子系统脉冲 (前四个为正，后四个为负)
        pulse_J[mask] = Omega if k <= 4 else -Omega
            
        # 填充 S 子系统脉冲 (4tau时为负，8tau时为正)
        if k == 4:
            pulse_S[mask] = -Omega
        elif k == 8:
            pulse_S[mask] = Omega

    # 交由底层计算
    H_pulse = [g * S_z * J_z, [J_x, pulse_J], [S_x, pulse_S]]
    
    e_ops_pulse = [S_x, S_y, S_z, S_x*S_x, S_y*S_y, S_z*S_z, S_x*S_y+S_y*S_x, S_y*S_z+S_z*S_y, S_z*S_x+S_x*S_z]
    
    # max_step 限制确保不跳跃脉冲，这是获得海森堡极限的核心
    opts = Options(store_states=False, nsteps=50000, max_step=delta_t/2.0)
    res_pulse = sesolve(H_pulse, psi_0, t_list, e_ops=e_ops_pulse, options=opts)
    
    Sx, Sy, Sz, Sx2, Sy2, Sz2, Sxy, Syz, Szx = res_pulse.expect
    V_min_array_pulse = calculate_Vmin_dynamics(Sx, Sy, Sz, Sx2-Sx**2, Sy2-Sy**2, Sz2-Sz**2, 
                                                0.5*Sxy-Sx*Sy, 0.5*Syz-Sy*Sz, 0.5*Szx-Sz*Sx)
    
    idx_pulse = np.argmin(V_min_array_pulse)
    t_min_pulse = t_list[idx_pulse]
    xi2_min_pulse = 4 * V_min_array_pulse[idx_pulse] / N_s
    
    return N_s, xi2_min_pulse, t_min_pulse, xi2_min_TAT, t_min_TAT

# ================= 主程序入口 =================
if __name__ == "__main__":
    # 论文图 4 参数
    N_j = 100
    tau = 0.001
    g = 1
    
    # 从小粒子数到大粒子数扫描，250是终点
    N_s_list = [10, 20, 40, 80, 150, 250]
    
    args_list = [(N_s, N_j, tau, g) for N_s in N_s_list]
    results = []
    
    # 让出1个内核保证系统不卡顿
    max_workers = max(1, os.cpu_count() - 1)
    print(f"🚀 启动 TAT 极限并行池 (进程数: {max_workers})")
    print(f"最大计算维度: {(N_j+1)} x {(N_s_list[-1]+1)} = {(N_j+1)*(N_s_list[-1]+1)}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(run_sim_task_TAT, args_list), total=len(args_list)):
            results.append(res)
            
    # 解析并提取多进程传回的结果
    xi2_min_results, t_min_results = [], []
    xi2_min_TAT_results, t_min_TAT_results = [], []
    
    for res in results:
        N_s, xi_p, t_p, xi_tat, t_tat = res
        xi2_min_results.append(xi_p)
        t_min_results.append(t_p)
        xi2_min_TAT_results.append(xi_tat)
        t_min_TAT_results.append(t_tat)
        
    N_s_array = np.array(N_s_list)
    
    # ================= 绘制完美复刻的图 4(c) 和 4(d) =================
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.suptitle(f'Dynamic synthesis TAT Scaling Law ($N_j={N_j}, \\tau={tau}$)', fontsize=16)

    # 理论黑线：利用论文公式 (13) 计算
    N_s_smooth = np.logspace(np.log10(N_s_list[0]*0.8), np.log10(N_s_list[-1]*1.2), 100)
    xi2_min_th = 1.8 / N_s_smooth
    t_min_th = (8 * np.log(4 * N_s_smooth)) / ((g**2) * tau * N_j * N_s_smooth)

    # 绘制 图 4(c) : xi^2 随着 N_s 的变化
    ax[0].plot(N_s_smooth, xi2_min_th, 'k--', linewidth=1.5, label='Eq. (13) $\\xi^2_{min} \propto 1/N_s$')
    ax[0].plot(N_s_array, xi2_min_TAT_results, 'b-', alpha=0.6, linewidth=2, label='Effective TAT')
    ax[0].plot(N_s_array, xi2_min_results, 'ro', markersize=7, label='Pulse scheme')
    ax[0].set_xlabel("$N_s$", fontsize=12)
    ax[0].set_ylabel("$\\xi^2_{min}$", fontsize=12)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].grid(True, which="both", ls="--", alpha=0.3)
    ax[0].legend(fontsize=11)

    # 绘制 图 4(d) : t_min 随着 N_s 的变化
    ax[1].plot(N_s_smooth, t_min_th, 'k--', linewidth=1.5, label='Eq. (13)')
    ax[1].plot(N_s_array, t_min_TAT_results, 'b-', alpha=0.6, linewidth=2, label='Effective TAT')
    ax[1].plot(N_s_array, t_min_results, 'ro', markersize=7, label='Pulse scheme')
    ax[1].set_xlabel("$N_s$", fontsize=12)
    ax[1].set_ylabel("$t_{min}$ (units of $g^{-1}$)", fontsize=12)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].grid(True, which="both", ls="--", alpha=0.3)
    ax[1].legend(fontsize=11)

    plt.tight_layout()
    plt.show()