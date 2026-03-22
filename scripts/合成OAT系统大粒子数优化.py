import os
import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor, jmat, qeye, sesolve, Qobj, Options
from scipy.stats import binom
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ================= 绘图全局设置 =================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True 
plt.rcParams['mathtext.fontset'] = 'cm'
# ================= 高效底层函数 =================
def custom_spin_coherent(j, theta, phi):
    """高效生成自旋相干态，彻底避免 QuTiP 矩阵求幂引起的内存或时间过载"""
    N = int(2 * j)
    i = np.arange(N + 1)
    p = np.sin(theta / 2)**2
    prob_m = binom.pmf(i, N, p)
    amp = np.sqrt(prob_m)
    phase = np.exp(1j * i * phi)
    return Qobj(amp * phase).unit()

def calculate_Vmin_dynamics(exp_Jx, exp_Jy, exp_Jz, Cov_xx, Cov_yy, Cov_zz, Cov_xy, Cov_yz, Cov_zx):
    """向量化计算协方差矩阵极小方差 V_min"""
    T = len(exp_Jx)
    R = np.sqrt(exp_Jx**2 + exp_Jy**2 + exp_Jz**2)
    R_safe = np.where(R == 0, 1e-12, R)
    
    n0 = np.zeros((T, 3, 1))
    n0[:, 0, 0] = exp_Jx / R_safe
    n0[:, 1, 0] = exp_Jy / R_safe
    n0[:, 2, 0] = exp_Jz / R_safe
    
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

# ================= 并行 Worker 函数 =================
def run_sim_task(args):
    """
    独立进程运行的仿真任务。
    利用公式(10)估算极小值时间，精准设定 t_max，避免不必要的冗长演化。
    """
    N_s, N_j, tau, g = args
    
    # 根据公式 (10) 精准预估波谷时间
    constant = 8 * (3**(1/6))
    t_min_est = constant / ((g**2) * tau * N_j * (N_s**(2/3)))
    
    # 我们只需要演化到波谷后面一点点即可截断 (类似于早停机制)
    t_max = 1.3 * t_min_est 
    
    j_j, j_s = N_j / 2.0, N_s / 2.0
    
    # ================= 1. 理论 OAT 对比组计算 =================
    s_z, s_x, s_y = jmat(j_s, 'z'), jmat(j_s, 'x'), jmat(j_s, 'y')
    psi_0_s = custom_spin_coherent(j_s, np.pi/2, 0)
    chi = g**2 * N_j * tau / 8.0
    H_OAT = chi * s_z * s_z
    
    t_list_oat = np.linspace(0, t_max, 1000)
    res_oat = sesolve(H_OAT, psi_0_s, t_list_oat, 
                      e_ops=[s_x, s_y, s_z, s_x*s_x, s_y*s_y, s_z*s_z, 
                             s_x*s_y+s_y*s_x, s_y*s_z+s_z*s_y, s_z*s_x+s_x*s_z], 
                      options=Options(store_states=False))
    
    SxO, SyO, SzO, Sx2O, Sy2O, Sz2O, SxyO, SyzO, SzxO = res_oat.expect
    V_min_array_oat = calculate_Vmin_dynamics(SxO, SyO, SzO, Sx2O-SxO**2, Sy2O-SyO**2, Sz2O-SzO**2, 
                                              0.5*SxyO-SxO*SyO, 0.5*SyzO-SyO*SzO, 0.5*SzxO-SzO*SxO)
    idx_oat = np.argmin(V_min_array_oat)
    t_min_oat = t_list_oat[idx_oat]
    xi2_min_oat = 4 * V_min_array_oat[idx_oat] / N_s

    # ================= 2. 真实 Pulse 方案计算 =================
    psi_0_j = custom_spin_coherent(j_j, np.pi / 2, 0)
    psi_0 = tensor(psi_0_j, psi_0_s)
    
    # 构造大维度算符 (即便Ns=1000, 算符也很稀疏，内存极小)
    J_z = tensor(jmat(j_j, 'z'), qeye(int(N_s + 1)))
    S_z = tensor(qeye(int(N_j + 1)), s_z)
    J_x = tensor(jmat(j_j, 'x'), qeye(int(N_s + 1)))
    S_x = tensor(qeye(int(N_j + 1)), s_x)
    S_y = tensor(qeye(int(N_j + 1)), s_y)
    
    delta_t = tau / 100.0
    Omega = np.pi / (2 * delta_t)
    
    t_num = int(t_max / delta_t) * 2
    t_list_pulse = np.linspace(0, t_max, t_num)
    
    def pulse_func(t, args):
        # 加入容差防止浮点数计算吃掉脉冲
        if t % tau > tau - delta_t - 1e-10:
            return Omega
        return 0.0

    H_pulse = [g * S_z * J_z, [J_x, pulse_func]]
    
    # 期望值算符
    e_ops_pulse = [S_x, S_y, S_z, 
                   S_x*S_x, S_y*S_y, S_z*S_z, 
                   S_x*S_y+S_y*S_x, S_y*S_z+S_z*S_y, S_z*S_x+S_x*S_z]
    
    # 强行约束最大步长保证踩中脉冲
    opts = Options(store_states=False, nsteps=20000, max_step=delta_t/2.0)
    res_pulse = sesolve(H_pulse, psi_0, t_list_pulse, e_ops=e_ops_pulse, options=opts)
    
    Sx, Sy, Sz, Sx2, Sy2, Sz2, Sxy, Syz, Szx = res_pulse.expect
    V_min_array_pulse = calculate_Vmin_dynamics(Sx, Sy, Sz, Sx2-Sx**2, Sy2-Sy**2, Sz2-Sz**2, 
                                                0.5*Sxy-Sx*Sy, 0.5*Syz-Sy*Sz, 0.5*Szx-Sz*Sx)
    
    idx_pulse = np.argmin(V_min_array_pulse)
    t_min_pulse = t_list_pulse[idx_pulse]
    xi2_min_pulse = 4 * V_min_array_pulse[idx_pulse] / N_s
    
    # 将计算结果传回主进程，避免传输海量数组
    return N_s, xi2_min_pulse, t_min_pulse, xi2_min_oat, t_min_oat

# ================= 主程序入口 =================
if __name__ == "__main__":
    N_j = 40
    tau = 0.005
    g = 1
    
    # 模拟从 4 到 1000 的各个节点，对数分布能完美呈现标度律
    N_s_list = [4, 10, 20, 40, 100, 200, 400, 800, 1000]
    
    # 准备多进程参数
    args_list = [(N_s, N_j, tau, g) for N_s in N_s_list]
    
    results = []
    # 使用所有可用 CPU 核心减 1，保持系统不卡顿
    max_workers = max(1, os.cpu_count() - 1)
    print(f"🚀 启动并行计算池，进程数: {max_workers}，系统维度最高至: {N_j+1} x {N_s_list[-1]+1}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # map 保持返回值和输入列表的顺序严格对应
        for res in tqdm(executor.map(run_sim_task, args_list), total=len(args_list)):
            results.append(res)
            
    # 解包结果
    xi2_min_results, t_min_results = [], []
    xi2_min_OAT_results, t_min_OAT_results = [], []
    
    for res in results:
        N_s, xi_p, t_p, xi_oat, t_oat = res
        xi2_min_results.append(xi_p)
        t_min_results.append(t_p)
        xi2_min_OAT_results.append(xi_oat)
        t_min_OAT_results.append(t_oat)
        
    N_s_array = np.array(N_s_list)
    
    # ================= 绘制完美复刻的图 2(b) 和 2(d) =================
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.suptitle(f'Dynamic synthesis Scaling Law ($N_j={N_j}, \\tau={tau} g^{{-1}}$)', fontsize=16)

    # 理论蓝线：使用方程 (10) 精确绘制
    N_s_smooth = np.logspace(np.log10(N_s_list[0]), np.log10(N_s_list[-1]), 100)
    constant = 8 * (3**(1/6))
    xi2_min_th = 0.5 * (N_s_smooth / 3)**(-2/3)
    t_min_th = constant / ((g**2) * tau * N_j * (N_s_smooth**(2/3)))

    # 图 2(b)
    ax[0].plot(N_s_smooth, xi2_min_th, 'b-', alpha=0.6, linewidth=2, label='Eq. (10)')
    ax[0].plot(N_s_array, xi2_min_OAT_results, 'k--', linewidth=1.5, label='OAT (Num)')
    ax[0].plot(N_s_array, xi2_min_results, 'ro', markersize=7, label='Pulse scheme')
    ax[0].set_xlabel("$N_s$", fontsize=12)
    ax[0].set_ylabel("$\\xi^2_{min}$", fontsize=12)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].grid(True, which="both", ls="--", alpha=0.3)
    ax[0].legend(fontsize=11)

    # 图 2(d)
    ax[1].plot(N_s_smooth, t_min_th, 'b-', alpha=0.6, linewidth=2, label='Eq. (10)')
    ax[1].plot(N_s_array, t_min_OAT_results, 'k--', linewidth=1.5, label='OAT (Num)')
    ax[1].plot(N_s_array, t_min_results, 'ro', markersize=7, label='Pulse scheme')
    ax[1].set_xlabel("$N_s$", fontsize=12)
    ax[1].set_ylabel("$t_{min}$ (units of $g^{-1}$)", fontsize=12)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].grid(True, which="both", ls="--", alpha=0.3)
    ax[1].legend(fontsize=11)

    plt.tight_layout()
    plt.show()