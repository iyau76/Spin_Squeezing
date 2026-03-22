import numpy as np
import matplotlib.pyplot as plt
from qutip import jmat, spin_coherent, sesolve, Options, Qobj
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import binom

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['text.usetex'] = True
def custom_spin_coherent(j, theta, phi):
    """
    使用解析公式高效生成自旋相干态，彻底避免 QuTiP 矩阵指数运算导致的内存溢出。
    """
    N = int(2 * j)
    # QuTiP 的基矢排序是从 m = j 递减到 m = -j
    # 我们设索引 i = 0, 1, ..., N，此时 m = j - i
    i = np.arange(N + 1)
    
    # 根据文献公式 (9) 和 QuTiP 基矢对应关系转换：
    # |c_m|^2 = C_{2j}^i * [sin^2(theta/2)]^i * [cos^2(theta/2)]^(2j-i)
    # 这正是一个成功率为 p = sin^2(theta/2) 的二项分布概率！
    p = np.sin(theta / 2)**2
    
    # binom.pmf 可以极高精度地计算大 N 的二项分布概率，防溢出
    prob_m = binom.pmf(i, N, p)
    amp = np.sqrt(prob_m)
    
    # 计算相位项 e^{i * (j-m) * phi_0}，由于 j-m = i，故为 e^{i * i * phi_0}
    phase = np.exp(1j * i * phi)
    
    # 组合复数概率幅并封装为 QuTiP 的态矢
    vec = amp * phase
    return Qobj(vec).unit()

class spin_system():
    def __init__(self, j, theta_0=np.pi/2, phi_0=0, chi=1, delta=0, model='OAT'):
        """
        初始化系统：构造算符和哈密顿量，但不立即进行时间演化。
        这样可以避免初始化时直接占用大量内存或计算资源。
        """
        self.j = j
        self.J_x = jmat(j, "x")
        self.J_y = jmat(j, "y")
        self.J_z = jmat(j, "z")
        
        if model == "OAT":
            self.H = chi * self.J_z * self.J_z + delta * self.J_z
        elif model == "TAT":
            self.H = chi * (self.J_z * self.J_z - self.J_y * self.J_y)
            
            print("TAT模型哈密顿量H=chi (Jz^2 - Jy^2)")
            
        self.psi_0 = custom_spin_coherent(j, theta=theta_0, phi=phi_0)
        
        # 预先定义好需要测量的期望值算符列表
        self.e_ops = [
            self.J_x, self.J_y, self.J_z, 
            self.J_x*self.J_x, self.J_y*self.J_y, self.J_z*self.J_z, 
            self.J_x*self.J_y + self.J_y*self.J_x, 
            self.J_y*self.J_z + self.J_z*self.J_y, 
            self.J_z*self.J_x + self.J_x*self.J_z
        ]

    def calculate_Vmin_dynamics(self, exp_Jx, exp_Jy, exp_Jz, Cov_xx, Cov_yy, Cov_zz, Cov_xy, Cov_yz, Cov_zx):
        """
        利用期望值计算 V_min
        """
        # 强制转换为 numpy 数组保证向量化计算正常运行
        exp_Jx, exp_Jy, exp_Jz = np.array(exp_Jx), np.array(exp_Jy), np.array(exp_Jz)
        Cov_xx, Cov_yy, Cov_zz = np.array(Cov_xx), np.array(Cov_yy), np.array(Cov_zz)
        Cov_xy, Cov_yz, Cov_zx = np.array(Cov_xy), np.array(Cov_yz), np.array(Cov_zx)
        
        T = len(exp_Jx)
        R = np.sqrt(exp_Jx**2 + exp_Jy**2 + exp_Jz**2)
        
        n0 = np.zeros((T, 3, 1))
        # 避免除以 0
        R_safe = np.where(R == 0, 1e-12, R)
        n0[:, 0, 0] = exp_Jx / R_safe
        n0[:, 1, 0] = exp_Jy / R_safe
        n0[:, 2, 0] = exp_Jz / R_safe
        
        I = np.eye(3).reshape(1, 3, 3) 
        n0T = np.transpose(n0, axes=(0, 2, 1)) 
        P = I - np.matmul(n0, n0T)
        
        Sigma = np.zeros((T, 3, 3))
        Sigma[:, 0, 0] = Cov_xx
        Sigma[:, 1, 1] = Cov_yy
        Sigma[:, 2, 2] = Cov_zz
        Sigma[:, 0, 1] = Sigma[:, 1, 0] = Cov_xy
        Sigma[:, 1, 2] = Sigma[:, 2, 1] = Cov_yz
        Sigma[:, 0, 2] = Sigma[:, 2, 0] = Cov_zx
        
        Sigma_tilde = np.matmul(P, np.matmul(Sigma, P))
        
        tr_Sigma = np.trace(Sigma_tilde, axis1=1, axis2=2)
        Sigma_tilde_sq = np.matmul(Sigma_tilde, Sigma_tilde)
        tr_Sigma_sq = np.trace(Sigma_tilde_sq, axis1=1, axis2=2)
        
        V_min = 0.5 * (tr_Sigma - np.sqrt(np.abs(2 * tr_Sigma_sq - tr_Sigma**2)))
        return V_min

    def t_min(self, t_max=6.0, t_num=300, chunk_steps=30):
        """
        分段动态演化（Early Stopping机制）
        """
        dt = t_max / t_num
        psi = self.psi_0
        t_current = 0.0
        
        all_V_min = []
        all_t = []
        
        max_chunks = int(np.ceil(t_num / chunk_steps))
        
        for i in range(max_chunks):
            # 构造当前段的时间列表，为确保无缝连接，包含上一段的终点
            t_chunk = np.linspace(t_current, t_current + chunk_steps * dt, chunk_steps + 1)
            
            # 【核心优化】：使用 Options 对象强制 QuTiP 在传入 e_ops 的同时保留状态
            opts = Options(store_states=True, 
                                       nsteps=25000, 
                                       atol=1e-10, 
                                       rtol=1e-10)
            result = sesolve(self.H, psi, t_chunk, e_ops=self.e_ops, options=opts)
            
            Jx_exp, Jy_exp, Jz_exp, Jx2_exp, Jy2_exp, Jz2_exp, Jxy_exp, Jyz_exp, Jzx_exp = result.expect
            
            # 计算协方差
            Cov_xx = Jx2_exp - Jx_exp**2
            Cov_yy = Jy2_exp - Jy_exp**2
            Cov_zz = Jz2_exp - Jz_exp**2
            Cov_xy = 0.5 * Jxy_exp - Jx_exp * Jy_exp
            Cov_yz = 0.5 * Jyz_exp - Jy_exp * Jz_exp
            Cov_zx = 0.5 * Jzx_exp - Jz_exp * Jx_exp
            
            V_min_chunk = self.calculate_Vmin_dynamics(Jx_exp, Jy_exp, Jz_exp, Cov_xx, Cov_yy, Cov_zz, Cov_xy, Cov_yz, Cov_zx)
            
            if i == 0:
                all_V_min.extend(V_min_chunk)
                all_t.extend(t_chunk)
            else:
                # 跳过第一个点，避免和上一段的终点数据重复
                all_V_min.extend(V_min_chunk[1:])
                all_t.extend(t_chunk[1:])
                
            # 【核心优化】：极小值检测，达到极小值后立刻停止计算
            if len(all_V_min) > 3:
                diff = np.diff(all_V_min)
                is_valley = (diff[:-1] < 0) & (diff[1:] > 0)
                valley_indices = np.where(is_valley)[0] + 1
                
                if len(valley_indices) > 0:
                    idx = valley_indices[0]
                    return all_t[idx], all_V_min[idx]
            
            # 将当前段最后一个波函数态作为下一段的初始态
            psi = result.states[-1]
            t_current = t_chunk[-1]
            
        # 若一直未触发反弹，返回全局最小值
        idx = np.argmin(all_V_min)
        return all_t[idx], all_V_min[idx]

# ================= 并行工作函数（必须定义在全局作用域以支持 Pickle） =================
def run_sim_theta(args):
    theta, current_j = args
    sys = spin_system(j=current_j, theta_0=theta, model='TAT')
    # 对于大规模粒子，使用合理的时间细分，如果到达极小值系统会自动跳出
    t_opt, v_opt = sys.t_min(t_max=0.05, t_num=800, chunk_steps=20)
    return theta, t_opt, v_opt

def run_sim_j(args):
    current_j, theta = args
    sys = spin_system(j=current_j, theta_0=theta, model='OAT')
    # 因为较小的 j 会花费更多时间才能达到极小值，所以设置稍大的 t_max
    t_opt, v_opt = sys.t_min(t_max=0.5, t_num=1000, chunk_steps=20)
    return current_j, t_opt, v_opt

# ================= 画图函数 =================
def draw_tminVmin_with_theta0():
    theta_scan_list = np.linspace(0.1, np.pi/2-0.1, 20)
    theta_near_equator = np.linspace(np.pi/2-0.1, np.pi/2+0.1, 10)
    theta_scan_list_1 = np.linspace(np.pi/2+0.1, np.pi-0.1, 20)
    theta_scan_list = np.concatenate((theta_scan_list, theta_near_equator, theta_scan_list_1))
    
    j_compare_list = [100]

    results_V_min_theta = {j_val: [] for j_val in j_compare_list}
    results_t_min_theta = {j_val: [] for j_val in j_compare_list}

    for current_j in j_compare_list:
        print(f"正在并行扫描 j = {current_j} 随 theta_0 的演变 ...")
        # 将参数打包，传递给进程池
        args_list = [(theta, current_j) for theta in theta_scan_list]
        
        # 【核心优化】：使用进程池进行并行模拟
        with ProcessPoolExecutor() as executor:
            # map 会保持输出结果与输入 args_list 顺序严格一致
            results = list(tqdm(executor.map(run_sim_theta, args_list), total=len(args_list)))
            
        for res in results:
            _, t_opt, v_opt = res
            results_t_min_theta[current_j].append(t_opt)
            results_V_min_theta[current_j].append(v_opt)

    # 画图部分... (与你原来的一致)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle(r'Scaling with Initial Angle $\theta_0$ (Semi-Log Y Scale)', fontsize=16)

    colors = ['purple', 'orange', 'teal']
    markers = ['o', 'd', '*']
    theta_scan_pi = theta_scan_list / np.pi

    for i, current_j in enumerate(j_compare_list):
        label_str = f'$j = {current_j}$'
        ax1.plot(theta_scan_pi, results_V_min_theta[current_j], marker=markers[i], color=colors[i], linestyle='-', linewidth=2, label=label_str)
        ax2.plot(theta_scan_pi, results_t_min_theta[current_j], marker=markers[i], color=colors[i], linestyle='-', linewidth=2, label=label_str)

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_xlabel(r'Initial Angle $\theta_0$ ($\times \pi$ rad)', fontsize=12)
        ax.axvline(0.5, color='gray', linestyle=':', label='Equator (0.5$\pi$)')
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    ax1.set_ylabel(r'Minimum Variance $V_{min\_min}$', fontsize=12)
    ax1.set_title(r'$V_{min}$ vs $\theta_0$', fontsize=14)
    ax2.set_ylabel(r'Optimal Time $t_{min}$ (s)', fontsize=12)
    ax2.set_title(r'$t_{min}$ vs $\theta_0$', fontsize=14)

    plt.tight_layout()
    plt.show()

def draw_tminVmin_with_j():
    j_scan_list = np.array([5, 10, 20, 40, 80, 160, 320, 640, 870, 1000])
    theta_compare_list = [np.pi/2, np.pi/2 * 0.98, np.pi/3]

    results_V_min = {theta: [] for theta in theta_compare_list}
    results_t_min = {theta: [] for theta in theta_compare_list}

    for theta in theta_compare_list:
        print(f"正在并行扫描 theta_0 = {theta/np.pi:.2f}π 随 j 的演变 ...")
        args_list = [(current_j, theta) for current_j in j_scan_list]
        
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(run_sim_j, args_list), total=len(args_list)))
            
        for res in results:
            _, t_opt, v_opt = res
            results_t_min[theta].append(t_opt)
            results_V_min[theta].append(v_opt)

    # 画图部分... (与你原来的一致)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle('Scaling with Spin Size $j$ (Log-Log Scale)', fontsize=16)

    colors = ['b', 'g', 'r']
    markers = ['o', 'x', 's']

    for i, theta in enumerate(theta_compare_list):
        label_str = rf'$\theta_0 = {theta/np.pi:.2f}\pi$'
        ax1.plot(j_scan_list, results_V_min[theta], marker=markers[i], color=colors[i], linestyle='-', linewidth=2, label=label_str)
        ax2.plot(j_scan_list, results_t_min[theta], marker=markers[i], color=colors[i], linestyle='-', linewidth=2, label=label_str)

    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Spin Size $j$', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    ax1.set_ylabel(r'Minimum Variance $V_{min\_min}$', fontsize=12)
    ax1.set_title(r'$V_{min}$ vs $j$', fontsize=14)
    ax2.set_ylabel(r'Optimal Time $t_{min}$ (s)', fontsize=12)
    ax2.set_title(r'$t_{min}$ vs $j$', fontsize=14)

    plt.tight_layout()
    plt.show()
    
def draw_tminximin_with_theta0():
    theta_scan_list = np.linspace(0.1, np.pi/2-0.1, 20)
    theta_near_equator = np.linspace(np.pi/2-0.1, np.pi/2+0.1, 10)
    theta_scan_list_1 = np.linspace(np.pi/2+0.1, np.pi-0.1, 20)
    theta_scan_list = np.concatenate((theta_scan_list, theta_near_equator, theta_scan_list_1))
    
    j_compare_list = [1000]

    results_V_min_theta = {j_val: [] for j_val in j_compare_list}
    results_t_min_theta = {j_val: [] for j_val in j_compare_list}

    for current_j in j_compare_list:
        print(f"正在并行扫描 j = {current_j} 随 theta_0 的演变 ...")
        # 将参数打包，传递给进程池
        args_list = [(theta, current_j) for theta in theta_scan_list]
        
        # 【核心优化】：使用进程池进行并行模拟
        with ProcessPoolExecutor() as executor:
            # map 会保持输出结果与输入 args_list 顺序严格一致
            results = list(tqdm(executor.map(run_sim_theta, args_list), total=len(args_list)))
            
        for res in results:
            _, t_opt, v_opt = res
            results_t_min_theta[current_j].append(t_opt)
            results_V_min_theta[current_j].append(v_opt * 2 / current_j)

    # 画图部分... (与你原来的一致)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
    fig.suptitle(r'Scaling with Initial Angle $\theta_0$ (Semi-Log Y Scale)', fontsize=16)

    colors = ['purple', 'orange', 'teal']
    markers = ['o', 'd', '*']
    theta_scan_pi = theta_scan_list / np.pi

    for i, current_j in enumerate(j_compare_list):
        label_str = f'$j = {current_j}$'
        ax1.plot(theta_scan_pi, results_V_min_theta[current_j], marker=markers[i], color=colors[i], linestyle='-', linewidth=2, label=label_str)
        ax2.plot(theta_scan_pi, results_t_min_theta[current_j], marker=markers[i], color=colors[i], linestyle='-', linewidth=2, label=label_str)

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_xlabel(r'Initial Angle $\theta_0$ ($\times \pi$ rad)', fontsize=12)
        ax.axvline(0.5, color='gray', linestyle=':', label='Equator (0.5$\pi$)')
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    ax1.set_ylabel(r'Minimum Variance $\xi_{S\_min}$', fontsize=12)
    ax1.set_title(r'$\xi_{S}$ vs $\theta_0$', fontsize=14)
    ax2.set_ylabel(r'Optimal Time $t_{min}$ (s)', fontsize=12)
    ax2.set_title(r'$t_{min}$ vs $\theta_0$', fontsize=14)

    plt.tight_layout()
    plt.show()
    
def draw_tminximin_with_j():
    j_scan_list = np.array([5, 10, 20, 40, 80, 160, 320, 640, 1000])
    theta_compare_list = [np.pi/2, np.pi/2 * 0.98, np.pi/3]

    results_V_min = {theta: [] for theta in theta_compare_list}
    results_t_min = {theta: [] for theta in theta_compare_list}

    for theta in theta_compare_list:
        print(f"正在并行扫描 theta_0 = {theta/np.pi:.2f}π 随 j 的演变 ...")
        args_list = [(current_j, theta) for current_j in j_scan_list]
        
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(run_sim_j, args_list), total=len(args_list)))
            
        for res in results:
            i = 0
            _, t_opt, v_opt = res
            results_t_min[theta].append(t_opt)
            results_V_min[theta].append(v_opt)
            i += 1
    
    # 画图部分... (与你原来的一致)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle('Scaling with Spin Size $j$ (Log-Log Scale)', fontsize=16)

    colors = ['b', 'g', 'r']
    markers = ['o', 'x', 's']

    for i, theta in enumerate(theta_compare_list):
        label_str = rf'$\theta_0 = {theta/np.pi:.2f}\pi$'
        ax1.plot(j_scan_list, results_V_min[theta] / j_scan_list * 2, marker=markers[i], color=colors[i], linestyle='-', linewidth=2, label=label_str)
        ax2.plot(j_scan_list, results_t_min[theta], marker=markers[i], color=colors[i], linestyle='-', linewidth=2, label=label_str)

    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Spin Size $j$', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    ax1.set_ylabel(r'Minimum Variance $\xi_{S\_min}$', fontsize=12)
    ax1.set_title(r'$\xi_{R}$ vs $j$', fontsize=14)
    ax2.set_ylabel(r'Optimal Time $t_{min}$ (s)', fontsize=12)
    ax2.set_title(r'$t_{min}$ vs $j$', fontsize=14)

    plt.tight_layout()
    plt.show()

# Windows下多进程必须被保护在 main 中
if __name__ == "__main__":
    # 你可以自由选择运行哪一个：
    #draw_tminximin_with_theta0()
    draw_tminximin_with_j()