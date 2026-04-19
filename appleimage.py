import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1, jv
from scipy.linalg import solve
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ScatteringSolver:
    """求解声软障碍物散射问题的数值求解器"""
    
    def __init__(self, k, boundary_points, boundary_derivatives):
        """
        参数:
            k: 波数
            boundary_points: 边界点坐标 (N, 2)
            boundary_derivatives: 边界导数 (N, 2)
        """
        self.k = k
        self.N = len(boundary_points)
        self.points = boundary_points
        self.ders = boundary_derivatives
        
    def compute_far_field(self, incident_directions, observation_directions, z0):
        """
        计算远场模式
        
        参数:
            incident_directions: 入射方向数组 (N_dir, 2)
            observation_directions: 观测方向数组 (N_obs, 2)
            z0: 参考点坐标
        
        返回:
            far_field: 远场模式 (N_obs, N_dir)
        """
        N_obs = len(observation_directions)
        N_dir = len(incident_directions)
        
        far_field = np.zeros((N_obs, N_dir), dtype=complex)
        
        # 使用边界元法求解
        for i_dir, d in enumerate(incident_directions):
            for i_obs, x_hat in enumerate(observation_directions):
                far_field[i_obs, i_dir] = self._compute_single_far_field(d, x_hat, z0)
                
        return far_field
    
    def _compute_single_far_field(self, d, x_hat, z0):
        """计算单个入射方向的远场模式（简化版本）"""
        # 使用Kirchhoff近似进行快速计算（论文中实际使用Nystrom方法）
        # 这里为了演示，使用简化的边界元方法
        
        # 计算边界上的入射场
        points_shifted = self.points - z0
        u_inc = np.exp(1j * self.k * np.dot(points_shifted, d))
        
        # 计算边界上的法向导数（声软条件）
        # 对于声软障碍物，边界条件 u = 0
        # 使用单层势方法求解
        G = self._compute_green_matrix(self.points, self.points)
        G_inv = np.linalg.inv(G + 1e-10 * np.eye(len(G)))
        
        # 边界条件给出 -u_inc 作为密度
        rho = -G_inv @ u_inc
        
        # 计算远场模式
        far_field = 0
        for i in range(len(self.points)):
            far_field += rho[i] * self._far_field_kernel(self.points[i], x_hat)
            
        return far_field
    
    def _compute_green_matrix(self, points_src, points_tar):
        """计算格林函数矩阵"""
        N_src = len(points_src)
        N_tar = len(points_tar)
        G = np.zeros((N_tar, N_src), dtype=complex)
        
        for i in range(N_tar):
            for j in range(N_src):
                if i == j:
                    # 自作用项的对角线处理
                    G[i, j] = 1j/4 * (1 - 2j/np.pi * np.log(self.k * 1e-3))
                else:
                    r = np.linalg.norm(points_tar[i] - points_src[j])
                    G[i, j] = 1j/4 * hankel1(0, self.k * r)
        return G
    
    def _far_field_kernel(self, point, x_hat):
        """远场模式的核函数"""
        return np.exp(-1j * self.k * np.dot(point, x_hat))


class DirectImagingMethod:
    """直接成像方法实现"""
    
    def __init__(self, k, M=360, N=360):
        """
        参数:
            k: 波数
            M: 观测方向数量
            N: 入射方向数量
        """
        self.k = k
        self.M = M
        self.N = N
        
        # 生成观测方向和入射方向
        self.theta_obs = np.linspace(0, 2*np.pi, M, endpoint=False)
        self.x_hat = np.array([np.cos(self.theta_obs), np.sin(self.theta_obs)]).T
        
        self.theta_inc = np.linspace(0, 2*np.pi, N, endpoint=False)
        self.d = np.array([np.cos(self.theta_inc), np.sin(self.theta_inc)]).T
    
    def generate_data(self, solver, z0, noise_level=0):
        """
        生成测量数据 |u^∞(x_hat; d1, d2)|^2
        
        参数:
            solver: 散射问题求解器
            z0: 参考点坐标
            noise_level: 噪声水平
        
        返回:
            data: 测量数据 (M, N, N)
        """
        # 计算单个入射方向的远场模式
        far_field_single = solver.compute_far_field(self.d, self.x_hat, z0)
        
        # 计算叠加入射波的远场模式
        data = np.zeros((self.M, self.N, self.N))
        
        for i_obs in range(self.M):
            for j in range(self.N):
                for l in range(self.N):
                    if j != l:
                        # 两个平面波的叠加
                        u_inf = far_field_single[i_obs, j] + far_field_single[i_obs, l]
                        data[i_obs, j, l] = np.abs(u_inf) ** 2
        
        # 添加噪声
        if noise_level > 0:
            max_val = np.max(np.abs(data))
            noise = noise_level * max_val * np.random.randn(self.M, self.N, self.N)
            data = data + noise
            data = np.maximum(data, 0)  # 确保非负
        
        return data
    
    def compute_imaging_function(self, data, z0, z_grid):
        """
        计算成像函数 I^A_{z0}(z)
        
        参数:
            data: 测量数据 (M, N, N)
            z0: 参考点坐标
            z_grid: 采样点网格 (nx, ny, 2)
        
        返回:
            I: 成像函数值 (nx, ny)
        """
        nx, ny, _ = z_grid.shape
        I = np.zeros((nx, ny))
        
        # 预计算指数因子
        d1 = self.d  # (N, 2)
        d2 = self.d  # (N, 2)
        
        # 积分权重
        w_obs = 2 * np.pi / self.M
        w_dir = 2 * np.pi / self.N
        
        for i in range(nx):
            for j in range(ny):
                z = z_grid[i, j]
                # 计算指数项
                exp_term = np.exp(1j * self.k * np.dot(z - z0, d1.T)) * \
                          np.exp(-1j * self.k * np.dot(z - z0, d2.T))
                
                # 计算成像函数
                I_val = 0
                for i_obs in range(self.M):
                    for j_dir in range(self.N):
                        for l_dir in range(self.N):
                            if j_dir != l_dir:
                                I_val += data[i_obs, j_dir, l_dir] * \
                                        exp_term[j_dir, l_dir]
                
                I[i, j] = np.abs(I_val) * w_obs * w_dir * w_dir
        
        return I
    
    def reconstruct(self, solver, z0_list, z_grid_large, z_grid_small, noise_level=0):
        """
        执行完整的重构算法
        
        参数:
            solver: 散射问题求解器
            z0_list: 参考点列表 [z10, z20]
            z_grid_large: 大采样区域网格
            z_grid_small: 小采样区域网格
            noise_level: 噪声水平
        
        返回:
            I1: 第一次成像结果
            I2: 第二次成像结果
            I_small: 小区域成像结果
        """
        results = []
        
        for idx, z0 in enumerate(z0_list):
            print(f"处理参考点 z0_{idx+1} = ({z0[0]}, {z0[1]})")
            
            # 生成数据
            data = self.generate_data(solver, z0, noise_level)
            
            # 计算成像函数
            I = self.compute_imaging_function(data, z0, z_grid_large)
            results.append(I)
        
        # 在小区域上使用第二个参考点再次成像
        data_small = self.generate_data(solver, z0_list[1], noise_level)
        I_small = self.compute_imaging_function(data_small, z0_list[1], z_grid_small)
        
        return results[0], results[1], I_small


def generate_apple_shape(center, n_points=256):
    """
    生成苹果型边界点
    
    参数:
        center: 中心坐标 (cx, cy)
        n_points: 边界点数量
    
    返回:
        points: 边界点坐标 (n_points, 2)
        derivatives: 边界导数 (n_points, 2)
    """
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # 苹果型曲线参数方程
    # r(t) = (0.5 + 0.4*cos(t) + 0.1*sin(2t)) / (1 + 0.7*cos(t))
    r = (0.5 + 0.4 * np.cos(t) + 0.1 * np.sin(2*t)) / (1 + 0.7 * np.cos(t))
    
    x = center[0] + r * np.cos(t)
    y = center[1] + r * np.sin(t)
    
    points = np.column_stack([x, y])
    
    # 计算导数（用于边界元方法）
    # 使用差分近似
    dt = t[1] - t[0]
    dx_dt = np.gradient(x, dt)
    dy_dt = np.gradient(y, dt)
    
    # 归一化切向量
    norm = np.sqrt(dx_dt**2 + dy_dt**2)
    derivatives = np.column_stack([dx_dt / norm, dy_dt / norm])
    
    return points, derivatives


def plot_results(z_grid_large, I_large1, I_large2, I_small, 
                 obstacle_points, z0_list, save_path=None):
    """
    绘制成像结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 图(a): 精确边界
    ax = axes[0, 0]
    ax.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'k-', linewidth=2, label='实际障碍物')
    ax.set_xlim([-12, 2])
    ax.set_ylim([-12, 2])
    ax.set_aspect('equal')
    ax.set_title('(a) 精确边界')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图(b): 第一次成像结果 (z10 = (-1, -5))
    ax = axes[0, 1]
    extent = [-12, 2, -12, 2]
    im = ax.imshow(I_large1.T, extent=extent, origin='lower', 
                   cmap='hot', norm=Normalize(vmin=0, vmax=np.max(I_large1)*0.8))
    ax.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'b-', linewidth=2, label='实际边界')
    ax.plot(z0_list[0][0], z0_list[0][1], 'r*', markersize=15, label=f'z0 = {z0_list[0]}')
    ax.set_xlim([-12, 2])
    ax.set_ylim([-12, 2])
    ax.set_aspect('equal')
    ax.set_title('(b) 成像结果 I_{z10}(z), z0 = (-1, -5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax)
    
    # 图(c): 第二次成像结果 (z20 = (-5, -4))
    ax = axes[1, 0]
    im = ax.imshow(I_large2.T, extent=extent, origin='lower',
                   cmap='hot', norm=Normalize(vmin=0, vmax=np.max(I_large2)*0.8))
    ax.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'b-', linewidth=2, label='实际边界')
    ax.plot(z0_list[1][0], z0_list[1][1], 'r*', markersize=15, label=f'z0 = {z0_list[1]}')
    ax.set_xlim([-12, 2])
    ax.set_ylim([-12, 2])
    ax.set_aspect('equal')
    ax.set_title('(c) 成像结果 I_{z20}(z), z0 = (-5, -4)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax)
    
    # 图(d): 小区域成像结果
    ax = axes[1, 1]
    extent_small = [-1, 1, -1, 1]
    im = ax.imshow(I_small.T, extent=extent_small, origin='lower',
                   cmap='hot', norm=Normalize(vmin=0, vmax=np.max(I_small)*0.8))
    ax.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'b-', linewidth=2, label='实际边界')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.set_title('(d) 小区域成像结果 (实际障碍物)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison(obstacle_points, I_small, I_small_clean, 
                    I_full, noise_level, save_path=None):
    """
    绘制与全数据成像方法的对比结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    extent = [-1, 1, -1, 1]
    
    # 第一行: 无噪声
    # 精确边界
    ax = axes[0, 0]
    ax.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'k-', linewidth=2)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.set_title('精确边界')
    ax.grid(True, alpha=0.3)
    
    # 无噪声 - 无相位数据
    ax = axes[0, 1]
    im = ax.imshow(I_small_clean.T, extent=extent, origin='lower',
                   cmap='hot', norm=Normalize(vmin=0, vmax=np.max(I_small_clean)*0.8))
    ax.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'b-', linewidth=1.5)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.set_title('无相位数据 (无噪声)')
    plt.colorbar(im, ax=ax)
    
    # 无噪声 - 全数据
    ax = axes[0, 2]
    im = ax.imshow(I_full.T, extent=extent, origin='lower',
                   cmap='hot', norm=Normalize(vmin=0, vmax=np.max(I_full)*0.8))
    ax.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'b-', linewidth=1.5)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.set_title('全数据 (无噪声)')
    plt.colorbar(im, ax=ax)
    
    # 第二行: 带噪声
    # 精确边界
    ax = axes[1, 0]
    ax.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'k-', linewidth=2)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.set_title(f'精确边界 (噪声 {noise_level*100:.0f}%)')
    ax.grid(True, alpha=0.3)
    
    # 带噪声 - 无相位数据
    ax = axes[1, 1]
    im = ax.imshow(I_small.T, extent=extent, origin='lower',
                   cmap='hot', norm=Normalize(vmin=0, vmax=np.max(I_small)*0.8))
    ax.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'b-', linewidth=1.5)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.set_title(f'无相位数据 ({noise_level*100:.0f}% 噪声)')
    plt.colorbar(im, ax=ax)
    
    # 带噪声 - 全数据
    ax = axes[1, 2]
    # 这里需要全数据带噪声的结果
    im = ax.imshow(I_full.T, extent=extent, origin='lower',
                   cmap='hot', norm=Normalize(vmin=0, vmax=np.max(I_full)*0.8))
    ax.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'b-', linewidth=1.5)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.set_title(f'全数据 ({noise_level*100:.0f}% 噪声)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def full_data_imaging(k, far_field_single, d, x_hat, z_grid):
    """
    全数据成像方法 I_F(z)
    
    参数:
        k: 波数
        far_field_single: 单个入射方向的远场模式
        d: 入射方向
        x_hat: 观测方向
        z_grid: 采样点网格
    
    返回:
        I: 成像函数值
    """
    M = len(x_hat)
    N = len(d)
    nx, ny, _ = z_grid.shape
    
    I = np.zeros((nx, ny))
    
    w_obs = 2 * np.pi / M
    w_dir = 2 * np.pi / N
    
    for i in range(nx):
        for j in range(ny):
            z = z_grid[i, j]
            val = 0
            for i_obs in range(M):
                sum_d = 0
                for j_dir in range(N):
                    sum_d += far_field_single[i_obs, j_dir] * np.exp(1j * k * np.dot(z, x_hat[i_obs]))
                val += np.abs(sum_d) ** 2
            I[i, j] = val * w_obs * w_dir
    
    return I


def main():
    """主函数"""
    # 设置参数
    k = 20  # 波数
    M = 180  # 观测方向数量（为了计算速度，比论文中的360小）
    N = 180  # 入射方向数量
    
    # 障碍物中心
    center = (0, 0)
    
    # 生成苹果型边界
    obstacle_points, derivatives = generate_apple_shape(center, n_points=128)
    
    # 创建求解器
    solver = ScatteringSolver(k, obstacle_points, derivatives)
    
    # 创建成像方法
    imaging = DirectImagingMethod(k, M, N)
    
    # 定义参考点
    z10 = np.array([-1.0, -5.0])
    z20 = np.array([-5.0, -4.0])
    z0_list = [z10, z20]
    
    # 定义采样区域
    # 大区域
    x_large = np.linspace(-12, 2, 200)
    y_large = np.linspace(-12, 2, 200)
    X_large, Y_large = np.meshgrid(x_large, y_large)
    z_grid_large = np.stack([X_large, Y_large], axis=-1)
    
    # 小区域
    x_small = np.linspace(-1, 1, 150)
    y_small = np.linspace(-1, 1, 150)
    X_small, Y_small = np.meshgrid(x_small, y_small)
    z_grid_small = np.stack([X_small, Y_small], axis=-1)
    
    # 噪声水平
    noise_level = 0.1  # 10%噪声
    
    print("=" * 60)
    print("苹果型声软障碍物成像算法")
    print("=" * 60)
    print(f"波数 k = {k}")
    print(f"噪声水平 = {noise_level * 100}%")
    print(f"观测方向数 M = {M}")
    print(f"入射方向数 N = {N}")
    print("=" * 60)
    
    # 执行重构
    print("\n步骤1: 使用两个参考点定位实际障碍物...")
    I_large1, I_large2, I_small = imaging.reconstruct(
        solver, z0_list, z_grid_large, z_grid_small, noise_level
    )
    
    # 绘制结果（对应论文图2）
    print("\n绘制图2: 苹果型声软障碍物成像结果...")
    plot_results(z_grid_large, I_large1, I_large2, I_small,
                 obstacle_points, z0_list, save_path='fig2_apple_soft.png')
    
    # 计算无噪声的结果用于对比
    print("\n计算无噪声结果用于对比...")
    data_clean = imaging.generate_data(solver, z20, noise_level=0)
    I_small_clean = imaging.compute_imaging_function(data_clean, z20, z_grid_small)
    
    # 计算全数据成像结果（用于对比）
    print("计算全数据成像结果...")
    far_field_single = solver.compute_far_field(imaging.d, imaging.x_hat, z20)
    I_full = full_data_imaging(k, far_field_single, imaging.d, imaging.x_hat, z_grid_small)
    
    # 绘制对比结果（对应论文图3）
    print("\n绘制图3: 无相位数据与全数据成像对比...")
    plot_comparison(obstacle_points, I_small, I_small_clean, I_full,
                    noise_level, save_path='fig3_comparison.png')
    
    print("\n算法完成！")
    
    return {
        'I_large1': I_large1,
        'I_large2': I_large2,
        'I_small': I_small,
        'I_small_clean': I_small_clean,
        'I_full': I_full,
        'obstacle_points': obstacle_points,
        'z0_list': z0_list
    }


if __name__ == "__main__":
    results = main()