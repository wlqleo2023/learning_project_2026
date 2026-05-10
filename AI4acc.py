import numpy as np

class SSRF_BPMSimulator_V2:
    """
    上海光源 (SSRF) 深度定制版 BPM 数字孪生模拟器
    包含：高阶多极展开、严谨转移阻抗、Lattice光学参数、ID与热扰动模型
    """
    def __init__(self, bpm_id, beta_x=10.0, beta_y=5.0, Dx=0.15):
        self.bpm_id = bpm_id
        
        # --- 1. SSRF 光学参数 (Lattice Optics) ---
        self.beta_x = beta_x   # 水平 beta 函数 (m)
        self.beta_y = beta_y   # 垂直 beta 函数 (m)
        self.Dx = Dx           # 色散函数 (m)
        
        # --- 2. 几何与安装参数 ---
        self.a = 0.035         # 真空盒水平半轴 35 mm
        self.b = 0.014         # 真空盒垂直半轴 14 mm
        self.r_button = 0.004  # 纽扣电极半径 ~4 mm
        
        # 安装角度 (以第一象限为例，计算其极坐标角度)
        phi_0 = np.arctan2(self.b, self.a)
        self.phi = {'A': phi_0, 'B': np.pi - phi_0, 'C': np.pi + phi_0, 'D': -phi_0}
        
        # --- 3. 高频电子学与阻抗参数 ---
        self.f_RF = 499.654e6          # SSRF 射频频率 499.654 MHz
        self.bunch_length_s = 15.3e-12 # 束团长度 15.3 ps
        self.C_button = 3.5e-12        # 纽扣对地电容 3.5 pF
        self.R_load = 50.0             # 负载电阻 50 Ω
        self.noise_floor = 1e-7        # 0.1 μm 电子学底噪
        
        # --- 4. 计算固定的转移阻抗 (Transfer Impedance) ---
        omega = 2 * np.pi * self.f_RF
        tau = self.R_load * self.C_button
        # 高通滤波器幅频响应
        hpf_mag = (omega * tau) / np.sqrt(1 + (omega * tau)**2)
        # 束团高斯频谱衰减
        attenuation = np.exp(-0.5 * (omega * self.bunch_length_s)**2)
        # 几何耦合覆盖因子 (约等于纽扣直径占垂直周长的比例)
        g_factor = self.r_button / (np.pi * self.b) 
        # 最终转移阻抗 (欧姆)
        self.Z_transfer = g_factor * self.R_load * hpf_mag * attenuation

    def _multipole_expansion(self, x, y):
        """
        核心物理：使用复变函数实现的截断多极展开 (至八极场 n=4)
        包含极坐标因子 2，并利用 tanh 实施平滑的物理边界饱和
        """
        # 使用 tanh 进行软物理边界约束，避免突变截断，且对梯度优化友好
        X_eq = np.tanh(x / self.a)
        Y_eq = np.tanh(y / self.b)
        
        # 构造复数 Z = X + jY，利用棣莫弗公式极速计算高阶多极项
        Z = X_eq + 1j * Y_eq
        
        Q_induced = {}
        for name, phi_i in self.phi.items():
            signal = 1.0  # 单极项 (Monopole)
            
            # 计算偶极(n=1) 到 八极(n=4) 的非线性耦合项
            for n in range(1, 5):
                Z_n = Z**n
                # Real(Z^n) 对应 cos(n*theta)，Imag(Z^n) 对应 sin(n*theta)
                # 多极展开公式: 2 * (r/R)^n * cos(n(phi_i - theta))
                term_n = 2 * (Z_n.real * np.cos(n * phi_i) + Z_n.imag * np.sin(n * phi_i))
                signal += term_n
                
            Q_induced[name] = signal / 4.0 # 归一化到4个电极
            
        return Q_induced

    def _apply_perturbations(self, t, dp_p, id_gap_mm, temp_delta_K):
        """
        多源扰动模型 (参考 Veglia et al. 2024 等前沿运行数据)
        """
        # 1. 色散引起的轨道偏移
        x_disp = self.Dx * dp_p
        
        # 2. 插入件 (ID) 间隙闭合引起的非线性 COD 扰动
        # 当 Gap 小于 30mm 时，剩余场积分呈指数级显著增强，并在目标 BPM 处产生正比于 sqrt(beta) 的偏移
        id_cod_x, id_cod_y = 0.0, 0.0
        if id_gap_mm < 30.0:
            # 拟合经验公式：Gap越小，指数项越大，引入微米级畸变
            kick_strength = 5e-6 * np.exp(-(id_gap_mm - 10.0) / 5.0) 
            id_cod_x = kick_strength * np.sqrt(self.beta_x)
            id_cod_y = (kick_strength * 0.5) * np.sqrt(self.beta_y) # 垂直面通常扰动较小
            
        # 3. 热漂移 (缓慢的低频正弦漂移 + 随机游走)
        # 典型SSRFF环境温变周期可能在小时级，这里用 t 模拟低频相移
        thermal_drift_x = 1e-6 * temp_delta_K * np.sin(2 * np.pi * t / 3600.0)
        
        return x_disp + id_cod_x + thermal_drift_x, id_cod_y

    def measure(self, t_sec, x_betatron, y_betatron, dp_p=0.0, id_gap_mm=50.0, temp_delta_K=0.5, I_beam=0.200):
        """
        执行一次带有全物理特性的数字孪生测量
        """
        # 1. 计算受扰动后的真实闭轨 (True Closed Orbit)
        dx_pert, dy_pert = self._apply_perturbations(t_sec, dp_p, id_gap_mm, temp_delta_K)
        x_true = x_betatron + dx_pert
        y_true = y_betatron + dy_pert

        # 2. 通过多极展开计算感应电荷空间分布
        charges = self._multipole_expansion(x_true, y_true)
        
        # 3. 施加真实的转移阻抗，得到纽扣上的高频电压信号 (伏特)
        signals = {k: v * I_beam * self.Z_transfer for k, v in charges.items()}
        
        V_A, V_B = signals['A'], signals['B']
        V_C, V_D = signals['C'], signals['D']
        total_V = V_A + V_B + V_C + V_D

        # 4. 经典差比和计算
        delta_x = (V_A + V_D) - (V_B + V_C)
        delta_y = (V_A + V_B) - (V_C + V_D)

        # 理论线性灵敏度系数 (椭圆管道近似)
        K_x = self.a / (2 * np.cos(self.phi['A']))
        K_y = self.b / (2 * np.sin(self.phi['A']))

        x_meas = K_x * (delta_x / total_V)
        y_meas = K_y * (delta_y / total_V)

        # 5. 叠加高斯电子学底噪
        x_meas += np.random.normal(0, self.noise_floor)
        y_meas += np.random.normal(0, self.noise_floor)

        return {
            'bpm_id': self.bpm_id,
            'x_meas': x_meas,
            'y_meas': y_meas,
            'x_true': x_true,   # 留给 AI 算 Loss 用的 Ground Truth
            'y_true': y_true,
            'sum_signal': total_V,
            'raw_voltages': signals
        }
