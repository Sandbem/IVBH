# coding: utf-8
# author: Washy [washy2718@outlook.com]
# date: 2025/04/12

import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MultipleLocator

from scipy.optimize import root,fminbound
from scipy.integrate import fixed_quad

# 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman, SimSong'
# 设置数学公式字体为stix
plt.rcParams['mathtext.fontset'] = 'stix'

# 定义类
class VaidyaBlackHole(object):
    def __init__(self, mu=1e-5, theta0=0):
        """初始化参数"""
        # 0时刻黑洞质量
        self.m0 = 1.
        # 更新吸积率相关参数
        self.update_mu(mu)
        # 更新倾角相关参数
        self.update_theta0(theta0)
        # 更新文件路径 - mu或theta0发生变化时必须运行本函数
        self.update_filepath()

    def _calculate_basic_parameters(self):
        """计算基础物理参数"""
        # 共形Killing视界
        sqrt_term = np.sqrt(1 - 16*self.mu)
        self.Rm = (self.r0/4) * (1 - sqrt_term)
        self.Rp = (self.r0/4) * (1 + sqrt_term)

        self.ucm = 1 / self.Rm
        self.ucp = 1 / self.Rp
        
        # 光子球参数
        sqrt_ph = np.sqrt(1 - 12*self.mu)
        self.Rph = (self.r0/2) * (1 - sqrt_ph)
        denom = self.Rph - 2*self.m0 - 2*self.Rph**2/self.r0
        self.bph = np.sqrt(self.Rph**3 / denom)
        
        # f(R)极值点
        self.Rf = self.r0 * np.sqrt(self.mu)
    
    def update_mu(self, new_mu):
        """更新吸积率参数"""
        # 吸积率 0 ~ 0.0625
        self.mu = new_mu
        # 常数
        self.r0 = self.m0 / self.mu
        # 更新基本参数
        self._calculate_basic_parameters()
        # 更新ISCO - mu: 0 - 6.06274e-3
        self.Risco = self.get_Risco()
        self.ucisco = 1 / self.Risco
        
    def update_theta0(self, new_theta0):
        """更新观测者倾角相关参量"""
        # 观测者倾角 [D]
        self.theta0 = int(new_theta0)
        # 转为弧度 [rad]
        self.theta0_r = self.theta0*np.pi/180
        
    def update_filepath(self):
        """h5文件路径"""
        # 存储文件夹
        savefoldpath = "./Results"
        
        b0b1foldpath = os.path.join(savefoldpath,"b0b1")
        if not os.path.exists(b0b1foldpath):
            os.makedirs(b0b1foldpath)
        # b0b1文件名
        b0b1filename = "c_b0b1_mu_{:.2e}_theta0_{:02d}.h5".format(
            self.mu, self.theta0)
        # b0b1文件存储路径
        self.b0b1filepath = os.path.join(b0b1foldpath, b0b1filename)

        brcdfoldpath = os.path.join(savefoldpath,"brcd")
        if not os.path.exists(brcdfoldpath):
            os.makedirs(brcdfoldpath)
        # brcd文件名
        brcdfilename = "c_brcd_mu_{:.2e}_theta0_{:02d}.h5".format(
            self.mu, self.theta0)
        # brcd文件存储路径
        self.brcdfilepath = os.path.join(brcdfoldpath, brcdfilename)
        # flux文件名
        fluxfilename = "c_flux_mu_{:.2e}_theta0_{:02d}.h5".format(
            self.mu, self.theta0)
        # flux文件存储路径
        self.fluxfilepath = os.path.join(brcdfoldpath, fluxfilename)

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 函数f(R).
##----------------------------------------------------------------------##
# Inputs:
#   rc          - 共形坐标系中光线距离黑洞中心距离
# Outputs:
#   fR          - 函数f(R)的值
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/05/17,19
##----------------------------------------------------------------------##
    def func_fR(self, rc):
        """f(R)"""
        return 1 - 2*self.m0/rc - 2*rc/self.r0

    def func_grad_fR(self, rc):
        """f(R)的一阶导数"""
        return 2*self.m0/rc**2 - 2/self.r0

    def func_grad2_fR(self, rc):
        """f(R)的二阶导数"""
        return -4*self.m0/rc**3

    def func_inte_fR(self, rc):
        """f(R)的一次积分"""
        coef_1 = self.r0/(2*(self.Rp - self.Rm))
        coef_2 = self.Rm*np.log(np.abs(rc/self.Rm - 1))
        coef_3 = self.Rp*np.log(np.abs(rc/self.Rp - 1))

        return coef_1 * (coef_2 - coef_3)

    def func_Theta(self, rc, tc=0):
        """共形因子"""
        return np.exp((tc + self.func_inte_fR(rc))/self.r0)

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 求解ISCO时需要使用到的方程.
##----------------------------------------------------------------------##
# Inputs:
#   rc          - 共形坐标系中光线距离黑洞中心距离
# Outputs:
#   0           - 方程等式右侧为零
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/05/19,25
##----------------------------------------------------------------------##
    def func_Risco(self, rc):
        """求解ISCO的方程"""
        # coef_1 = 3 * rc**6 
        # coef_2 = -3 * self.r0 * rc**5 
        # coef_3 = (1 + 5*self.mu) * self.r0**2 * rc**4
        # coef_4 = -8 * self.mu * self.r0**3 * rc**3 
        # coef_5 = 2 * self.mu * (1 + 13*self.mu) * self.r0**4 * rc**2
        # coef_6 = -16 * self.mu**2 * self.r0**5 * rc
        # coef_7 = 24 * self.mu**3 * self.r0**6

        # return coef_1 + coef_2 + coef_3 + coef_4 + coef_5 + coef_6 + coef_7
        
        # 考虑共形因子
        coef_1 = 6 * rc**4
        coef_2 = -8 * self.r0 * rc**3
        coef_3 = (1 + 22*self.mu) * self.r0**2 * rc**2
        coef_4 = -8 * self.mu * self.r0**3 * rc 
        coef_5 = 12 * self.mu**2 * self.r0**4

        return coef_1 + coef_2 + coef_3 + coef_4 + coef_5

    def get_Risco(self):
        """共形坐标系 - 求解ISCO"""
        # 考虑共形因子
        if self.mu<=0.00606274:
            x = root(self.func_Risco, [2,10,15,20], method='hybr').x
            rcisco = sorted(x[x>self.Rm])[0]
        else:
            rcisco = np.nan
        # 不考虑共形因子
        # rcisco = self.r0/24 * (1 + np.sqrt(1+384*self.mu) \
        #   - np.sqrt(2*(1-96*self.mu+np.sqrt(1+384*self.mu))))

        return rcisco

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 基于近地点计算冲击参数.
##----------------------------------------------------------------------##
# Inputs:
#   Rnear       - 近地点位置
# Outputs:
#   b           - 冲击参数
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/05/19
##----------------------------------------------------------------------##
    def get_Rnear2b(self, Rnear):
        """基于近地点计算冲击参数"""
        if Rnear<self.Rph:
            raise Exception("ERROR: Rnear is too small.")
        
        return np.sqrt(Rnear**3 / (Rnear - 2*self.m0 - 2*Rnear**2/self.r0))
    
    def func_b2Rnear(self, Rnear, b):
        """依据冲击参数求解近地点的方程"""
        return self.get_Rnear2b(Rnear) - b
    
    def get_b2Rnear(self, b):
        """"基于冲击参数计算近地点"""
        return root(self.func_b2Rnear,1.5*self.Rph,args=(b),method="lm").x[0]

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 函数G(U)
##----------------------------------------------------------------------##
# Inputs:
#   uc          - 光线距离黑洞中心距离的倒数 1/R
#   b           - 冲击参数
# Outputs:
#   Guc         - G(U)函数, 即 (d u / d \phi)^2
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/04/12
##----------------------------------------------------------------------##
    def func_G_uc(self, uc, b):

        return 2 * self.m0 * uc**3 - uc**2 + 2 * uc / self.r0 + 1 / b**2

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 光线偏转角度对U的导数
##----------------------------------------------------------------------##
# Inputs:
#   uc          - 光线距离黑洞中心距离的倒数 1/R
#   b           - 冲击参数
# Outputs:
#   dphi_duc    - 光线偏转角度对U的导数 d \phi / d U
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/04/12, 05/25
##----------------------------------------------------------------------##
    def func_dphi_duc(self, uc, b):
        """函数 d \phi / d u_c"""
        return 1 / np.sqrt(self.func_G_uc(uc,b))
    
    def get_phi_uc(self, uc_0, uc_1, b):
        """光线偏转角度 \phi"""
        return fixed_quad(self.func_dphi_duc, uc_0, uc_1, args=[b], n=2000)[0]

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 计算光线的最大偏转角度
##----------------------------------------------------------------------##
# Inputs:
#   b           - 冲击参数
# Outputs:
#   gamma_max   - 光线的最大偏转角度
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/04/12, 04/28, 05/25
##----------------------------------------------------------------------##
    def get_gamma_max(self, b):
        # 积分下限
        integ_down = 1 / self.Rp

        # 如果冲击参数大于临界冲击参数
        if b>=self.bph:
            # 计算近地点位置
            Rnear = self.get_b2Rnear(b)
            # 不会被黑洞捕获的光线: 积分上限最大为近地点
            integ_up = 1 / Rnear
        else:
            # 会被黑洞捕获的光线: 积分上限最大为内视界
            integ_up = 1 / self.Rm

        return self.get_phi_uc(integ_down, integ_up, b)

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 计算从观测者位置出发的光线偏转角度
##----------------------------------------------------------------------##
# Inputs:
#   b           - 冲击参数
#   Robs        - 观测者位置
# Outputs:
#   uc          - 光线距离黑洞中心距离的倒数 1/R
#   gamma       - 光线的偏转角度
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/04/17, 05/25
##----------------------------------------------------------------------##
    def get_gamma_all(self, b):
        # 初始化偏转角度
        gamma = np.zeros(2000)

        # 积分下限
        integ_down = 1/self.Rp

        # 如果冲击参数大于临界冲击参数
        if b>=self.bph:
            # 计算近地点位置
            Rnear = self.get_b2Rnear(b)
            # 定义积分范围
            uc = np.zeros(2000)
            uc[:1000] = np.linspace(1/self.Rp, 1/Rnear, 1000)
            uc[1000:] = np.linspace(1/Rnear, 1/self.Rp, 1000)
            for idx in np.arange(1000):
                # 不会被黑洞捕获的光线: 积分上限最大为近地点
                integ_up = min(uc[idx], 1/Rnear)
                # 计算光线的偏转角度
                gamma[idx] = self.get_phi_uc(integ_down, integ_up, b)
            for idx in np.arange(1000,2000,1):
                # 不会被黑洞捕获的光线: 积分上限最大为近地点
                integ_up = min(uc[idx], 1/Rnear)
                # 计算光线的偏转角度
                gamma[idx] = 2 * self.get_phi_uc(integ_down, 1/Rnear, b) \
                    - self.get_phi_uc(integ_down, integ_up, b)
        else:
            # 定义积分范围
            uc = np.linspace(1/self.Rp, 1/self.Rm, 2000)
            for idx in np.arange(len(uc)):
                # 会被黑洞捕获的光线: 积分上限最大为内视界
                integ_up = min(uc[idx], 1/self.Rm)
                # 计算光线的偏转角度
                gamma[idx] = self.get_phi_uc(integ_down, integ_up, b)

        return uc, gamma

##----------------------------------------------------------------------##
# INFO: 坐标换算 - 共形变换 r = rc*exp((tc+F(R))/r0)
##----------------------------------------------------------------------##
# Inputs:
#   rc          - 共形坐标系中光线距离黑洞中心距离
#   r           - 真实坐标系中光线距离黑洞中心距离
#   tc          - 时间项
# Outputs:
#   0           - 方程等式右侧为零
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/05/17
##----------------------------------------------------------------------##
    def func_r2R(self, rc, r, tc=0):
        """依据tc,r求解rc的方程"""
        return rc*self.func_Theta(rc,tc) - r

    def get_r2R(self, r, tc=0):
        """基于tc,r计算rc"""
        # 计算共形坐标系下对应的距离
        rc = root(self.func_r2R, .99999999*self.Rp, args=(r,tc), method="lm").x[0]
        # 判断是否合理
        if rc>self.Rp:
            # 抛出异常
            raise Exception("ERROR: R is too large!")
        elif rc<=self.Rm:
            rc = root(self.func_r2R, 1.000000001*self.Rm, args=(r,tc), method="lm").x[0]

        return rc

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 计算吸积盘指定R处的光学图像
##----------------------------------------------------------------------##
# Inputs:
#   b           - 冲击参数
#   Rd          - 吸积盘上光源距离黑洞中心距离
#   gamma       - 光线的偏转角度
# Outputs:
#   0           - 用于求解冲击参数b的方程
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/04/28
##----------------------------------------------------------------------##
    def func_rcg2b_f(self, b, Rd, gamma):
        """光线未经过近地点的求解方程"""
        return fixed_quad(self.func_dphi_duc,1/self.Rp,1/Rd,args=[b],n=2000)[0] - gamma
    
    def get_rcg2b_f(self, Rd, gamma):
        """基于rc, gamma计算冲击参数 - 不经过近地点"""
        # 计算当前条件下冲击参数的最大值
        bd = self.get_Rnear2b(Rd)

        return root(self.func_rcg2b_f,bd,args=(Rd,gamma),method="lm").x[0]

    def func_rcg2b_t(self, b, Rd, gamma):
        """光线经过近地点的求解方程"""
        # 计算近地点位置
        Rnear = self.get_b2Rnear(b)
        # 计算偏转角度最大值
        gamma_max = fixed_quad(self.func_dphi_duc,1/self.Rp,1/Rnear,args=[b],n=2000)[0]
        # 额外偏转角度
        gamma_add = fixed_quad(self.func_dphi_duc,1/self.Rp,1/Rd,args=[b],n=2000)[0]

        return 2 * gamma_max - gamma_add - gamma
    
    def get_rcg2b_t(self, Rd, gamma):
        """基于rc, gamma计算冲击参数 - 经过近地点"""
        # 计算当前条件下冲击参数的最大值
        bd = self.get_Rnear2b(Rd)

        return root(self.func_rcg2b_t,bd,args=(Rd,gamma),method="lm").x[0]

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 计算观测者处指定冲击参数光线的角半径
##----------------------------------------------------------------------##
# Inputs:
#   b           - 光线的冲击参数
#   Robs        - 观测者位置
# Outputs:
#   alpha_sh    - 观测者处的角半径
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/04/30, 05/25
##----------------------------------------------------------------------##    
    def get_tan_alpha_sh(self, b, Robs):
        """基于冲击参数计算指定观测者位置的角半径的tan值"""
        # 初始化
        fcR = self.func_fR(Robs)
        
        delta = Robs**2/b**2 - fcR
        delta[delta<=0] = np.nan

        return np.sqrt(fcR / delta)

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 计算指定观测者角度不同吸积盘半径光学图像的冲击参数.
##----------------------------------------------------------------------##
# Outputs:
#   在 Results/ 文件夹下生成 c_b0b1_mu_x.xxe-xx_theta0_xx.h5 文件
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/04/30, 05/08,14,19
##----------------------------------------------------------------------##
    def get_images_paras_c(self):
        # 设定偏转角度 [rad]
        alpha = np.linspace(0,2*np.pi,500)
        # 计算光线偏转角度 - 直接图像 [rad]
        if self.theta0_r==0:
            gamma_0 = np.full_like(alpha, np.pi/2)
        else:
            gamma_0 = np.arccos(np.cos(alpha) \
                / np.sqrt(np.cos(alpha)**2 + 1/np.tan(self.theta0_r)**2))
        # 计算光线偏转角度 - 次级图像 [rad]
        gamma_1 = 2*np.pi - gamma_0

        # 光源半径
        # Rd_min = np.ceil(self.Risco*10)/10
        # Rd = np.arange(Rd_min, 15.1, .1)
        # if (Rd_min-self.Risco)!=0:
        #     Rd = np.append(self.Risco, Rd)
        Rd = np.linspace(self.Risco, 45.1, 400)
        
        # 计算当前条件下冲击参数的最大值
        bd = np.sqrt(Rd**3/(Rd - 2*self.m0 - 2*Rd**2/self.r0))
        
        # 初始化参数
        b_0 = np.zeros([len(Rd),len(alpha)])
        b_1 = np.zeros([len(Rd),len(alpha)])
        # 循环计算
        for idx1 in np.arange(len(Rd)):
            # 计算近地点位置
            gamma_d = self.get_gamma_max(bd[idx1])
            # 循环计算冲击参数
            for idx2 in np.arange(len(alpha)):
                # 直接图像
                if gamma_0[idx2] < gamma_d:
                    b_0[idx1,idx2] = self.get_rcg2b_f(Rd[idx1],gamma_0[idx2])
                else:
                    b_0[idx1,idx2] = self.get_rcg2b_t(Rd[idx1],gamma_0[idx2])
                # 次级图像
                b_1[idx1,idx2] = root(self.func_rcg2b_t, self.bph,
                    args=(Rd[idx1],gamma_1[idx2]), method="lm").x[0]

        # 存储数据文件
        print("Save: " + self.b0b1filepath)
        with h5py.File(self.b0b1filepath,'w') as f: 
            # 存储alpha
            f.create_dataset("alpha", data=alpha, compression="lzf")
            # 存储光源半径
            f.create_dataset("Rd", data=Rd, compression="lzf")
            # 存储主要图像的冲击参数
            f.create_dataset("b_0", data=b_0, compression="lzf")
            # 存储次级图像的冲击参数
            f.create_dataset("b_1", data=b_1, compression="lzf")

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 计算指定真实位置不同吸积盘半径的光学图像.
##----------------------------------------------------------------------##
# Inputs:
#   rcobs       - [共形坐标系] 观测者的位置
# Outputs:
#   alpha       - 观测者平面角度分布
#   rcd         - [共形坐标系] 吸积盘光源半径
#   hc_0        - [共形坐标系] 主要图像在观测者平面上与原点的距离
#   hc_1        - [共形坐标系] 次级图像在观测者平面上与原点的距离
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/05/19,21,25
##----------------------------------------------------------------------##
    def get_hc(self, Robs):
        # 读取文件数据
        print("Load: " + self.b0b1filepath)
        with h5py.File(self.b0b1filepath,'r') as f:
            # alpha
            alpha = f["alpha"][:]
            # 共形坐标系 - 光源位置
            rcd = f["Rd"][:]
            # 主要图像冲击参数
            b_0 = f["b_0"][:]
            # 次级图像冲击参数
            b_1 = f["b_1"][:]

        # tan 角半径
        tan_alpha_sh_0 = self.get_tan_alpha_sh(b_0, Robs)
        # tan 角半径
        tan_alpha_sh_1 = self.get_tan_alpha_sh(b_1, Robs)

        # 共形坐标系 -  观测者平面上主要图像的距离
        hc_0 = Robs * tan_alpha_sh_0
        # 共形坐标系 -  观测者平面上次级图像的距离
        hc_1 = Robs * tan_alpha_sh_1

        return alpha, rcd, hc_0, hc_1

##----------------------------------------------------------------------##
# INFO: 真实坐标系 - 计算指定真实位置不同吸积盘半径的光学图像.
##----------------------------------------------------------------------##
# Inputs:
#   r_obs       - [真实坐标系] 观测者的位置
#   tc          - 时间项
# Outputs:
#   alpha       - 观测者平面角度分布
#   rd          - [真实坐标系] 吸积盘光源半径
#   h_0         - [真实坐标系] 主要图像在观测者平面上与原点的距离
#   h_1         - [真实坐标系] 次级图像在观测者平面上与原点的距离
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/05/25
##----------------------------------------------------------------------##
    def get_h(self, r_obs, tc=0):
        # 变换至共形坐标系的观测位置
        Robs = self.get_r2R(r_obs, tc)
        
        # 调用共形坐标系函数
        alpha, rcd, hc_0, hc_1 = self.get_hc(Robs)

        # 真实坐标系 - 光源位置
        rd = rcd * self.func_Theta(rcd, tc)

        # 共形因子 - 观测者处
        eta = r_obs / Robs
        # 真实坐标系 -  观测者平面上主要图像的距离
        h_0 = hc_0 * eta
        # 真实坐标系 -  观测者平面上次级图像的距离
        h_1 = hc_1 * eta

        return alpha, rd, h_0, h_1

##----------------------------------------------------------------------##
# INFO: 计算指定圆形赤道轨道上粒子的相对辐射强度和红移因子
##----------------------------------------------------------------------##
# Inputs:
#   rc          - 共形坐标系中光线距离黑洞中心距离
# Outputs:
#   flux_0 - 辐射强度
#   zadd1_0     - 主要图像的红移因子
#   zadd1_1     - 次级图像的红移因子
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/05/15,16,21
##----------------------------------------------------------------------##
    def get_intensity_zadd1(self):
        # 读取文件数据
        print("Load: " + self.b0b1filepath)
        with h5py.File(self.b0b1filepath,'r') as f:
            # alpha
            alpha = f["alpha"][:]
            # 共形坐标系 - 光源位置
            rcd = f["Rd"][:]
            # 主要图像冲击参数
            b_0 = f["b_0"][:]
            # 次级图像冲击参数
            b_1 = f["b_1"][:]
        
        # 计算辐射强度最大值
        _, flux_max = self.get_intensity_max()

        # 初始化
        flux_0 = np.zeros_like(b_0)
        zadd1_0 = np.zeros_like(b_0)
        zadd1_1 = np.zeros_like(b_1)
        # 循环计算
        for idx1 in np.arange(len(rcd)):
            # 计算不同光源位置处的相对辐射强度
            flux_0[idx1,:] = self.get_intensity(rcd[idx1]) / flux_max
            # 直接图像 - 红移因子
            zadd1_0[idx1,:] = self.get_zadd1(rcd[idx1], b_0[idx1,:], alpha)
            # 次级图像 - 红移因子
            zadd1_1[idx1,:] = self.get_zadd1(rcd[idx1], b_1[idx1,:], alpha)
        
        return flux_0, zadd1_0, zadd1_1

##----------------------------------------------------------------------##
# INFO: 辐射强度 - 计算指定圆形赤道轨道上粒子的辐射通量
##----------------------------------------------------------------------##
# Inputs:
#   rc          - 共形坐标系中光线距离黑洞中心距离
# Outputs:
#   intensity   - 辐射强度
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/05/15,16,21
##----------------------------------------------------------------------##
    def get_intensity(self, rc):
        # 赤道平面上诱导度量的行列式
        g = -rc**4 * self.func_Theta(rc)**8
        coef_1 = -1 / (4*np.pi*np.sqrt(-g))
        coef_2 = self.func_grad_Omega(rc)
        coef_3 = self.func_E(rc) - self.func_Omega(rc)*self.func_L(rc)
        # 积分
        integer = fixed_quad(self.func_intensity,self.Risco,rc,n=2000)[0]

        return coef_1 * coef_2 / coef_3**2 * integer

    def get_intensity_max(self):
        """辐射强度最大值"""
        rc_max = fminbound(lambda x: -self.get_intensity(x), self.Risco,30)

        return rc_max, self.get_intensity(rc_max)

    def func_gtt(self, rc):
        """gtt"""
        Theta_sq = self.func_Theta(rc)**2

        return -self.func_fR(rc) * Theta_sq

    def func_grad_gtt(self, rc):
        """gtt的一阶导数"""
        Theta_sq = self.func_Theta(rc)**2

        return -2 * self.m0 / rc**2 * Theta_sq

    def func_gpp(self, rc):
        """gpp"""
        Theta_sq = self.func_Theta(rc)**2

        return rc**2 * Theta_sq

    def func_grad_gpp(self, rc):
        """gpp的一阶导数"""
        Theta_sq = self.func_Theta(rc)**2
        fR = self.func_fR(rc)
        
        return (2*rc + 2*rc**2/(self.r0*fR)) * Theta_sq

    def func_Omega(self, rc):
        """Omega"""
        grad_gtt = self.func_grad_gtt(rc)
        grad_gpp = self.func_grad_gpp(rc)

        return np.sqrt(-grad_gtt / grad_gpp)

    def func_grad_Omega(self, rc):
        """Omega的一阶导数"""
        fR = self.func_fR(rc)
        grad2_fR = self.func_grad2_fR(rc)
        Theta_sq = self.func_Theta(rc)**2
        grad_gtt = self.func_grad_gtt(rc)
        grad_gpp = self.func_grad_gpp(rc)
        Omega = self.func_Omega(rc)

        coef_1 = -grad2_fR * grad_gpp * Theta_sq
        term = 2 * rc * (1 - 3*self.m0/rc - rc/self.r0)
        coef_2 = 2 + 2 * term / (self.r0 * fR**2)
        coef_3 = -(coef_1 - coef_2 * grad_gtt * Theta_sq)
        
        return coef_3 / (2 * grad_gpp**2 * Omega)

    def func_E(self, rc):
        """E"""
        gtt = self.func_gtt(rc)
        gpp = self.func_gpp(rc)
        Omega = self.func_Omega(rc)

        denominator = -gtt - gpp * Omega**2

        return -gtt / np.sqrt(denominator)

    def func_L(self, rc):
        """L"""
        gtt = self.func_gtt(rc)
        gpp = self.func_gpp(rc)
        Omega = self.func_Omega(rc)

        denominator = -gtt - gpp * Omega**2

        return gpp * Omega / np.sqrt(denominator)

    def func_grad_L(self, rc):
        """L的一阶导数"""
        gtt = self.func_gtt(rc)
        grad_gtt = self.func_grad_gtt(rc)
        gpp = self.func_gpp(rc)
        grad_gpp = self.func_grad_gpp(rc)
        Omega = self.func_Omega(rc)
        grad_Omega = self.func_grad_Omega(rc)

        denominator = -gtt - gpp * Omega**2
        coef_1 = (grad_gpp * Omega + gpp * grad_Omega) * np.sqrt(denominator)
        coef_2 = -grad_gtt - grad_gpp*Omega**2 - 2*gpp*Omega*grad_Omega
        coef_3 = gpp * Omega * coef_2 / (2 * np.sqrt(denominator))

        return (coef_1 - coef_3) / denominator

    def func_intensity(self, rc):
        """被积分项"""
        coef_1 = self.func_E(rc) - self.func_Omega(rc)*self.func_L(rc)
        
        return coef_1 * self.func_grad_L(rc)

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 计算红移因子
##----------------------------------------------------------------------##
# Inputs:
#   rc          - 共形坐标系中光线距离黑洞中心距离
#   b           - 冲击参数
# Outputs:
#   zadd1       - 红移因子 1+z
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/05/16
##----------------------------------------------------------------------##
    def get_zadd1(self, rc, b, alpha):
        gtt = self.func_gtt(rc)
        gpp = self.func_gpp(rc)
        Omega = self.func_Omega(rc)

        coef_1 = 1 + b*Omega*np.sin(self.theta0_r)*np.sin(alpha)
        denominator = -gtt - gpp*Omega**2
        
        return coef_1 / np.sqrt(denominator)

##----------------------------------------------------------------------##
# INFO: 共形坐标系 - 给定观测值位置计算图像和辐射并保存
##----------------------------------------------------------------------##
# Inputs:
#   hnum        - 网格数量
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/05/26,27
##----------------------------------------------------------------------##
    def save_brcd_h5(self, hnum=1024):
        """保存图像至h5."""
        # 定义观测者位置
        Robs = .9*self.Rp
        # 定义图像尺寸 - 为了保证图片效果需要与Robs同时进行修改
        hlim = 10.

        # 打印提示信息
        print("正在计算%d*%d=%.2e个网格点."%(hnum,hnum,hnum*hnum))

        # 初始化参量
        hxy = np.linspace(-hlim, hlim, hnum)
        # 生成二维网格
        px,py = np.meshgrid(hxy,hxy)
        # 计算每个网格点的角度 [R]
        alpha_0 = np.arctan2(px,-py)
        alpha_1 = np.arctan2(px,py)
        # 计算 tan 角半径 的平方
        tan_alpha_sh_sq = (px**2 + py**2) / Robs**2
        # 计算冲击参数
        b = Robs / np.sqrt(self.func_fR(Robs) * (1 + 1/tan_alpha_sh_sq))
        # 计算偏转角度 - 直接图像
        gamma_0 = np.arccos(np.cos(alpha_0) \
            / np.sqrt(np.cos(alpha_0)**2 + 1/np.tan(self.theta0_r)**2))
        # 计算偏转角度 - 次级图像
        gamma_1 = 2*np.pi - np.arccos(np.cos(alpha_1) \
            / np.sqrt(np.cos(alpha_1)**2 + 1/np.tan(self.theta0_r)**2))
        
        # 初始化 - 最大偏转角度
        gamma_max = np.zeros_like(b)
        # 初始化 - 数值求解初值
        uc0 = np.zeros_like(b)
        # 循环求解
        for idx1 in np.arange(hnum):
            for idx2 in np.arange(hnum):
                if b[idx1,idx2]<self.bph:
                    # 计算近地点位置
                    uc_near = self.ucm
                    # 初值
                    uc0[idx1,idx2] = self.ucisco
                else:
                    # 计算近地点位置
                    uc_near = 1 / self.get_b2Rnear(b[idx1,idx2])
                    # 初值
                    uc0[idx1,idx2] = min(uc_near, self.ucisco)
                # 计算偏转角度最大值
                gamma_max[idx1,idx2] = self.get_phi_uc(self.ucp, uc_near, b[idx1,idx2])
        
        # 初始化光源位置 - 直接图像
        ucd_0 = np.zeros_like(b)
        # 初始化光源位置 - 次级图像
        ucd_1 = np.zeros_like(b)
        # 循环求解
        for idx1 in np.arange(hnum):
            for idx2 in np.arange(hnum):
                # 直接图像 - 计算光源位置
                ucd_0[idx1,idx2] = self.func_b2ucd_0(
                    uc0[idx1,idx2], b[idx1,idx2], gamma_0[idx1,idx2], gamma_max[idx1,idx2]
                )
                # 次级图像 - 计算光源位置
                ucd_1[idx1,idx2] = self.func_b2ucd_1(
                    uc0[idx1,idx2], b[idx1,idx2], gamma_1[idx1,idx2], gamma_max[idx1,idx2]
                )
        # 计算光源位置
        rcd_0 = 1/ucd_0
        rcd_1 = 1/ucd_1

        # 存储数据文件
        print("Save: " + self.brcdfilepath)
        with h5py.File(self.brcdfilepath,'w') as f:
            # 观测者位置
            f.create_dataset("Robs", data=Robs)
            # ISCO
            f.create_dataset("Risco", data=self.Risco)
            # 图像尺寸
            f.create_dataset("hlim", data=hlim)
            # 网格数量
            f.create_dataset("hnum", data=hnum)
            # 二维网格 - X轴
            f.create_dataset("px", data=px, compression="lzf")
            # 二维网格 - Y轴
            f.create_dataset("py", data=py, compression="lzf")
            # 直接图像 - alpha
            f.create_dataset("alpha_0", data=alpha_0, compression="lzf")
            # 次级图像 - alpha
            f.create_dataset("alpha_1", data=alpha_1, compression="lzf")
            # 冲击参数
            f.create_dataset("b", data=b, compression="lzf")
            # 直接图像 - 光源半径
            f.create_dataset("rcd_0", data=rcd_0, compression="lzf")
            # 次级图像 - 光源半径
            f.create_dataset("rcd_1", data=rcd_1, compression="lzf")
    
    def save_brcd_flux_h5(self):
        """保存辐射强度至h5."""
        # 读取文件数据
        print("Load: " + self.brcdfilepath)
        with h5py.File(self.brcdfilepath,'r') as f:
            # ISCO
            Risco = f["Risco"][()]
            # 网格数量
            hnum = f["hnum"][()]
            # 直接图像 - alpha
            alpha_0 = f["alpha_0"][:]
            # 次级图像 - alpha
            alpha_1 = f["alpha_1"][:]
            # 冲击参数
            b = f["b"][:]
            # 直接图像 - 光源半径
            rcd_0 = f["rcd_0"][:]
            # 次级图像 - 光源半径
            rcd_1 = f["rcd_1"][:]

        # 计算辐射强度最大值
        _, flux_max = self.get_intensity_max()

        # 初始化
        flux_0 = np.zeros_like(b)
        flux_1 = np.zeros_like(b)
        zadd1_0 = np.zeros_like(b)
        zadd1_1 = np.zeros_like(b)
        obsflux_0 = np.zeros_like(b)
        obsflux_1 = np.zeros_like(b)
        
        # 循环计算
        for idx1 in np.arange(hnum):
            for idx2 in np.arange(hnum):
                if rcd_0[idx1,idx2]>=Risco:
                    # 计算不同光源位置处的辐射强度
                    flux_0[idx1,idx2] = self.get_intensity(rcd_0[idx1,idx2])
                    # 直接图像 - 红移因子
                    zadd1_0[idx1,idx2] = self.get_zadd1(
                        rcd_0[idx1,idx2], b[idx1,idx2], alpha_0[idx1,idx2])
                    # 观测强度
                    obsflux_0[idx1,idx2] = flux_0[idx1,idx2] / zadd1_0[idx1,idx2]**4
                if rcd_1[idx1,idx2]>=Risco:
                    # 计算不同光源位置处的辐射强度
                    flux_1[idx1,idx2] = self.get_intensity(rcd_1[idx1,idx2])
                    # 次级图像 - 红移因子
                    zadd1_1[idx1,idx2] = self.get_zadd1(
                        rcd_1[idx1,idx2], b[idx1,idx2], alpha_1[idx1,idx2])
                    # 观测强度
                    obsflux_1[idx1,idx2] = flux_1[idx1,idx2] / zadd1_1[idx1,idx2]**4

        # 存储数据文件
        print("Save: " + self.fluxfilepath)
        with h5py.File(self.fluxfilepath,'w') as f:
            # 最大辐射通量
            f.create_dataset("flux_max", data=flux_max)
            # 直接图像 - 辐射强度
            f.create_dataset("flux_0", data=flux_0, compression="lzf")
            # 直接图像 - 红移因子
            f.create_dataset("zadd1_0", data=zadd1_0, compression="lzf")
            # 次级图像 - 辐射强度
            f.create_dataset("flux_1", data=flux_1, compression="lzf")
            # 次级图像 - 红移因子
            f.create_dataset("zadd1_1", data=zadd1_1, compression="lzf")
            # 直接图像 - 观测者辐射强度
            f.create_dataset("obsflux_0", data=obsflux_0, compression="lzf")
            # 次级图像 - 观测者辐射强度
            f.create_dataset("obsflux_1", data=obsflux_1, compression="lzf")

    def func_gamma_b2uc(self, uc, b, gamma):
        """已知偏转角度: 冲击参数与光源位置的函数"""
        return self.get_phi_uc(self.ucp, uc, b) - gamma

    def get_gamma_b2uc(self, uc0, b, gamma):
        """已知偏转角度: 由冲击参数计算光源位置"""
        return root(self.func_gamma_b2uc, uc0, args=(b,gamma),method="lm").x[0]
    
    def func_b2ucd_0(self, uc0, b, gamma_0, gamma_max):
        """直接图像 - 已知偏转角度: 由冲击参数计算光源位置"""
        if gamma_0<gamma_max:
            gamma = gamma_0
        else:
            # 不满足计算条件直接返回
            if b<self.bph:
                return np.nan
            else:
                gamma = 2*gamma_max - gamma_0
        
        return self.get_gamma_b2uc(uc0, b, gamma)
    
    def func_b2ucd_1(self, uc0, b, gamma_1, gamma_max):
        """次级图像 - 已知偏转角度: 由冲击参数计算光源位置"""
        if b<self.bph or gamma_1>2*gamma_max:
            return np.nan
        gamma = 2 * gamma_max - gamma_1

        return self.get_gamma_b2uc(uc0, b, gamma)

##----------------------------------------------------------------------##
# INFO: 真实坐标系 - 给定观测值位置和时间生成观测辐射图像
##----------------------------------------------------------------------##
# Inputs:
#   robs        - 观测者位置
#   tc          - 时间
#   flag        - True/单帧动画 False/论文插图
##----------------------------------------------------------------------##
# author: Washy [washy2718@outlook.com]
# date: 2025/05/28
##----------------------------------------------------------------------##
    def save_flux_png(self, robs, tc, flag=False):
        # 读取文件数据
        print("Load: " + self.brcdfilepath)
        with h5py.File(self.brcdfilepath,'r') as f:
            # 直接图像 - alpha
            alpha_0 = f["alpha_0"][:]
            # 冲击参数
            b = f["b"][:]
        
        # 读取文件数据
        print("Load: " + self.fluxfilepath)
        with h5py.File(self.fluxfilepath,'r') as f:
            # ISCO
            flux_max = f["flux_max"][()]
            # 网格数量
            obsflux_0 = f["obsflux_0"][()]
            # 直接图像 - alpha
            obsflux_1 = f["obsflux_1"][:]
        obs0 = obsflux_0 / flux_max
        obs1 = obsflux_1 / flux_max
        obs0[obs0==0] = obs1[obs0==0]
        # obsflux = (obsflux_0+obsflux_1) / flux_max

        # 判断tc的数据类型
        if type(tc) in [int, float]:
            # 共形坐标 - 观测者位置
            Robs = self.get_r2R(robs, tc)
            # tan 角半径
            tan_alpha_sh = self.get_tan_alpha_sh(b, Robs)
            # 真实坐标系 - 观测者平面上的距离
            hc = robs * tan_alpha_sh
            # 真实坐标系 - 网格坐标
            px = hc * np.sin(alpha_0) 
            py = -hc * np.cos(alpha_0)
            # 绘图
            self.plot_flux_tc(px,py,obs0,tc,flag)
        elif type(tc) in [np.ndarray, list, range]:
            # 循环所有tc
            for idx in np.arange(len(tc)):
                # 共形坐标 - 观测者位置
                Robs = self.get_r2R(robs, tc[idx])
                # tan 角半径
                tan_alpha_sh = self.get_tan_alpha_sh(b, Robs)
                # 真实坐标系 - 观测者平面上的距离
                hc = robs * tan_alpha_sh
                # 真实坐标系 - 网格坐标
                px = hc * np.sin(alpha_0) 
                py = -hc * np.cos(alpha_0)
                # 绘图
                self.plot_flux_tc(px,py,obs0,tc[idx],flag)
        else:
            raise Exception("ERROR: the type of tc is wrong!")
    
    def plot_flux_tc(self, px, py, pz, tc, flag=False, hlim=30):
        """绘制图片并保存"""
        print("正在绘制 tc = %03d" % tc)
        plt.figure(figsize=(4.8,4.8),dpi=300)

        max_level = 3.0
        # 需要显示的等高线
        levels = np.linspace(0,max_level,300)

        # 直接图像 + 次级图像
        
        plt.contourf(px, py, pz, levels, cmap='hot')
        # plt.contourf(px, py, pz0, levels, cmap='hot')

        # 添加颜色棒
        plt.colorbar(fraction=0.046, pad=0.04, ticks=np.arange(0,max_level+.5,.5))

        plt.gca().set_facecolor('k')
        plt.gca().set_aspect('equal', adjustable='box')

        # 设置副坐标刻度间隔
        plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(2))

        plt.xlim(-hlim,hlim)
        plt.ylim(-hlim,hlim)

        # 调整子图参数 - 使其充满整个画布
        plt.tight_layout(pad=.2)

        if flag:
            plt.text(0.01, 0.98, 'tc = %03d'%tc, transform=plt.gca().transAxes,
                fontsize=8, ha='left', va='center', color='white')
            plt.savefig("imgs/tc_r/flux_mu_{:.2e}_theta0_{:2d}_tc_{:03d}.png".format(
                self.mu,self.theta0,tc))
        else:
            plt.title(r"$\mu$ = " + "%.1e" % self.mu \
                + r",$\quad \theta_0$ = " + "%d$^\circ$" % self.theta0 \
                + r",$\quad t_c$ = " + "%03d" % tc)
            plt.savefig("imgs/flux_mu_{:.2e}_theta0_{:2d}_tc_{:03d}.png".format(
                self.mu,self.theta0,tc))
        plt.close('all')
