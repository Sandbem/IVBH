from src.funcs import *

# 吸积率 0 ~ 0.0625
mus = [5e-3,1e-5]
# 观测者倾角 [D]
theta0s = [17,53,80]
# 网格数量
hnum = 1000

# 初始化对象
vbh = VaidyaBlackHole()

# 生成光子运动轨迹文件
for idx1 in np.arange(len(mus)):
    mu = mus[idx1]
    # 更新mu
    vbh.update_mu(mu)
    
    for idx2 in np.arange(len(theta0s)):
        theta0 = theta0s[idx2]
        # 更新theta0
        vbh.update_theta0(theta0)
        
        # 更新文件路径 - mu或theta0发生变化时必须运行本函数
        vbh.update_filepath()
        
        # 打印日志
        print("计算中: mu = %.2e theta0 = %02d" % (mu, theta0))
        # 保存brcd
        vbh.save_brcd_h5(hnum)

        # 保存flux
        vbh.save_brcd_flux_h5()
