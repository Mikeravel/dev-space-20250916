"""
约束条件:1.无人机最大航程约束
"""
import initial as ini
import distence as dis

# 调用该函数可以验证某个粒子的编码是否满足所有约束条件
def limit_verify(UdoT):
    flag = 1
    if range_limit(UdoT) == 0:
        flag = 0
    return flag

# 通过下面的式子约束总航程不能超过最大航程，形参为某各粒子的编码
def range_limit(UdoT):
    U_range, U_complete = dis.dis_U(UdoT)
    for i in range(ini.nu):
        if U_range[i] <= ini.uav_max_range[i]:
            flag = 1     # 总航程在允许的范围内，标志记为1
        else:
            flag = 0     # 总航程超出允许的范围内，标志记为0
            break
    return flag