import initial as ini
def deco(mn, un):
    UdoT = [[] for k in range(ini.nu)]  # 使用一个列表存储每架无人机需要执行的任务
    for i in range(ini.nt):
        t = mn[i]
        u = un[i]
        UdoT[u].append(t)
    return UdoT
