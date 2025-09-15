import initial as ini
def re_deco(mn, un):
    TforU = [0 for k in range(ini.nt)]
    for t in range(ini.nt):
        TforU[mn[t]] = un[t]
    return TforU