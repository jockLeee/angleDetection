from numpy import *
from scipy import optimize, odr
import functools
from matplotlib import pyplot as plt, cm, colors

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# 方法一 代数逼近法
method_1 = '代数逼近法    '
# 坐标
x = r_[14,   15,  15,  16,  16,  17,  17,  18,  19,  20,  20,  21,  22,  23,  24,  25,  26,  26,  27,  28,  29,  30,  31,  32]
y = r_[-261, -260, -261, -259, -260, -258, -259, -258, -257, -256, -257, -255, -254, -254, -253, -253, -252, -253, -252, -251, -251, -250, -250, -250]
# basename = 'arc'
print(type(x))
# 质心坐标
x_m = mean(x)
y_m = mean(y)

print(x_m, y_m)

# 相对坐标
u = x - x_m
v = y - y_m

# 相对坐标下定义中心(uc, vc)的线性系统：
#    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
#    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
Suv = sum(u * v)
Suu = sum(u ** 2)
Svv = sum(v ** 2)
Suuv = sum(u ** 2 * v)
Suvv = sum(u * v ** 2)
Suuu = sum(u ** 3)
Svvv = sum(v ** 3)

# 求线性系统
A = array([[Suu, Suv], [Suv, Svv]])
B = array([Suuu + Suvv, Svvv + Suuv]) / 2.0
uc, vc = linalg.solve(A, B)

xc_1 = x_m + uc
yc_1 = y_m + vc

# 半径残余函数
Ri_1 = sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
R_1 = mean(Ri_1)
residu_1 = sum((Ri_1 - R_1) ** 2)
ncalls_1 = 0
residu2_2 = 0

# 方法二 最小二乘法
method_2 = "最小二乘法    "


# 修饰器：用于输出反馈
def countcalls(fn):
    "decorator function count function calls "

    @functools.wraps(fn)
    def wrapped(*args):
        wrapped.ncalls += 1
        return fn(*args)

    wrapped.ncalls = 0
    return wrapped


def calc_R(xc, yc):
    """ 计算s数据点与圆心(xc, yc)的距离 """
    return sqrt((x - xc) ** 2 + (y - yc) ** 2)


@countcalls
def f_2(c):
    """ 计算半径残余"""
    Ri = calc_R(*c)
    return Ri - Ri.mean()


# 圆心估计
center_estimate = x_m, y_m
center_2, _ = optimize.leastsq(f_2, center_estimate)

xc_2, yc_2 = center_2
Ri_2 = calc_R(xc_2, yc_2)
# 拟合圆的半径
R_2 = Ri_2.mean()
residu_2 = sum((Ri_2 - R_2) ** 2)
residu2_2 = sum((Ri_2 ** 2 - R_2 ** 2) ** 2)
ncalls_2 = f_2.ncalls

# 输出列表
# fmt = '%-22s  %10.5f  %10.5f  %10.5f  %10d  %10.6f  %10.6f  %10.2f'
# print(('\n%-22s' + ' %10s' * 7) % tuple('方法  Xc  Yc  Rc  nb_calls  std(Ri)  residu  residu2'.split()))
# print('-' * (22 + 7 * (10 + 1)))
# print(fmt % (method_2, xc_2, yc_2, R_2, ncalls_2, Ri_2.std(), residu_2, residu2_2))

# 方法二b 基于Jacobian函数的最小二乘法
# method_2b = "基于Jacobian函数的最小二乘法"
#
#
# def calc_R(xc, yc):
#     """ 计算数据点据圆心(xc, yc)的距离 """
#     return sqrt((x - xc) ** 2 + (y - yc) ** 2)
#
#
# def f_2b(c):
#     """ 计算数据点与以c=(xc, yc)为圆心圆的半径差值 """
#     Ri = calc_R(*c)
#     return Ri - Ri.mean()
#
#
# def Df_2b(c):
#     """ 雅可比矩阵"""
#     xc, yc = c
#     df2b_dc = empty((len(c), x.size))
#
#     Ri = calc_R(xc, yc)
#     df2b_dc[0] = (xc - x) / Ri  # dR/dxc
#     df2b_dc[1] = (yc - y) / Ri  # dR/dyc
#     df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, newaxis]
#
#     return df2b_dc
#
#
# center_estimate = x_m, y_m
# center_2b, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)
#
# xc_2b, yc_2b = center_2b
# Ri_2b = calc_R(*center_2b)
# R_2b = Ri_2b.mean()
# residu_2b = sum((Ri_2b - R_2b) ** 2)


# 方法三 正交距离回归法


method_3 = "正交距离回归法"

def f_3(beta, x):
    """ 圆的隐式定义 """
    return (x[0] - beta[0]) ** 2 + (x[1] - beta[1]) ** 2 - beta[2] ** 2


"""参数初始化"""
R_m = calc_R(x_m, y_m).mean()
beta0 = [x_m, y_m, R_m]

lsc_data = odr.Data(row_stack([x, y]), y=1)
lsc_model = odr.Model(f_3, implicit=True)
lsc_odr = odr.ODR(lsc_data, lsc_model, beta0)
lsc_out = lsc_odr.run()

xc_3, yc_3, R_3 = lsc_out.beta
Ri_3 = calc_R(xc_3, yc_3)
residu_3 = sum((Ri_3 - R_3) ** 2)

# 方法三b  基于Jacobian函数的正交距离回归法
# method_3b  = "基于Jacobian函数的正交距离回归法"
#
# def f_3b(beta, x):
#     """ 圆定义 """
#     return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 -beta[2]**2
#
# def jacb(beta, x):
#     """ 计算关于参数β的雅可比函数，返回df_3b/dβ。"""
#     xc, yc, r = beta
#     xi, yi    = x
#
#     df_db    = empty((beta.size, x.shape[1]))
#     df_db[0] =  2*(xc-xi)                     # d_f/dxc
#     df_db[1] =  2*(yc-yi)                     # d_f/dyc
#     df_db[2] = -2*r                           # d_f/dr
#
#     return df_db
#
# def jacd(beta, x):
#     """ 计算关于输入x的雅可比函数，返回df_3b/dx。"""
#     xc, yc, r = beta
#     xi, yi    = x
#
#     df_dx    = empty_like(x)
#     df_dx[0] =  2*(xi-xc)                     # d_f/dxi
#     df_dx[1] =  2*(yi-yc)                     # d_f/dyi
#
#     return df_dx
#
# def calc_estimate(data):
#     """ 从数据中返回对参数的第一次估计。 """
#     xc0, yc0 = data.x.mean(axis=1)
#     r0 = sqrt((data.x[0]-xc0)**2 +(data.x[1] -yc0)**2).mean()
#     return xc0, yc0, r0
#
# lsc_data  = odr.Data(row_stack([x, y]), y=1)
# lsc_model = odr.Model(f_3b, implicit=True, estimate=calc_estimate, fjacd=jacd, fjacb=jacb)
# lsc_odr   = odr.ODR(lsc_data, lsc_model)
# lsc_odr.set_job(deriv=3)
# lsc_odr.set_iprint(iter=1, iter_step=1)
# lsc_out   = lsc_odr.run()
#
# xc_3b, yc_3b, R_3b = lsc_out.beta
# Ri_3b       = calc_R(xc_3b, yc_3b)
# residu_3b   = sum((Ri_3b - R_3b)**2)

fmt = '%-22s  %10.5f  %10.5f  %10.5f  '
print(('\n%-22s' + ' %10s' * 3) % tuple('方法  Xc  Yc  Rc'.split()))
print('-' * (22 + 4 * (10 + 1)))
print(fmt % (method_1, xc_1, yc_1, R_1))
print('-' * (22 + 4 * (10 + 1)))
print(fmt % (method_2, xc_2, yc_2, R_2))
print('-' * (22 + 4 * (10 + 1)))
print(fmt % (method_3, xc_3, yc_3, R_3))
# 输出图
plt.close('all')


# def plot_all1(residu1=False):
#     plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
#     plt.axis('equal')
#
#     theta_fit = linspace(-pi, pi, 180)
#
#     x_fit1 = xc_1 + R_1 * cos(theta_fit)
#     y_fit1 = yc_1 + R_1 * sin(theta_fit)
#     plt.plot(x_fit1, y_fit1, 'k--', label=method_1, lw=2)
#     plt.plot([xc_1], [yc_1], 'gD', mec='r', mew=1)
#
#     # draw
#     plt.xlabel('x')
#     plt.ylabel('y')
#
#     # 数据
#     plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
#     plt.legend(loc='best', labelspacing=0.1)
#
#     # 标题
#     # plt.grid()
#     plt.title('Least Squares Circle')
#     # plt.savefig('%s_residu%d.png' % (basename, 2 if residu2 else 1))   # 保存
# plot_all1(residu1=True)
#
# def plot_all2(residu2=False):
#     plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
#     plt.axis('equal')
#
#     theta_fit = linspace(-pi, pi, 180)
#     x_fit2 = xc_2 + R_2 * cos(theta_fit)
#     y_fit2 = yc_2 + R_2 * sin(theta_fit)
#     plt.plot(x_fit2, y_fit2, 'k--', label=method_2, lw=2)
#     plt.plot([xc_2], [yc_2], 'gD', mec='r', mew=1)
#
#     # draw
#     plt.xlabel('x')
#     plt.ylabel('y')
#
#     # 数据
#     plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
#     plt.legend(loc='best', labelspacing=0.1)
#
#     # 标题
#     # plt.grid()
#     plt.title('Least Squares Circle')
#     # plt.savefig('%s_residu%d.png' % (basename, 2 if residu2 else 1))   # 保存
# plot_all2(residu2=True)
#
# def plot_all2b(residu2b=False):
#     plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
#     plt.axis('equal')
#
#     theta_fit = linspace(-pi, pi, 180)
#
#     x_fit2b = xc_2b + R_2b * cos(theta_fit)
#     y_fit2b = yc_2b + R_2b * sin(theta_fit)
#     plt.plot(x_fit2b, y_fit2b, 'k--', label=method_2b, lw=2)
#     plt.plot([xc_2b], [yc_2b], 'gD', mec='r', mew=1)
#
#     # draw
#     plt.xlabel('x')
#     plt.ylabel('y')
#
#     # 数据
#     plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
#     plt.legend(loc='best', labelspacing=0.1)
#
#     # 标题
#     # plt.grid()
#     plt.title('Least Squares Circle')
#     # pltsavefig('%s_residu%d.png' % (basename, 2 if residu2 else 1))   # 保存
# plot_all2b(residu2b=True)
#
# def plot_all3(residu3=False):
#     plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
#     plt.axis('equal')
#
#     theta_fit = linspace(-pi, pi, 180)
#
#     x_fit3 = xc_3 + R_3 * cos(theta_fit)
#     y_fit3 = yc_3 + R_3 * sin(theta_fit)
#     plt.plot(x_fit3, y_fit3, 'k--', label=method_3, lw=2)
#     plt.plot([xc_3], [yc_3], 'gD', mec='r', mew=1)
#
#     # draw
#     plt.xlabel('x')
#     plt.ylabel('y')
#
#     # 数据
#     plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
#     plt.legend(loc='best', labelspacing=0.1)
#
#     # 标题
#     # plt.grid()
#     plt.title('Least Squares Circle')
#     # plt.savefig('%s_residu%d.png' % (basename, 2 if residu2 else 1))   # 保存
# plot_all3(residu3=True)
#
# def plot_all3b(residu3b=False):
#     plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
#     plt.axis('equal')
#
#     theta_fit = linspace(-pi, pi, 180)
#
#     x_fit3b = xc_3b + R_3b * cos(theta_fit)
#     y_fit3b = yc_3b + R_3b * sin(theta_fit)
#     plt.plot(x_fit3b, y_fit3b, 'k--', label=method_3b, lw=2)
#     plt.plot([xc_3b], [yc_3b], 'gD', mec='r', mew=1)
#
#     # draw
#     plt.xlabel('x')
#     plt.ylabel('y')
#
#     # 数据
#     plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
#     plt.legend(loc='best', labelspacing=0.1)
#
#     # 标题
#     # plt.grid()
#     plt.title('Least Squares Circle')
#     # plt.savefig('%s_residu%d.png' % (basename, 2 if residu2 else 1))   # 保存
# plot_all3b(residu3b=True)


def plot_all(residu=False):
    plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
    plt.axis('equal')
    theta_fit = linspace(-pi, pi, 180)

    x_fit1 = xc_1 + R_1 * cos(theta_fit)
    y_fit1 = yc_1 + R_1 * sin(theta_fit)
    plt.plot(x_fit1, y_fit1, 'k--', label=method_1, lw=2)
    plt.plot([xc_1], [yc_1], 'gD', mec='r', mew=1)

    x_fit2 = xc_2 + R_2 * cos(theta_fit)
    y_fit2 = yc_2 + R_2 * sin(theta_fit)
    plt.plot(x_fit2, y_fit2, 'bo-', label=method_2, lw=2)
    plt.plot([xc_2], [yc_2], 'gD', mec='r', mew=1)

    x_fit3 = xc_3 + R_3 * cos(theta_fit)
    y_fit3 = yc_3 + R_3 * sin(theta_fit)
    plt.plot(x_fit3, y_fit3, 'r2-', label=method_3, lw=2)
    plt.plot([xc_3], [yc_3], 'gD', mec='r', mew=1)

    # x_fit3b = xc_3b + R_3b * cos(theta_fit)
    # y_fit3b = yc_3b + R_3b * sin(theta_fit)
    # plt.plot(x_fit3b, y_fit3b, 'y3-', label=method_3b, lw=2)
    # plt.plot([xc_3b], [yc_3b], 'gD', mec='r', mew=1)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
    plt.legend(loc='best', labelspacing=0.1)

    # 标题
    # plt.grid()
    # plt.title('Least Squares Circle')
    # plt.savefig('%s_residu%d.png' % (basename, 2 if residu2 else 1))   # 保存


plot_all(residu=True)

plt.show()
