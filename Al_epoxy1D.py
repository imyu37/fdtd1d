# !/usr/bin/env python                    
# -*- coding:utf-8 -*-                    
# ****************************************
# author: YOng                            
# homepage:http://blog.163.com/wanyong_37/
# Copyright ©2011 YOng All right reserved 
# ****************************************

# 一维铝/环氧树脂声子晶体：主程序
import numpy as np
from constants import *
from lattice import *
import matplotlib.pyplot as plt

lam1 = E1*D1/(1+D1)/(1-2*D1)
muu1 = E1/(1+D1)/2.0          #铝的拉梅系数
lam0 = E0*D0/(1+D0)/(1-2*D0)                                           
muu0 = E0/(1+D0)/2.0          #环氧树脂的拉梅系数
                                        
cl1 = ((lam1+2*muu1)/rho1)**(1.0/2)
ct1 = (muu1/rho1)**(1.0/2)
cl0 = ((lam0+2*muu0)/rho0)**(1.0/2)
ct0 = (muu0/rho0)**(1.0/2)
c = np.array([[cl1, cl0, ct1, ct0]]).T
ca = max(max(c)) #8.0298e3
ci = min(min(c)) #32402 

fa = 3e4         #源的最大频率

lamin = ci/fa
dx = lamin/20
dt = dx/ca/2

en = int(round(a/dx)) #en为晶格的划分的单元格数
if en%2==1:
    en = en+1    # 确保en及en/2为偶数		

dx = a/en        # 修正空间步长
zson = int(p*en)      #网格规模
z0 = 10          #源的位置
tson = 2048*32   #计算时间步数

pl = zson - en/2

t0= 40        #源的中心；
spread = 1/fa/dt/np.sqrt(np.pi) #高斯源g=exp(-*((t0-T)/spread)^2)的
                                #最大频率fa对应的spread

cdt = cl0*dt
cm = (cdt-dx)/(cdt+dx)      #MurABC系数

vx0 = np.zeros([1, zson+1])
tau0 = np.zeros([1, zson+1])  #vx0,tau0代表先前值(更新前)
vx1 = np.zeros([1, zson+1])   #vx1,tau1代表当前值(更新后)
tau1 = np.zeros([1, zson+1])  

w = np.zeros([1, tson])
ss = np.zeros([1, tson])

E = E0*np.ones([1, zson+1])
rho = rho0*np.ones([1, zson+1])  
cc = np.zeros_like(E)
dd = np.zeros_like(E)
# 一维声子晶体的参数分布
for kk in range(1, zson): 
    if (2*kk)%en == 0:
        E[0, kk] = (E0 + E1)/2.0
        rho[0, kk] = (rho0 + rho1)/2.0
    else:
        for ii in range(0, int(round(p))):
            if en*(ii+1) - en/2 < kk < en*(ii+1):
                E[0, kk] = E1
                rho[0, kk] = rho1
                
cc = dt/dx*np.ones_like(cc)/rho[0,:]
dd = dt/dx*np.ones_like(dd)*E[0,:]

T = 0
for n in range(0, tson):
    ss[0, n] = np.exp(-((t0-T)/spread)**2) 
    for i in range(1, zson):
        vx1[0, i] = vx0[0, i] + cc[0, i]*(tau0[0, i] - tau0[0, i-1])
        
    vx1[0, 0] = vx0[0, 1] + cm*(vx1[0, 1] - vx0[0, 0])
    vx1[0, zson] = vx0[0, zson-1] + cm*(vx1[0, zson-1] - vx0[0, zson])
    
    vx1[0, z0-1] = vx1[0, z0-1] + ss[0, n-1] #加入激励
   
    for i in range(1, zson):
        tau1[0, i] = tau0[0, i]+dd[0, i]*(vx1[0, i+1]-vx1[0, i])
        
    tau1[0, 0] = tau0[0, 1] + cm*(tau1[0, 1] - tau0[0, 0])
    tau1[0, zson] = tau0[0, zson-1] +cm*(tau1[0, zson-1] - tau0[0, zson]) 
    
    vx0 = vx1
    tau0 = tau1
    
    w[0, T] = vx0[0, pl-1]          #提取数据；
    T=T+1 


fs = 1/dt                           #采样频率(Hz)
n = fs/fa
nn = int(round(tson/n))
tmp1 = np.fft.fft(w)
F = dt*np.ones_like(tmp1)*tmp1                
tmp2 = np.fft.fft(ss)
S = dt*np.ones_like(tmp2)*tmp2 
F1 = F[0,0:nn+1]         # G(k)=F(k)(k=1:N/2+1)
S1 = S[0,0:nn+1]
G1 = np.abs(F1)
S0 = np.abs(S1)
tmp3 = range(0,nn+1)
f = fs/tson/1.e3*np.ones_like(tmp3)*range(0,nn+1)     # 使频率轴f从零开始
pbg = G1/S0
plt.semilogy(f,pbg,'-')              # 绘制振幅-频率图
plt.xlabel('$\mathrm{frequency(10^{3}Hz)}$',fontsize=15)
plt.ylabel('$\mathrm{amplitude(dB)}$',fontsize=15)
plt.title('$\mathrm{Bandgaps \ of \ epoxy/Al \ 1D \ phononic \ crystal}$',fontsize=15)
plt.xlim([0,30])
plt.savefig('Al_epoxy 1D.png',dpi=600)
plt.show()
