import matplotlib
matplotlib.use("Agg")
import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import seaborn as sns
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.integrate import quad, quadrature,nquad
from scipy.stats import linregress
import time
from math import *
import os
import sys
import multiprocessing
from joblib import Parallel, delayed
e=0.0000000000000001 # number for float comparison
#%precision 4
npr.seed(501)
sys.setrecursionlimit(2000)
num_cores=multiprocessing.cpu_count()
#---------------------------------------------------------------------------------
#define parameters
Zmax,Zmin,d,R=100,-100,2475,675
Theta_min,Theta_max=0,np.pi*85/180
Emin,Emax=0,100
disk1=0
disk2=tan(Theta_max)*(d+Zmax-Zmin)+R
Area=np.pi*(disk2**2-disk1**2)
N=10000
run_n=8
#--------------------------------------------------------------------------------
# neccessary functions

#muon flux
P1=0.102573
P2=-0.068287
P3=0.958633
P4=0.0407253
P5=0.817285

def cos_m(th):
    if(cos(th)**2+P1**2+P2*(cos(th)**P3)+P4*(cos(th)**P5)<0): print(th)
    temp1=sqrt(cos(th)**2+P1**2+P2*(cos(th)**P3)+P4*(cos(th)**P5))
    temp2=sqrt(1+P1**2+P2**2+P4**2)
    p=temp1/temp2
    return p

def probability_density(E,Theta):
    th=Theta
    t1=E*(1+3.64/(E*(cos_m(th)**1.29)))
    t2=1/(1+1.1*E*cos_m(th)/115)+0.054/(1+1.1*E*cos_m(th)/850)
    p=0.14*(t1**-2.7)*t2*sin(th)   # the sin(th) is added to account for the solid angle (Jacobin) so the flux per Radian instead of per sr
    return p

max_flux=0.0043

#random generation function
def Rand1(func):
    run=0
    while(run<2000):
        u=npr.uniform(Emin,Emax)
        v=npr.uniform(Theta_min,Theta_max)
        p=npr.rand()
        pmax=func(Emin+e,Theta_max)*(1+0.0001) #assumes an absolutly increasing function of angle ( only good for high energies >100)
        if p<=func(u,v)/pmax: return [u,v]
    print("Warning: maximum recursion limit, fake event created!")
    return [-100,-100]
# intersection with can

def intersection(x,y,z,vx,vy,vz,Zmax,Zmin,R,d):
    k=-d/vz
    xn=x+vx*k
    yn=y+vy*k
    if(xn**2 +yn**2 <=R**2): return [1,xn,yn,Zmax] # intersection with top surface
    else:
        a=vx**2+vy**2
        b=2*(vx*x+vy*y)
        c=x**2 +y**2-R**2
        delta=b**2 -4*a*c
        if(delta<0 or a==0): return [0,0,0,0]
        else: 
            k1=(-b+sqrt(delta))/(2*a)
            k2=(-b-sqrt(delta))/(2*a)
            if(d+Zmax+k1*vz<=Zmax and d+Zmax+k1*vz>=Zmin): return [1,x+k1*vx,y+k1*vy,d+Zmax+k1*vz]
            elif(d+Zmax+k2*vz<=Zmax and d+Zmax+k2*vz>=Zmin): return [1,x+k2*vx,y+k2*vy,d+Zmax+k2*vz]
            else: return [0,0,0,0]


#-------------------------------------------------------------------------------------------
# load energy loss file

loss=pd.read_csv("loss.txt",sep='\s+',skiprows=26,header=None)
loss.columns=['E', 'momentum', 'a', 'bb', 'bp','bph','br','rate','range','delta','beta','rate2']

#change units
loss['range']/=100.0
loss['E']/=1000.0
loss['rate']*=(100.0/1000)*1.03 # additional factor for the density of sea water

slop, intercept, r_value, p_value, std_err =linregress(loss.loc[108:,'E'],loss.loc[108:,'rate'])
def rang(E):
    ind=loss['E'].searchsorted(E,side='right')
    ra=(2*loss.loc[ind-1,'rate']+(loss.loc[ind,'rate']-loss.loc[ind-1,'rate'])*(E-loss.loc[ind-1,'E'])/(loss.loc[ind,'E']-loss.loc[ind-1,'E']))/2
    print(E,loss.loc[ind,'E'],loss.loc[ind-1,'E'],loss.loc[ind,'rate'],loss.loc[ind-1,'rate'],ra)
    d=loss.loc[ind-1,'range']+(E-loss.loc[ind-1,'E'])/ra
    return d
def Rate(E):
    ind=loss['E'].searchsorted(E,side='left')
    if(ind==131): return (slop*E+intercept)
    #ra=(2+loss.loc[ind-1,'rate']+(loss.loc[ind,'rate']-loss.loc[ind-1,'rate'])*(d-loss.loc[ind-1,'range'])/(loss.loc[ind,'range']-loss.loc[ind-1,'range']))/2
    return loss.loc[ind-1,'rate']
def los(E,d):
    xtot=d
    while(xtot>0):
        rate=Rate(E)
        dx=min(xtot,0.2*E/rate)
        E-=rate*dx
        xtot-=dx
        if(E<0.1): break
    return E
def threshold(d):
    ind=loss['range'].searchsorted(d,side='right')
    return loss.loc[ind-1,'E']
#--------------------------------------------------------------------------------------------

# define functions simulation
r1m=tan(Theta_max)*(d)-R
r2m=tan(Theta_max)*(d+Zmax-Zmin)+R
norm=(r2m**2-r1m**2)/(disk2**2-	disk1**2)
#print("normalization for surface disk",norm)
def make_muon2(Rmax,run):
    E,th=Rand1(probability_density)
    ph=npr.uniform(0,2*np.pi)
    phi=npr.uniform(0,2*np.pi)
    z=0
    vx=sin(th)*cos(phi)
    vy=sin(th)*sin(phi)
    vz=-cos(th)
    r1=max(0,tan(th)*d-R)
    r2=tan(th)*(d+Zmax-Zmin)+R
    r1=r1/Rmax
    r2=r2/Rmax
    r=disk2*sqrt(npr.uniform(r1**2,r2**2))
    rej=npr.uniform(0,1)
    x=r*cos(ph)
    y=r*sin(ph)
    if(rej>(r2**2-r1**2)/norm): return make_muon2(Rmax,run+1)
    return [x,y,z,vx,vy,vz,E,th,phi,run]

def make_muon_at_detector():
    npr.seed()
    run_n=0
    run_id=0
    while(run_n<10000):
        temp=make_muon2(disk2,0)
        run_id+=temp[9]+1
        temp2=intersection(temp[0],temp[1],temp[2],temp[3],temp[4],temp[5],Zmax,Zmin,R,d)
        if(temp2[0]):
            dist=(temp2[1]-temp[0])**2+(temp2[2]-temp[1])**2+(Zmax+d-temp2[3])**2
            dist=sqrt(dist)
            newE=los(temp[6],dist)
            if newE>0.1:
                return [temp2[1],temp2[2],temp2[3],temp[3],temp[4],temp[5],newE,run_id]
        run_n+=1
    return [-5,-5,-5,-5,-5,-5,-5,-5]
#---------------------------------------------------------------------------------

# method3: monte carlo with geometric optimization and energy loss

 #set energy threshhold
Emin= threshold(min(d,sqrt(d**2+disk1**2)))-5
while(los(Emin,d)>0.1): Emin-=5
Emax=Emin*20
#print("ratio",Emin,probability_density(Emax,0)/probability_density(Emin,0))
flux,err = nquad(probability_density,[[Emin,Emax],[0,Theta_max]],full_output=False)
#print("hit rate is :",flux/(probability_density(Emin,0)*Theta_max*(Emax-Emin)))
flux*=10000*2*np.pi
err*=10000*2*np.pi
#print("energy flux",flux)
#print("energy flux error",err)
#-----------------------------------------------------------------
  #start of simulation
if __name__ == "__main__":
    t0=time.time()
    sampl4= Parallel(n_jobs=min(num_cores,14))(delayed(make_muon_at_detector)() for i in range(N))
    t1=time.time()
    print("time",t1-t0)


#-----------------------------------------------------------------
# store the results of method 3
result3=pd.DataFrame(sampl4,columns=['x','y','z','vx','vy','vz','E','num_calls'])
result3.to_csv("result_{run_number}.csv".format(run_number=run_n),sep='\t')
methods.to_csv("efficiency_{run_number}.csv".format(run_number=run_n),sep='\t')
fig=plt.figure(figsize=(20,8))
ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((1,2), (0, 1), colspan=1, rowspan=1)
sns.distplot(result3["E"],bins='auto',kde=False,ax=ax1,hist_kws={"histtype": "step"})
sns.distplot(result3["vz"],bins='auto',kde=False,ax=ax2,hist_kws={"histtype": "step"})
fig2=plt.figure(figsize=(12,8))
plt.axes(projection='3d')
ax_pos = plt.subplot2grid((1,1), (0, 0),projection='3d')
ax_pos.scatter(result3[:]["x"],result3[:]["y"],result3[:]["z"],marker='o')
fig.savefig("hists1.png")
fig2.savefig("3d1.png")
#plt.show()
