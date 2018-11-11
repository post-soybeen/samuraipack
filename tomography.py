#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:50:52 2018

@author: matsuo
"""

import numpy as np
import scipy as sc
import scipy.signal as sig
import plato.signal as signal
import os
import sys
import time
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import ticker 
import pandas as pd
import loadexcel as lo
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import jit

# old style
matplotlib.rcParams['xtick.direction'] = 'in'
#matplotlib.rcParams['xtick.top'] = True
#matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.direction'] = 'in'
#matplotlib.rcParams['ytick.right'] = True
#matplotlib.rcParams['ytick.minor.visible'] = True

# my fabalit
matplotlib.rcParams['font.family'] = 'DejaVu Sans'#'Times New Roman'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.linewidth'] = 3
matplotlib.rcParams['xtick.major.size'] = 6
matplotlib.rcParams['xtick.minor.size'] = 4
matplotlib.rcParams['xtick.major.width'] = 2.0
matplotlib.rcParams['xtick.minor.width'] = 1.0
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['lines.markersize'] = 10
matplotlib.rcParams['lines.markeredgewidth'] = 1.0
matplotlib.rcParams['ytick.major.size'] = 6
matplotlib.rcParams['ytick.minor.size'] = 4
matplotlib.rcParams['ytick.major.width'] = 2.0
matplotlib.rcParams['ytick.minor.width'] = 1.0
#matplotlib.rcParams.update(matplotlib.rcParamsDefault) #デフォルト値に戻す

##---time start end-------------
start = 200000#200000
end = 500000#500000
diff = end - start 
d=1e-6
nfft=10000
fn = 3
#size = os.path.getsize(header[0][1])#全体サイズ
##------------------------------



class tomography:
    #----------------------------------
    br_1=121 #1面データ個数
    data_num = 8*121 #１面データサイズ
    data_1 = 8 #データ一つあたりのサイズ
    #br_sum = size/data_num
    #----------------------------------
    def __init__(self,file,start=200000,end=500000,als=-1,ale=-1,d=1e-6,nfft=10000,nperseg=10000):
        if len(file[0])==600000:
            self.start,self.end = 0,600000
        else: 
            self.start,self.end = start,end
        self.als = als
        self.ale = ale
        if als == -1:
            self.als=0
        if ale == -1:
            self.ale=self.end-self.start
        self.file=file
        self.lenfile=len(file)
        self.datasize=len(file[0])
        self.d=d
        self.f=1/d
        self.nfft=nfft
        self.nfreq=self.nfft//2+1
        self.nperseg=nperseg
        self.grid=(11,11)
        self.ff=self.file.reshape(self.lenfile,self.datasize,*self.grid) #fixedfile
        #@jit
    def xy(self,x,y):
        """
        グリッドの位置番号を返す
        (1~11,1~11)と入力し、0~120の値を返す
        """
        xy = (x-1)+11*(y-1)
        return xy
    
    def xy_r(self,x,y):
        """
        グリッドの中心からの位置を番号で
        (-5~5,-5~5)で入力し、位置番号を返す
        """
        xy_r = x+11*y+60
        return xy_r
        
    
    #@jit
    def xy_reverse(self,num):
        """
        グリッドの座標を返す
        0~120の位置番号を(0-10,0~10)で返す
        """
        x,y=num//11,num%11
        return x,y
    
    #@jit
    def make_frequency(self):
        f,_ = sig.welch(self.file[0,:,self.xy(1,1)],fs=1/self.d,nfft=self.nfft,nperseg=self.nperseg,scaling='density')
        return f
    
    def totalemission(self,s=-1,e=-1,filelist=[]):
        """
        全発光量の合計プロファイルを示す
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        if filelist==[]:
            total = np.zeros(self.datasize)
            total = np.sum(self.file[0,:,:],axis=1)
        else:
            fn = len(filelist)
            total = np.zeros(fn,self.datasize)
            for i,j in enumerate(filelist):
                total[i] = np.sum(self.file[j,:,:],axis=1)
        return total
        
    
    def average_profile(self,filelist=[]):
        """
        全グリッドの平均プロファイルを作る
        ファイルリストの数によってそれぞれのプロファイルをつくる
        """
        if filelist==[]:
            ave = np.zeros(self.datasize)
            ave = np.average(self.file[0,:,:],axis=1)
        else:
            fn = len(filelist)
            ave = np.zeros(fn,self.datasize)
            for i,j in enumerate(filelist):
                ave[i] = np.average(self.file[j,:,:],axis=1)
        return ave
    
    #@jit
    def row_average(self,s=-1,e=-1,filelist=[],col=False):
        """
        中心軸の平均発光プロファイルを示す
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        file = self.file.reshape(self.lenfile,self.datasize,11,11)
        if filelist==[]:
            ave = np.zeros(11)
            if col==False:
                ave = np.average(file[0,s:e,5,:],axis=0)
            else:
                ave = np.average(file[0,s:e,:,5],axis=0)
        else:
            fn = len(filelist)
            ave = np.zeros((fn,11))
            if col==False:
                for i,j in enumerate(filelist):
                    ave[i] = np.average(file[j,s:e,5,:],axis=0)
            else:
                for i,j in enumerate(filelist):
                    ave[i] = np.average(file[j,s:e,:,5],axis=0)
        return ave
    
    def pickup(self,x=6,y=6,s=-1,e=-1,filelist=[]):
        """
        ある点をピックアップする。選ぶグリッドは1からカウント
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        file = self.file.reshape(self.lenfile,self.datasize,11,11)
        if filelist==[]:
            return file[0,s:e,x-1,y-1]
        else:
            a=np.zeros((len(filelist),e-s))
            for i,j in enumerate(filelist):
                a[i] = file[0,s:e,x-1,y-1]
            return a

    
    #@jit
    def psd_grid(self,x,y,s=-1,e=-1,filelist=[]):
        """
        各グリッドのPSD
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        if filelist==[]:
            _,psd = sig.welch(self.file[0,s:e,self.xy(x,y)],fs=1/self.d,nfft=self.nfft,nperseg=self.nperseg,scaling='density')
            return psd
        else:
            flen = len(filelist)
            psd = np.zeros((flen,self.nfreq))
            for i,j in enumerate(filelist):
                _,psd[i] = sig.welch(self.file[j,s:e,self.xy(x,y)],fs=1/self.d,nfft=self.nfft,nperseg=self.nperseg,scaling='density')
            return psd.sum(axis=0)/flen
    
    def psd_field(self,x,y,s=-1,e=-1,filelist=[]):
        """
        一面の内ある範囲のpsd
        x=[-5~5],y=[-5~5] (numpy array) 
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        flen = self.lenfile
        psd = np.zeros((len(x),len(y),flen,self.nfreq))
        for i in x:
            for j in y:
                for k in range(flen):
                    _,psd[np.where(x==i),np.where(y==j),k] = sig.welch(self.file[k,s:e,self.xy_r(i,j)],fs=1/self.d,nfft=self.nfft,nperseg=self.nfft,scaling='density')          
        psd = psd.sum(axis=2)/flen
        X,Y = np.meshgrid(x,y)
        return X,Y,psd
    
    def psd_sum_brane(self,s=-1,e=-1,filelist=[],separate=False):
        """
        発光量の合計に対するpsdを出力
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        if filelist==[]:
            file = np.sum(self.file[0,:,:],axis=1)
            f,psd = sig.welch(file[0,s:e],fs=1/self.d,nfft=self.nfft,nperseg=self.nperseg,scaling='density')
            return f,psd
        else:
            psd = np.zeros((len(filelist),self.nfreq))
            for i,j in enumerate(filelist):
                file = np.sum(self.file[j,:,:],axis=1)
                f,psd[i] = sig.welch(file[s:e],fs=1/self.d,nfft=self.nfft,nperseg=self.nperseg,scaling='density')
            if separate==False:
                psd = np.average(psd[:,:],axis=0)
                return f,psd
            else:
                return f,psd
            
    #@jit
    def psd_max_xy(self,s=-1,e=-1,filelist=[]):
        """
        PSDが最大となるグリッドを周波数ごとの配列で出力、最後に各グリッドのPSD配列を出力
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        psd_f=np.empty((11,11,self.nfreq))
        for y in range(1,12):
            for x in range(1,12):
                psd_f[y-1,x-1,s:e] = self.psd_grid(x,y)#[np.where(f==ch_freq)])            
        maxa=[]
        for i in range(5001):
            maxa = np.append(maxa,np.argmax(psd_f[:,:,i]))
        x_f,y_f = maxa%11+1,maxa//11+1
        return x_f.astype(int),y_f.astype(int),psd_f     #x_f,y_f=[6,7,6,7,7,7,7],[5,8,7,8,8,8,7,5]みたいな感じ
    
    #@jit
    def coh_phase_grid(self,x,y,x1,y1,s=-1,e=-1):
        """
        各グリッドのコヒーレンスと位相を計算
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        flen = len(self.file)
        crs = np.zeros((flen,self.nfreq),dtype="complex128")
        pxx = np.zeros((flen,self.nfreq))
        pyy = np.zeros((flen,self.nfreq))
        for i in range(flen):#各クロススペクトル、パワースペクトルを計算、コヒーレンスとフェーズを返す
            _,crs[i] = sig.csd(self.file[i,s:e,self.xy(x,y)],self.file[i,s:e,self.xy(x1,y1)],fs=1/self.d,nperseg=self.nfft,nfft=self.nperseg,scaling='density')
            _,pxx[i] = sig.welch(self.file[i,s:e,self.xy(x,y)],fs=1/self.d,nfft=self.nfft,nperseg=self.nperseg,scaling='density')
            _,pyy[i] = sig.welch(self.file[i,s:e,self.xy(x1,y1)],fs=1/self.d,nfft=self.nfft,nperseg=self.nperseg,scaling='density')
        crs=crs.sum(axis=0)/flen
        pxx=pxx.sum(axis=0)/flen
        pyy=pyy.sum(axis=0)/flen
        coh = abs(crs)**2/pxx/pyy
        phase = signal.xphase(crs)
        return coh,phase
    
    #def coh_phase(Crs,Pxx,Pyy):
    #    coh = abs(Crs)**2/Pxx/Pyy
    #    phase = signal.xphase(Crs)
    #    return coh,phase
    
    #@jit
    def coh_phase_field(self,x,y,s=-1,e=-1):
        """
        各周波数の一面のコヒーレンス、位相、位相差イメージを計算
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        coh_f = np.zeros((11,11,self.nfreq))
        phase_f = np.zeros((11,11,self.nfreq))
        for y1 in range(1,12):
            for x1 in range(1,12):
                coh_f[y1-1,x1-1,s:e],phase_f[y1-1,x1-1,s:e] = self.coh_phase_grid(x,y,x1,y1)
        return coh_f,phase_f
    
    #@jit
    def crs_phase_grid(self,x,y,x1,y1,s=-1,e=-1):
        """
        各グリッドのクロススペクトルと位相差を計算
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        flen = len(self.file)
        crs = np.zeros((flen,self.nfreq),dtype="complex128")    
        for i in range(flen):#各クロススペクトルとフェーズを返す
            _,crs[i] = sig.csd(self.file[i,s:e,self.xy(x,y)],self.file[i,s:e,self.xy(x1,y1)],fs=1/self.d,nperseg=self.nperseg,nfft=self.nfft,scaling='density')
        crs=crs.sum(axis=0)/flen
        phase = signal.xphase(crs)
        return crs,phase
    
    #@jit
    def phase_field(self,x,y,s=-1,e=-1):
        """
        各周波数の一面の、位相差を計算
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        phase_f = np.zeros((11,11,self.nfreq))
        for y1 in range(1,12):
            for x1 in range(1,12):
                _,phase_f[y1-1,x1-1,s:e] = self.crs_phase_grid(x,y,x1,y1)
        return phase_f
    
    #@jit
    def Power_cos(self,psd_f,phase_f):
        Pow = psd_f*np.cos(phase_f)
        return Pow
    
    #@jit
    def Pow_make(self,ch_freq):
        freq=np.where(self.f==ch_freq)[0][0]
        x,y,psd = self.psd_max_xy()
        phase = self.phase_field(self.file,x[freq],y[freq])
        Pow = self.Power_cos(psd,phase)
        return Pow
    
    def moment_basis_function(self,m_min=0,m_max=4,delta=1,filelist=[]):
        """
        θモーメントの基底関数の生成
        """
        bline = np.arange(m_min,m_max+delta,delta)
        basis = np.zeros((len(bline)*2-1,*self.grid))
        r1 = np.zeros(self.grid)
        r2 = np.zeros(self.grid)
        a = np.arange((-self.grid[0]+1)//2,self.grid[1]//2+1,1)
        r1[:]=a
        r2=np.transpose(r1)
        theta = np.arctan2(r2,r1)
        for i,j in enumerate(bline):
            if i==0:
                basis[0]=np.cos(j*theta)/np.sqrt(np.sum(np.cos(j*theta)**2))
            else:
                basis[2*i]  = np.sin(j*theta)/np.sqrt(np.sum(np.sin(j*theta)**2))
                basis[2*i-1]= np.cos(j*theta)/np.sqrt(np.sum(np.cos(j*theta)**2))
        return basis
        
    def moment(self,fnl=[],s=-1,e=-1,m_min=0,m_max=5,delta=1,reshort=True):
        """
        θモーメントの算出
        """
        if s==-1:
            s=self.als
        if e==-1:
            e=self.ale
        diff = e-s
        basis = self.moment_basis_function(m_min=m_min,m_max=m_max,delta=delta)
        lb    = len(basis)
        P     = np.zeros((lb,lb)) 
        for i in range(lb):
            for j in range(lb):
                P[i,j] = np.sum(basis[i]*basis[j]) 
        """
        一面でできたら進む
        if fnl==[]:
            bvector = np.zeros((self.lenfile,diff,lb,*self.grid))
        else:
            l=len(fnl)
            bvector = np.zeros((l,diff,lb,*self.grid)
        """
        #一面で試す　sinθ分で
        file = self.file.reshape((self.lenfile,diff,*self.grid))
#        da = self.datasize #diffはdaだった
        bvector = np.zeros((diff,lb))
        for i in range(diff):
            for j in range(lb):
                bvector[i,j]=np.sum(file[0,i]*basis[j])
            if i%10000==0:
                print("vectorize process is = "+str(i))
        gamma=0
        I = np.identity(lb)
        Pg = P+gamma*I
        at = np.zeros((diff,lb))
        for i in range(diff):
            at[i] = np.linalg.solve(Pg,bvector[i])
            if i%10000==0:
                print("solving process is = "+str(i))
        #再構成
        reconst=np.zeros([*self.grid])
        for i in range(lb):
            reconst += at[0,i]*basis[i]
        
        if reshort == True:
            return at
        else:
            return basis,P,bvector,at,reconst
        
        
        """
        for i,j in enumerate(fnl):#まずはsinθだけやってみる
            for k in range(diff):
                for l in range(lb):
                    bvector[i,l,k] = basis[2]*file[j,2]
        """
    def tomopixel(self,t,filenumber=0,rowvalue=False,colvalue=False):
        """
        トモグラフィーのピクセル画像と中心軸の値を出力する
        """
        r,c = np.zeros(11),np.zeros(11)
        if rowvalue==True:
            r=self.ff[filenumber,t,5,:]
        if colvalue==True:
            c=self.ff[filenumber,t,:,5]
        return self.ff[filenumber,t],r,c
        
    
if __name__ == '__main__':
    
#ロード
    header1=[]
    header2=[]
    fnl=[0,1,2]
    header1.append(lo.header_generator(power=3,magfield=1500,date=170712,home=True))
    header1.append(lo.header_generator(power=4,magfield=1500,date=170712,home=True))
    header1.append(lo.header_generator(power=5,magfield=1500,date=170712,home=True))
    header1.append(lo.header_generator(power=6,magfield=1500,date=170712,home=True))
#    header2.append(lo.header_generator(power=3,magfield=1500,date=170712,home=True,group="8"))
#    header2.append(lo.header_generator(power=4,magfield=1500,date=170712,home=True,group="8"))
#    header2.append(lo.header_generator(power=5,magfield=1500,date=170712,home=True,group="8"))
#    header2.append(lo.header_generator(power=6,magfield=1500,date=170712,home=True,group="8"))
#    file6=[0,0,0,0]
#    file8=[0,0,0,0]
    tomo=[0,0,0,0]
#    tomo8=[0,0,0,0]
    for i in range(4):
 #       file6[i] = lo.file_loader(header1[i][1:],stime=0,etime=600000,file_number_list=fnl)
#        file8[i] = lo.file_loader(header2[i][1:],stime=0,etime=600000,file_number_list=[0])
        tomo[i]  = tomography(file6[i]) #インスタンスを更新
#        tomo8[i] = tomography(file8[i])
        
##base image
#    basis = tomo[0].moment_basis_function(m_max=5,delta=1)
#    fig,axes=plt.subplots(2,2)
#    axes[0,0].imshow(basis[4],origin="low")
#    axes[0,1].imshow(basis[5],origin="low")
#    axes[1,0].imshow(basis[6],origin="low")
#    axes[1,1].imshow(basis[7],origin="low")
#    plt.show()
#    
#    
##reconstruct image and raw image    
#    at = tomo[0].moment(m_max=5,delta=1)
#    fig,ax = plt.subplots(1,2)
#    ax[0].imshow(e,origin="low")
#    ax[1].imshow(file[0][0,0].reshape(11,11),origin="low")
#    plt.show()
#    
#    
#    modeat = np.zeros((7,300000))
#    modeat[0] = np.sqrt(np.sum(at[:,0:1]**2,axis=1))
#    modeat[1] = np.sqrt(np.sum(at[:,1:3]**2,axis=1))
#    modeat[2] = np.sqrt(np.sum(at[:,3:5]**2,axis=1))
#    modeat[3] = np.sqrt(np.sum(at[:,5:7]**2,axis=1))
#    modeat[4] = np.sqrt(np.sum(at[:,7:9]**2,axis=1))
#    modeat[5] = np.sqrt(np.sum(at[:,9:11]**2,axis=1))

#モードごとの周波数変化
#    psdmo = np.zeros((7,5001))
#
#    for i in range(6):
#         f,psdmo[i] = sig.welch(modeat[i,200000:300000],fs=100000,nfft=10000,nperseg=10000)
#    
#    plt.figure()
#    for i in range(6):
#        plt.semilogy(f[0:151],psdmo[i,0:151],label="m="+str(i))
#    plt.xticks(np.arange(0,1800,300),("0","3","6","9","12","15"))
#    plt.xlabel("frequency [kHz]",fontsize=25)
#    plt.ylabel("Intensity (a.u.)",fontsize=25)
#    plt.title("azimuthal moment",fontsize=35)
#    plt.legend(loc="upper right")
#    plt.tight_layout()
#    plt.show()
    
    """プロットモジュール"""
    """2D image"""
#    fig,axes = plt.subplots(1,4,sharey=True,figsize=(20,6))
#    for i in range(4):
#        plt.sca(axes[i])
#        im=plt.imshow(tomo[i].ff[0,400000],clim=(0,35))
#        plt.title(str(i+3)+"kW",fontsize=50)
##        plt.ylim([-8,8])
#        plt.yticks([0,5,10],["-8","0","8"],fontsize=30)
#        xt=[0,5,10]
#        plt.xticks(xt,["-8","0","8"],fontsize=30)
#        plt.xlabel("[cm]",fontsize=30)
#        if i==0:
#            plt.ylabel("[cm]",fontsize=35)
#        plt.suptitle("An example",fontsize=50)
#        cax = make_axes_locatable(axes[i]).append_axes("right",0.2, pad=0.05)
#        cb  = plt.colorbar(im,cax=cax)
#        cb.ax.set_yticklabels(cb.ax.get_yticklabels(),fontsize=25)
#    plt.tight_layout(rect=[0,0,1,0.87])
#    plt.show()
    
    """average profile"""
#    fig,axes = plt.subplots(1,4,sharey=True,figsize=(20,6))
#    for i in range(4):
#        plt.sca(axes[i])
#        plt.plot(tomo[i].average_profile())
#        plt.title(str(i+3)+"kW",fontsize=50)
#        plt.ylim([0,40])
#        plt.yticks([0,20,40],fontsize=25)
#        xt=[0,200000,400000,600000]
#        plt.xticks(xt,[str(i//1000) for i in xt],fontsize=30)
#        plt.xlabel("time [ms]",fontsize=30)
#        if i==0:
#            plt.ylabel("Intensity",fontsize=35)
#        plt.suptitle("Average profile",fontsize=50)
#    plt.tight_layout(rect=[0,0,1,0.87])
#    plt.show()
    """row average"""
#    fig,axes = plt.subplots(1,4,sharey=True,figsize=(20,6))
#    for i in range(4):
#        plt.sca(axes[i])
#        plt.plot(tomo[i].row_average())
#        plt.title(str(i+3)+"kW",fontsize=50)
#        plt.ylim([0,30])
#        plt.yticks([0,15,30],fontsize=35)
#        xt=[0,5,10]
#        plt.xticks(xt,[-8,0,8],fontsize=35)
#        plt.xlabel("[cm]",fontsize=35)
#        if i==0:
#            plt.ylabel("Intensity",fontsize=35)
#        plt.suptitle("Average profile of center row",fontsize=60)
#    plt.tight_layout(rect=[0,0,1,0.9])
#    plt.show()
    """example 3point in time-series data"""
##    fig,axes = plt.subplots(1,4,sharey=True,figsize=(20,6))
#    for i in range(4):
#        plt.figure()#あとで入れた
##        plt.sca(axes[i])
#        st=300000;en=301000
#        inten=np.zeros((3,en-st))
#        inten[0]= tomo[i].pickup(x=6,y=8,s=st,e=en)
##        inten[1]= tomo[i].pickup(x=8,y=6,s=st,e=en)
#        inten[2]= tomo[i].pickup(x=5,y=5,s=st,e=en)
#        plt.plot(inten[0],label="x=6,y=8")
##        plt.plot(inten[1],label="x=8,y=6",alpha=0.8)
#        plt.plot(inten[2],label="x=5,y=5",alpha=0.7)
#        plt.legend(fontsize=20,loc="upper right")
##        plt.title(str(i+3)+"kW",fontsize=35)
#        plt.ylim([5,44])
#        plt.yticks([10,20,30,40],fontsize=25)
#        xt  = np.linspace(st,en,3)
#        xt1 = np.linspace(0,en-st,3)
#        plt.xticks(xt1,[str(i/1000) for i in xt],fontsize=25)
#        plt.xlabel("time [ms]",fontsize=30)
##        if i==0:
##            plt.ylabel("Intensity",fontsize=30)
##        plt.suptitle("A piece of row data",fontsize=40)
#        plt.tight_layout(rect=[0,0,1,0.92])
#    plt.show()
    """example of psd and errorbar"""
    s,e,n,fn = 100000,300000,10000,4
    def totalrange_psd(s,e,n,fn):
        N,pf  = (e-s)//n,n//2+1
        psd   = np.zeros((fn,n//2+1))
        psd_p = np.zeros(fn,dtype="int")
        psd_max = np.zeros(fn)
        for i in range(fn):
            psd[i]     = tomo[i].psd_grid(6,8,s,e,filelist=[0,1,2])
            psd_p[i]   = int(*sig.argrelmax(psd[i,20:150],order=100))+20
            psd_max[i] = psd[i,psd_p[i]]
        return psd,psd_p,psd_max
    
    def each_psd(s,e,n,fn):    
        N,pf  = (e-s)//n,n//2+1
        inten = np.zeros((fn,N,pf))
        inten_max,inten_p = np.zeros((fn,N)),np.zeros((fn,N),dtype="int")    
        for i in range(fn):
            for j in range(N):
                inten[i,j] = tomo[i].psd_grid(6,8,s+n*j,s+n*(j+1),filelist=[0,1,2])
                inten_p[i,j] = int(*sig.argrelmax(inten[i,j,20:150],order=100))+20
                inten_max[i,j] = inten[i,j,int(inten_p[i,j])]
        se = np.std(inten_max,axis=1)
        return se
    
    s,e,n,fn = 100000,200000,10000,4
    b,b_p,b_max = totalrange_psd(s,e,n,fn)    
    b_se = each_psd(s,e,n,fn)
    
    s,e,n,fn = 450000,550000,10000,4
    a,a_p,a_max = totalrange_psd(s,e,n,fn)    
    a_se = each_psd(s,e,n,fn)
    
    fig,axes = plt.subplots(2,2,sharex=True,sharey=True)
    plt.rcParams['xtick.labelsize']=25
    plt.rcParams['ytick.labelsize']=25
    for k in range(fn//2):
        for j in range(fn//2):
            i = 2*k+j
            ax = axes[k,j]
            ax.semilogy(b[i,0:150],label="100-200[ms]")
            ax.semilogy(a[i,0:150],label="450-550[ms]")
            ax.errorbar(b_p[i],b_max[i],yerr=b_se[i],fmt='ro',ecolor='g',markersize=5,elinewidth=3,capsize=10,label="SD(former)")
            ax.errorbar(a_p[i],a_max[i],yerr=a_se[i],fmt='ro',ecolor='m',markersize=5,elinewidth=3,capsize=10,label="SD(later)")
            ax.legend(fontsize=12)
            ax.set_xticks(np.linspace(0,150,6))
            ax.set_xticklabels(np.linspace(0,15,6,dtype="int"))
            ax.set_ylim([1e-5,1e-1])
            ax.set_title(str(i+3)+"kW",fontsize=40)
            if k==1:
                ax.set_xlabel("frequency [kHz]",fontsize=30)
            if j==0:
                ax.set_ylabel("Intensity (a.u.)",fontsize=25)
    plt.suptitle("peak difference of plasma, grid(6,8)",fontsize=45)
    plt.tight_layout()
    plt.show()
    
    """local point fourier transform"""
#    fig,axes = plt.subplots(8,4,sharey=True,sharex=True)
#    f = tomo[0].make_frequency()
#    for i in range(4):
##        plt.figure()
#        inten=np.zeros((8,5001))
#        for j in range(8):
#            plt.sca(axes[j,i])
#            N,n = 100000,50000
#            k,k1 = N+j*n,N+(j+1)*n
#            inten[j]=tomo[i].psd_grid(6,8,k,k1,filelist=[0,1,2])
#            plt.semilogy(inten[j],label=str(k//1000)+"-"+str(k1//1000)+"[ms]",alpha=0.7)
##            plt.legend(fontsize=10,loc="upper left")
#            ymin,ymax=-5,1
#            plt.ylim([pow(10,ymin),pow(10,ymax)])
#            plt.yticks(np.logspace(ymin,ymax,3).tolist(),fontsize=10)
#            xmin,xmax=0,150
#            plt.xlim([xmin,xmax])
#            xt=np.linspace(xmin,xmax,6).tolist()
#            plt.xticks(xt,[str(int(i//10)) for i in xt],fontsize=10)
#            
#            if i==0:
#                plt.ylabel(str(k//1000)+"-"+str(k1//1000)+"[ms]",fontsize=10)
#            if j==0:
#                plt.title(str(i+3)+"kW")
#            if j==7:
#                plt.xlabel("frequency [kHz]",fontsize=15)
            
#        inten[0]=tomo[i].psd_grid(6,8,200000,300000,filelist=[0,1,2])
##        inten[1]=tomo[i].psd_grid(6,8,300000,400000,filelist=[0,1,2])
#        inten[2]=tomo[i].psd_grid(6,8,400000,500000,filelist=[0,1,2])
#        plt.semilogy(inten[0],label="200-300[ms]")
#        plt.semilogy(inten[1],alpha=0.7,label="300-400[ms]")
#        plt.semilogy(inten[2],alpha=0.7,label="400-500[ms]")

#        xt=np.linspace(xmin,xmax,6).tolist()
#        plt.xticks(xt,[str(int(i//10)) for i in xt],fontsize=25)
#        plt.xlabel("frequency [kHz]",fontsize=30)
        #        plt.title(str(i+3)+"kW",fontsize=35)
#        if i==0:
#            plt.ylabel("Intensity (a.u.)",fontsize=30)
#        plt.suptitle("Local intensity ",fontsize=40)
#        plt.tight_layout(rect=[0,0,1,0.9])
#    plt.show()
    """psd of grid summension"""
#    sumg = np.zeros((4,5001))
#    for i in range(4):
#        f,sumg[i]=tomo[i].psd_sum_brane(filelist=[0,1,2])
#    fig,axes= plt.subplots(1,4,sharey=True,figsize=(20,6))
#    for i in range(4):
#        plt.sca(axes[i])
#        plt.semilogy(sumg[i],label=str(i+3)+"kW")
##        plt.title(str(i+3)+"kW",fontsize=35)
#        ymin,ymax=-5,1
#        plt.ylim([pow(10,ymin),pow(10,ymax)])
#        plt.yticks(np.logspace(ymin,ymax,3).tolist(),fontsize=25)
#        xmin,xmax=0,150
#        plt.xlim([xmin,xmax])
#        xt=np.linspace(xmin,xmax,6).tolist()
#        plt.xticks(xt,[str(int(i//10)) for i in xt],fontsize=25)
#        plt.xlabel("frequency [kHz]",fontsize=30)
#        if i==0:
#            plt.ylabel("Intensity (a.u.)",fontsize=30)
#        plt.legend(fontsize=20,loc="upper right")
#    plt.suptitle("Total emission PSD (ArI)",fontsize=40)
#    plt.tight_layout(rect=[0,0,1,0.94])
#    plt.show()
    
    
    """2d psd"""
        #one figure
#    grid=(6,8)
#    tmrevolution = np.zeros((4,30,5001))
#    for i in range(4):
#        for j in range(30):
#            tmrevolution[i,j]=tomo[i].psd_grid(*grid,s=j*20000,e=(j+1)*20000,filelist=[0,1,2])
#    
#    tim,tcax,tcb = [0 for i in range(4)],[0 for i in range(4)],[0 for i in range(4)]
#    fig,tax = plt.subplots(1,4,sharex=True,sharey=False,figsize=(15,20))#,subplot_kw={"projection":"3d"})
#    x,y = np.arange(0,150),np.arange(0,30)
#    X,Y = np.meshgrid(x, y)
#    for i in range(4):
#        tim[i] = tax[i].imshow(np.log(tmrevolution[i,:,:150].transpose()),cmap='bwr',origin="lower")
#        tcax[i] = make_axes_locatable(tax[i]).append_axes("right", size="10%", pad=0.05)
#        tim[i].set_clim(-10,-5)
#        tcb[i]  = plt.colorbar(tim[i],cax=tcax[i])
#        tcb[i].ax.set_yticklabels(tcb[i].ax.get_yticklabels(),fontsize=25)
##        tcb[i].ax.set_ylabel("Intensity (a.u.)")
#        tax[i].set_title("%dkW"% (i+3),fontsize=25)
#        tax[i].set_ylabel("frequency[kHz]",fontsize=25)
#        tax[i].set_yticks([0,30,60,90,120,150])
#        tax[i].set_yticklabels(["0","3","6","9","12","15"],fontsize=25)
#        tax[i].set_xlabel("time[ms]",fontsize=25)
#        tax[i].set_xticks([0,10,20,29])
#        tax[i].set_xticklabels(["0","200","400","600"],fontsize=17)
#        tim[i].set_clim(-10,-5)
#    plt.suptitle("Intensity (a.u.), grid={0}".format(grid),fontsize=30)
#    plt.tight_layout(rect=(0,0,1,0.95))
#    plt.show()

        #each figure
#    grid=(6,8)
#    tmrevolution = np.zeros((4,30,5001))
#    for i in range(4):
#        for j in range(30):
#            tmrevolution[i,j]=tomo[i].psd_grid(*grid,s=j*20000,e=(j+1)*20000,filelist=[0,1,2])
#    
#    tim,tcax,tcb = [0 for i in range(4)],[0 for i in range(4)],[0 for i in range(4)]
##    fig,tax = plt.subplots(1,4,sharex=True,sharey=False,figsize=(15,20))#,subplot_kw={"projection":"3d"})
#    x,y = np.arange(0,150),np.arange(0,30)
#    X,Y = np.meshgrid(x, y)
#    for i in range(4):
#        plt.figure(figsize=(6,6))
#        ax=plt.gca()
#        im = plt.imshow(np.log(tmrevolution[i,:,:150].transpose()),cmap='bwr',origin="lower",clim=(-10,-5),aspect='auto')
#        cax = make_axes_locatable(ax).append_axes("right", size="15%", pad=0.05)
#        cb  = plt.colorbar(im,cax=cax)
#        plt.yticks(np.arange(-5,-10),fontsize=25)
##        plt.ylabel("Intensity (a.u.)",fontsize=30,rotation=270)
#        plt.sca(ax)
##        plt.title("%dkW"% (i+3),fontsize=25)
#        plt.ylabel("frequency[kHz]",fontsize=30)
#        plt.yticks([0,30,60,90,120,150],["0","3","6","9","12","15"],fontsize=25)
#        plt.xlabel("time[ms]",fontsize=25)
#        plt.xticks([0,10,20,30],["0","200","400","600"],fontsize=25)
##        ax.set_aspect(0.3)
##    plt.suptitle("Intensity (a.u.), grid={0}".format(grid),fontsize=30)
#        plt.tight_layout(rect=(0,0,1,0.95))
#    plt.show()
    
    """tomograpy image with row intensity"""
#    fig,axes = plt.subplots(1,4,figsize=(10,6))
#    for i in range(4):
#        plt.sca(axes[i])
#        ax = plt.gca()
##        ax.set_aspect(1.)
#        tp,row,col = tomo[i].tomopixel(200000,0,rowvalue=True,colvalue=True)
#        im = ax.imshow(tp,cmap='bwr',origin="lower")
#        ax.set_aspect(1.)
#        ax.set_xticks([])
#        ax.set_yticks([])
#        #colorbar
#        cax = make_axes_locatable(ax).append_axes("right",0.1, pad=0.05)
#        cb  = plt.colorbar(im,cax=cax)
#        #sideplot
#        rax = make_axes_locatable(ax).append_axes("left",size=0.3 , pad=0.1)#,sharey=ax)
##        rax.hist(row,orientation="horizontal")
##        tax.hist(col,orientation="vertical")
#        rax.plot(row,np.arange(0,11,1))
#        rax.set_xlim([0,35])
#        rax.set_xticks([0,35])
#        rax.yaxis.tick_left()
#        rax.yaxis.set_label_position("left")
#        tax = make_axes_locatable(ax).append_axes("top", size=0.3 , pad=0.1)#,sharex=ax)
#        tax.plot(col)
#        tax.yaxis.tick_right()
#        tax.set_ylim([0,35])
#        tax.set_yticks([0,35])
#        tax.xaxis.tick_top()
#        tax.xaxis.set_label_position("top")
#        
#        rax.xaxis.set_tick_params(labelbottom=False)
#        tax.yaxis.set_tick_params(labelleft=False)
#        plt.tight_layout()
#        plt.show()
#        tcb[i].ax.set_ylabel("Intensity (a.u.)")
#        ax.set_title("%dkW"% (i+3),fontsize=20)
#        ax.set_xlabel("frequency[kHz]",fontsize=20)
#        ax.set_xticks([0,30,60,90,120,150])
#        ax.set_xticklabels(["0","3","6","9","12","15"],fontsize=25)
#        ax.set_ylabel("time[ms]",fontsize=20)
#        ax.set_yticks([0,15,30])
#        ax.set_yticklabels(["0","300","600"],fontsize=17)
#        ax.set_clim(-10,-5]