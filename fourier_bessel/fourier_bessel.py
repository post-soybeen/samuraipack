#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:53:12 2017

@author: matsuo
"""

import numpy as np
import scipy as sc
import scipy.special as spe
import scipy.signal as sig
#from sympy import *  
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import sys
import os
import time
from numba import jit 


#==============================================================================
# constant

d = 1e-5
nfft=1000
lfft=nfft//2+1

#==============================================================================

#@jit
def mode_reverse(fbnumber):
    """
    フーリエベッセル関数の関数番号から
    mode,node,cos=1or0 を返す
    0~82までを入力、(0~6,0~8,0~1)を返す
    """
    if fbnumber<0 or 82<fbnumber:
        calable=False
        raise "please input within 0~82"
    else:
        calable=True
    if calable:
        if fbnumber-8 <= 0:
            return 0,fbnumber,0
        elif fbnumber-22 <= 0:
            n  = fbnumber-9
            n0 = n//2
            n1 = n%2
            return 1,n0,n1
        else:
            n  = fbnumber-23
            n0 = n//12+2
            n1 = (n-n//12*12)//2
            n2 = (n-n//12*12)%2
            return n0,n1,n2

#@jit
def mode_at(m=0,n=0,sin=0):
    """
    各モード、ノード、sin(or cos)成分の番地を返す
    sinはcos成分なら0、sin成分なら1
    """
    if sin<0 or 1<sin:
        raise ValueError("sin == 0 or 1, cos成分の時は 0、 sinの時は 1.")
    if m==0:
        if n<0 or 8<n:
            raise ValueError("0<=n<=8です")
        return n
    elif m==1:
        if n<0 or 6<n:
            raise ValueError("0<=n<=6です")
        return 10+n*2-1+sin
    elif m==2:
        if n<0 or 5<n:
            raise ValueError("0<=n<=5です")
        return 24+n*2-1+sin
    elif m==3:
        if n<0 or 5<n:
            raise ValueError("0<=n<=5です")
        return 36+n*2-1+sin
    elif m==4:
        if n<0 or 5<n:
            raise ValueError("0<=n<=5です")
        return 48+n*2-1+sin
    elif m==5:
        if n<0 or 5<n:
            raise ValueError("0<=n<=5です")
        return 60+n*2-1+sin
    elif m==6:
        if n<0 or 5<n:
            raise ValueError("0<=n<=5です")
        return 72+n*2-1+sin
    else:
        raise ValueError("0<=m<=6です")
def mode_se(mode=0,end=0):
    """
    各モードの最初、もしくは最後の番地を返す\n
    modeは0~6、endは0か1を取る
    """
    start_array=[0,9,23,35,47,59,71]
    end_array=[8,22,34,46,58,70,82]
    if 0<=mode and mode<=6:
        if end==0:
            return start_array[mode]
        elif end==1:
            return end_array[mode]
        else:
            raise ValueError("end == 0 or 1 です")
    else:
        raise ValueError("0<=m<=6です")

#------------ベッセル関数群-----------------------
def jn(n,x):
    return spe.jn(n,x)
def j0(x):
    return spe.jn(0,x)
def j1(x):
    return spe.jn(1,x)
def j2(x):
    return spe.jn(2,x)
def j3(x):
    return spe.jn(3,x)
def j4(x):
    return spe.jn(4,x)
def j5(x):
    return spe.jn(5,x)
def j6(x):
    return spe.jn(6,x)
#------------------------------------------------

#@jit
def x_zeros_comp():
    """
    ベッセル関数基底の元を作成
    """
    dx = 0.01#x軸走査時の刻み幅
    x_zeros = np.zeros((7,9))   #ゼロ点
    x_zero_k = np.zeros((7,9)) #大体のゼロ点位置    
    for n1 in range(7):  #ゼロ点付近の点を追加
        i=0;x=0;
        while i < 9:
            if jn(n1,x)*jn(n1,x+dx) < 0:
                x_zero_k[n1][i] = x
                i += 1
            x += dx
    x_zeros[0] = sc.optimize.fsolve(j0,x_zero_k[0])
    x_zeros[1] = sc.optimize.fsolve(j1,x_zero_k[1])
    x_zeros[2] = sc.optimize.fsolve(j2,x_zero_k[2])
    x_zeros[3] = sc.optimize.fsolve(j3,x_zero_k[3])
    x_zeros[4] = sc.optimize.fsolve(j4,x_zero_k[4])
    x_zeros[5] = sc.optimize.fsolve(j5,x_zero_k[5])
    x_zeros[6] = sc.optimize.fsolve(j6,x_zero_k[6])
    return x_zeros

#@jit
def bessel_array(x_zeros,mode,node,width=110):
    """
    ベッセルと三角関数の基底を作成
    """
    r_b = np.zeros((width,width))
#    r_arctan = np.zeros((width,width))
    r1 = np.zeros((width,width))
    r2 = np.zeros((width,width))
    if width%2 ==0:
        a = np.arange(-width/2+0.5,width/2+0.5,1)
    else:
        a = np.arange(-width//2,width//2+1,1)
    r1[:]=a
    r2=np.transpose(r1)
    r_b = np.sqrt(r1**2+r2**2)/width*2*x_zeros[mode][node]
    r_at = np.arctan2(r2,r1)
    return r_b,r_at

#@jit
def make_bessel(bessel_array,r_at,mode,sin=0):
    """
    フーリエベッセル関数の基底を作成
    """
    j=np.ones(bessel_array.shape)
    if mode == 0:
        if sin==0:
            j = spe.jn(mode,bessel_array)
        else:
            pass
    else:
        if sin == 1:
            j = spe.jn(mode,bessel_array)*np.sin(mode*r_at)
        else:
            j = spe.jn(mode,bessel_array)*np.cos(mode*r_at)
    return j

#@jit
def fb_one(x_zeros,mode,node,sin1=0,width1=110):
    """
    ある一つのフーリエベッセル関数の基底を作成
    """
    bessel_argue,r_at = bessel_array(x_zeros,mode,node,width=width1)
    fb = make_bessel(bessel_argue,r_at,mode,sin=sin1)
    return fb

def shibori(width,area_width_ratio):
    """
    あるエリア外をゼロにする関数
    
    widthにグリッド幅
    
    area_width_ratioが幅の割合
    """
    if width%2==1:
        x = np.arange(-(width//2),width//2+1); y = np.arange(-(width//2),width//2+1);
        y = y.reshape((width,1))
    else:
        x = np.arange(-(width//2)+0.5,width//2+0.5); y = np.arange(-(width//2)+0.5,width//2+0.5);
        y = y.reshape((width,1))
    r2 = x**2+y**2
    area = np.where(r2<=(width/2*area_width_ratio)**2,1,0)#width*nを一旦抜いてる
    return area
    
def weight(r0=0.65,r_threshold=0.6,ratio=0.9,width=11,r_edge=1):
    """
    X^2のそれぞれに重み関数を付ける
    r0は1/2点、r_thresholdはratioになる点、widthは幅、r_edgeは座標の端の値
    """
    _a = np.log((1-ratio)/ratio)/(2*(r_threshold-r0))#係数は左のように決定できる
    _x = np.linspace(-r_edge,r_edge,width)
    _y = np.linspace(-r_edge,r_edge,width)[:,np.newaxis]
    _r = np.sqrt(_x**2+_y**2)
    weight_result = (np.tanh(_a*(r0-_r))+1)/2
    return weight_result

#@jit
def fb_complete(width,n=1):
    """
    全てのフーリエベッセル関数の基底を作成
    """
    x_zeros = x_zeros_comp()
    fb_complete = np.zeros((7,9,2,width*n,width*n))
    for i in range(7):
        for j in range(9):
            for k in range(2):
                fb_complete[i,j,k] = fb_one(x_zeros,i,j,sin1=k,width1=width*n)
#    wn = width
#    print(wn)
    if width%2==1:
        x = np.arange(-(width//2),width//2+1); y = np.arange(-(width//2),width//2+1);
        y = y.reshape((width,1))
    else:
        x = np.arange(-(width//2)+0.5,width//2+0.5); y = np.arange(-(width//2)+0.5,width//2+0.5);
        y = y.reshape((width,1))
    r2 = x**2+y**2
    Dj = np.where(r2<=(width/2)**2,1,0)#width*nを一旦抜いてる
    if n==1:
        return fb_complete[:,:,:]*Dj
    else:
        return fb_partial(fb_complete,n)[:,:,:]*Dj



def fb_normalize(fb_complete,width):
    """
    基底を規格化する
    """
    fb_normalize = fb_complete/np.pad(np.sqrt((fb_complete**2).sum(axis=4).sum(axis=3))[:,:,:,np.newaxis,np.newaxis],[[0,0],[0,0],[0,0],[0,width-1],[0,width-1]],'edge')
    return fb_normalize

#@jit    
def fb_inpro_matrix(fb_complete,weight=-1):
    """
    フーリエベッセル関数の基底の内積を取る\n
    その結果を行列にまとめる
    """
    if weight==-1:
        w=1
    else:
        w=weight
    P = np.zeros((83,83))
    for i in range(83):
        for j in range(83):
            x,y = mode_reverse(i),mode_reverse(j)
            P[i,j]=np.sum(w*fb_complete[x[0],x[1],x[2]]*fb_complete[y[0],y[1],y[2]])
    return P

#@jit
def fb_base_eps_vector(fb_base,file_tomography,width,filenumber=[0],mnum=-1,decimate=10,weight=1):
    """
    フーリエベッセル関数の基底と発光量行列との内積を取る
    pythonで作った基底は(7,9,2,11,11)で入力、発光量は(121)で入力
    fortranで作った基底は(83,11,11)で入力、発行量は(121)で入力
    基底はfb_complete(110)>fb_partial(10)から、発光量データはトモグラフィーから
    トモグラフィのデータは(ファイルの個数,面の個数,全グリッド数)、(ファイルの個数,面の個数,11,11)、もしくは、(11,11)で入力
    mnumは作る配列の個数
    2018/10/29にndim=3を別の定義にする。トモグラフィーのデータは読み込み時に11*11に変えたもののみ受け付け
    """
    w = weight
    fdim = np.ndim(file_tomography)    #    if fdim >= 3 and fdim <= 4:#ファイルの次元で判別 2018/10/29に撤廃
    if fdim == 4:
        ft,ft0 = len(file_tomography),len(file_tomography[0])#ファイルの個数とその長さを測っている
        ft_decimated = ft0//decimate # 高速化のため10個のデータを間引いてやってる decimateってのもある 使う配列の個数
        if mnum!=-1:#作る配列の数を決める
            ft_decimated = mnum
        
        f_tomo = file_tomography
    #        if fdim==3:                                   2018/10/29に撤廃
    #            f_tomo = file_tomography.reshape(ft,ft0,11,11)#トモグラフィデータの形を11*11に形を変えてる
        if filenumber==-1:
            flen = len(file_tomography)
            filenumber = list(range(flen))
        else:
            flen = len(filenumber)
        b = np.zeros((flen,ft_decimated,83))#各面ごとの基底関数と発光量データの内積を取るための配列を確保
        
        fb_base_sorted = np.zeros((83,11,11))
        for fi,fnum in enumerate(filenumber):
            print(fnum)#試し
            if fb_base.shape[0:3]==(7,9,2):#pythonの分
                for i in range(83):
                    x = mode_reverse(i)
                    fb_base_sorted[i] = fb_base[x[0],x[1],x[2]]
                for i in range(ft_decimated):
                    for j in range(83):
                        b[fi,i,j] = np.sum(w*fb_base_sorted[j]*f_tomo[fnum,i*decimate])#内積を取る
                    if (i+1)%10000==0:
                        print("finished generating "+str(i+1)+" vector")#10000面済むごとにプリント
            elif fb_base.shape[0:3]==(83,11,11):#fortranの分
                for i in range(ft_decimated):
                    for j in range(83):
                        b[fi,i,j] = np.sum(w*fb_base[j]*f_tomo[fnum,i*10])#内積を取る
                    if (i+1)%10000==0:
                        print("finished generating "+str(i+1)+" vector")#10000面済むごとにプリント
            else:
                raise ValueError("基底関数の配列が違います")
#        if b.shape[0] == 1:
#            b = b.reshape(ft_decimated,83) #後々変える 2018/10/29撤廃
        return b
    
    if fdim == 3:
        ft = len(file_tomography)#ファイルの個数とその長さを測っている
        ft_decimated = ft//decimate # 高速化のため10個のデータを間引いてやってる decimateってのもある 使う配列の個数
        if mnum!=-1:#作る配列の数を決める
            ft_decimated = mnum
        
        f_tomo = file_tomography
        
        b = np.zeros((ft_decimated,83))#各面ごとの基底関数と発光量データの内積を取るための配列を確保
        
        fb_base_sorted = np.zeros((83,11,11))
        
        if fb_base.shape[0:3]==(7,9,2):#pythonの分
            for i in range(83):
                x = mode_reverse(i)
                fb_base_sorted[i] = fb_base[x[0],x[1],x[2]]
            for i in range(ft_decimated):
                for j in range(83):
                    b[i,j] = np.sum(w*fb_base_sorted[j]*f_tomo[i*decimate])#内積を取る
                if (i+1)%10000==0:
                    print("finished generating "+str(i+1)+" vector")#10000面済むごとにプリント
        elif fb_base.shape[0:3]==(83,11,11):#fortranの分
            for i in range(ft_decimated):
                for j in range(83):
                    b[i,j] = np.sum(w*fb_base[j]*f_tomo[i*decimate])#内積を取る
                if (i+1)%10000==0:
                    print("finished generating "+str(i+1)+" vector")#10000面済むごとにプリント
        else:
            raise ValueError("基底関数の配列が違います")
        return b
    
    elif fdim == 2:#11*11の一面を入力
        #各関数と内積取れているか確認
        ft = len(file_tomography)
        b = np.zeros(83)
        f_tomo = file_tomography
        if fb_base.shape[0:3]==(7,9,2):#pythonの分
            for j in range(83):
                x = mode_reverse(j)
                b[j] = np.sum(w*fb_base[x[0],x[1],x[2]]*f_tomo)
        elif fb_base.shape[0:3]==(83,11,11):
             for j in range(83):
                b[j] = np.sum(w*fb_base[j]*f_tomo)
        else:
            raise ValueError("基底関数の配列が違います")
        return b
    else:
        raise ValueError("トモグラフィ配列が違います")

#@jit
def fb_solve(P,gamma,b):
    """
    固有値ベクトルを解く
    Pもbも一辺の長さは等しい
    """
    if np.ndim(b)==1:#固有値ベクトルの個数で判別
        gam_I = gamma*np.identity(len(b))
        Pg = P+gam_I
        a = np.linalg.solve(Pg,b)
        return a
    elif np.ndim(b)==2:#複数面の場合
        lb = len(b)#面の個数
        lb0 = len(b[0,:])#固有値クトルの個数
        a = np.zeros((lb,lb0))
        gam_I = gamma*np.identity(lb0)
        Pg = P + gam_I
        for i in range(lb):
            a[i] = np.linalg.solve(Pg,b[i])
            if (i+1)%10000==0:
                print("finished "+str(i+1)+" solving data")
        return a
    elif np.ndim(b)==3:#ファイル、面ともに複数の場合
        lb = len(b)#ファイルの個数
        lb0 = len(b[0,:])#面の個数
        lb0_0 = len(b[0,0,:])#固有値クトルの個数
        a = np.zeros((lb,lb0,lb0_0))
        gam_I = gamma*np.identity(lb0_0)
        Pg = P + gam_I
        for i in range(lb):
            for j in range(lb0):
                a[i,j] = np.linalg.solve(Pg,b[i,j])
                if (i*lb0+j+1)%10000==0:
                    print("finished "+str(i*lb0+j+1)+" solving data")
        return a
    else:
        raise ValueError("Vector dimension is different!")

#@jit
def fb_reconst(fb_base,eigenvector):
    """
    フーリエベッセル係数から画像を再構成
    """
    if np.ndim(eigenvector.shape)==1:
        rec = np.zeros((11,11))
        for i in range(83):
            rec += fb_base[mode_reverse(i)]*eigenvector[i]
        return rec

def fb_2d(file_fb,fb_base,start,nsum=1):
    """
    対称成分と非対称成分の生成
    """
    fb_n0 = np.zeros(fb_base.shape[3:5])
    fb_n_non0 = np.zeros(fb_base.shape[3:5])
    for i in range(9):
        fb_n0 += np.sum(file_fb[0,start:start+nsum,mode_at(n=i)])/nsum*fb_base[0,i,0]
    for i in range(1,7):
        for j in range(7):
            for k in range(2):
                if i == 1:
                    fb_n_non0 += np.sum(file_fb[0,start:start+nsum,mode_at(m=i,n=j,sin=k)])/nsum*fb_base[i,j,k]
                else:
                    if j == 6:
                        pass
                    else:
                        fb_n_non0 += np.sum(file_fb[0,start:start+nsum,mode_at(m=i,n=j,sin=k)])/nsum*fb_base[i,j,k]
    return fb_n0,fb_n_non0

def read_elem(readfile="elem_array.dat"):
    """
    fortranのデータを読み込む
    Bmovie.cとfmovie2.fからよもこまないといけない
    今回は基底関数を(83,11,11)の形で出力する
    """
    with open(readfile, "rb") as f:
        elem_array = np.fromfile(f,np.float64,83*11*11).reshape(83,11,11)
    return elem_array




#@jit
def plot3D(figure=[],shape=(),sharex=False,sharey=False,subplot_kw={"projection":"3d"},width=11):
    if shape == ():
        fig,axes = plt.subplots(len(figure),1,sharex=sharex,sharey=sharey,subplot_kw=subplot_kw)
    elif len(shape)==2:
        fig,axes = plt.subplots(shape[0],shape[1],sharex=sharex,sharey=sharey,subplot_kw=subplot_kw)
    else:
        raise "Please input 2 length tuple as it fits figure."
    
    x,y = np.arange(0,width),np.arange(0,width)
    X,Y = np.meshgrid(x, y)
    axes.plot_surface(X,Y,figure,cmap='bwr')
#    plt.show()

#@jit
def p_partial(a,b,n):
    """
    a*bの二次元配列をn倍分割する
    """
    l_a = len(a)
    l_b = len(b)
    c = np.empty((l_a,l_b))
    c1 = np.empty((l_a//n,l_b//n))
    for i in range(l_a):
        for j in range(l_b):
            c[i,j] = a[i]+b[j]
    for i in range(l_a//n):
        for j in range(l_b//n):
            c1[i,j] = np.sum(c[n*i:n*(i+1),n*j:n*(j+1)])/n**2
    return c1

#@jit
def partial(a,n):
    """
    a*aの二次元配列をn倍分割する
    """
    x,y = a.shape
    c1 = np.empty((x//n,y//n))
    for i in range(x//n):
        for j in range(y//n):
            c1[i,j] = np.sum(a[n*i:n*(i+1),n*j:n*(j+1)])/n**2
    return c1

#@jit
def fb_partial(bessel_complete,n):
    """
    フーリエベッセル関数の基底をn分割する
    """
    a,b,c,x,y = bessel_complete.shape
    fb_adjust=np.zeros((a,b,c,x//n,y//n))
    for i in range(a):
        for j in range(b):
            for k in range(c):
                fb_adjust[i,j,k] = partial(bessel_complete[i,j,k],n)
    return fb_adjust


#@jit    
def make_frequency(file,d,nfft):
    """
    横軸の周波数配列を作る
    """
    freq,_ = sig.welch(file[0,:,mode_at(0,1,0)],fs=1/d,nfft=nfft,nperseg=nfft)
    return freq

#@jit
def fb_crs_phase(file,m=0,n=0,sin=0,m1=0,n1=0,sin1=0):
    """
    各成分同士のクロススペクトル、phase  phaseを作る際クロススペクトルを使用することになるので、phaseはここからとってください
    """
    flen = len(file)
    crs = np.zeros((flen,lfft),dtype='complex128')
    for i in range(flen):
        _,crs[i] = sig.csd(file[i,:,mode_at(m,n,sin)],file[i,:,mode_at(m1,n1,sin1)],fs=1/d,nfft=nfft,nperseg=nfft) 
    crs = crs.sum(axis=0)
    phase = sig.xphase(crs)
    return crs,phase    

#@jit    
def fb_psd(file,fnlist=[],m=0,n=0,sin=0,start=0,stop=30000,nfft=1000):
    """
    各モードのパワースペクトル
    """
    lfft=nfft//2+1
    if fnlist==[]:
        flen = len(file)
        fnlist = list(range(flen))
    else:
        flen=len(fnlist)
    psd = np.zeros((flen,lfft))
    for i,j in zip(range(flen),fnlist):
        _,psd[i] = sig.welch(file[j,start:stop,mode_at(m,n,sin)],fs=1/d,nfft=nfft,nperseg=nfft)
    psd = psd.sum(axis=0)/flen
    return psd

#@jit
def fb_coherence(file,m=0,n=0,sin=0,m1=0,n1=0,sin1=0):
    """
    各モードのコヒーレンス
    """
    psd0      = fb_psd(file,m,n,sin)
    psd1      = fb_psd(file,m1,n1,sin1) 
    crs,_     = fb_crs_phase(file,m,n,sin,m1,n1,sin1)
    coherence = abs(crs)**2/psd0/psd1 
    return coherence
#@jit
def mpower(file,fnlist=[],mode=0,start=0,stop=30000,nfft=1000):
    """
    あるモードの強度を算出する
    """
    lfft=nfft//2+1
    if mode == 0:
        mpower = np.zeros((1,9,lfft))
        for i in range(9):
            mpower[0,i,:] = fb_psd(file,m=mode,n=i,fnlist=fnlist,start=start,stop=stop,nfft=nfft)
        return mpower
    if mode == 1:
        mpower = np.zeros((2,7,lfft))
        for i in range(7):  
            for j in range(2):
                mpower[j,i,:] = fb_psd(file,m=mode,n=i,sin=j,fnlist=fnlist,start=start,stop=stop,nfft=nfft)
        return mpower
    elif 1 < mode < 7:
        mpower = np.zeros((2,6,lfft))
        for i in range(6):
            for j in range(2):
                mpower[j,i,:] = fb_psd(file,m=mode,n=i,sin=j,fnlist=fnlist,start=start,stop=stop,nfft=nfft)
        return mpower
    else:
        raise ValueError("input mode 0~6")
    
def dict_ex(filename="diction",**kwargs):
    """
    ファイリングを試す
    """
    np.savez(filename+".npz",kwargs)
    loadfile = np.load(filename+".npz")
    l0 = loadfile[list(loadfile.keys())[0]]
    return l0

if __name__ == '__main__':
    
    pass
        
#    #フーリエベッセル級数をcontour
#    width = 100 #幅
#    node = 0 #ノード数
#    mode = 0 #モード数
#    i_num = 4 #規格化ゼロ点
#
#    #        print (x_zero_k)
#    #    jnf = lambda n,x:jn(n,x) #jnを1次に
#    x_zero = x_zeros_comp()
#        
#    for i in range(7):
#        print (x_zero[i])
#    
#    x,y = np.mgrid[-width:width+1:1,-width:width+1:1]
#    #x1,y1 = np.mgrid[1:width+1:1,1:width+1:1]
#    
#    r = np.sqrt(x**2+y**2) #半径
#    #r1 = np.sqrt(x1**2+y1**2) #半径
#    
#    z = jn(node,r/width*x_zero[node][i_num])*np.cos(mode*np.arctan2(y,x)) #nモードi番目のゼロ点
#    
#    #figure-----------------------------------------------------
#    fig = plt.figure(figsize=(6,5))
#    im = plt.pcolor(x, y, z, cmap='BuPu', vmin=z.min(), vmax=z.max())
#    fig.colorbar(im)
#    plt.title('pcolor')
#    #plt.xticks([x-0.5 for x in range(1,12)] ,range(1,12))
#    #plt.yticks([y-0.5 for y in range(1,12)] ,range(1,12))
#    #plt.axis([x.min(), x.max(), y.min(), y.max()])
#    #plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5],range(1,12))
#    plt.show()
#    #-----------------------------------------------------
#    
#    header = "/Volumes/kilimanjaroHD/experiment/170712/group6/output_data/070/eFB.dat"
#    print (os.path.getsize(header))
#    
#    data_num = 30000 #データ番号/10
#    data_1 = 8 #データ一つあたり
#    data_br = 83 #１面あたり83面
#    data_wa = 60000#欲しいデータ数
#    data_num -= 1
#    
#    start = time.time()
#    #file読み込み
#    with open(header, "rb") as f:
#        start = time.time()
#        f.seek(0*data_1*data_br*data_num)  #欲しいデータ番号
#        g = np.fromfile(f,np.float64,data_br*data_wa).reshape(data_wa,data_br)
#    
#    print (str(time.time()-start) + " [sec]")
#    
#    #z1 = g[]*jn(node_v,r/width*x_zero[node][i_num])*np.sin(mode_v*np.arctan2(y,x))
#    ##@jit
#    def fbsin(x,y,k,node,mode,data_num):
#        r = np.sqrt(x**2+y**2)
#        return g[data_num][k]*jn(mode,r/width*x_zero[mode][node])*np.sin(mode*np.arctan2(y,x))
#    
#    ##@jit
#    def fbcos(x,y,k,node,mode,data_num):
#        r = np.sqrt(x**2+y**2)
#        return g[data_num][k]*jn(mode,r/width*x_zero[mode][node])*np.cos(mode*np.arctan2(y,x))
#    
#    ##@jit
#    def zfb(x,y,data_num):
#        a=0;k=0;
#        for k in range(9):
#            a += fbcos(x,y,k,k,0,data_num)
#        for k in range(9,23,2):
#            a += fbcos(x,y,k,(k-7)//2,1,data_num)
#        for k in range(10,24,2):
#            a += fbsin(x,y,k,(k-8)//2,1,data_num)
#        for k in range(23,35,2):
#            a += fbcos(x,y,k,(k-21)//2,2,data_num)
#        for k in range(24,36,2):
#            a += fbsin(x,y,k,(k-22)//2,2,data_num)
#        for k in range(35,47,2):
#            a += fbcos(x,y,k,(k-33)//2,3,data_num)
#        for k in range(36,48,2):
#            a += fbsin(x,y,k,(k-34)//2,3,data_num)
#        for k in range(47,59,2):
#            a += fbcos(x,y,k,(k-45)//2,4,data_num)
#        for k in range(48,60,2):
#            a += fbsin(x,y,k,(k-46)//2,4,data_num)
#        for k in range(59,71,2):
#            a += fbcos(x,y,k,(k-57)//2,5,data_num)
#        for k in range(60,72,2):
#            a += fbsin(x,y,k,(k-58)//2,5,data_num)
#        for k in range(71,83,2):
#            a += fbcos(x,y,k,(k-69)//2,6,data_num)
#        for k in range(72,84,2):
#            a += fbsin(x,y,k,(k-70)//2,6,data_num)
#        return a
#    
#    #print zfb(0,0,data_num)
#    #print g[data_num][0:9],sum(g[data_num][0:9])
#    
#    z1 = zfb(x,y,data_num) #nモードi番目のゼロ点
#    
#    #figure-----------------------------------------------------
#    fig = plt.figure()#figsize=(6,5))
#    #im = plt.pcolor(x, y, z1, cmap='BuPu', vmin=z1.min(), vmax=z1.max())
#    #im = plt.pcolor(x, y, z1, cmap='RdBu', vmin=z1.min(), vmax=z1.max())
#    im = plt.pcolor(x, y, z1, cmap='RdBu', vmin=0, vmax=z1.max())
#    fig.colorbar(im)
#    plt.title('eFB')
#    plt.show()
#    #-----------------------------------------------------
#    
#    header = "/Volumes/kilimanjaroHD/experiment/170712/group6/output_data/070/ebes.dat"
#    
#    g = pd.read_table(header,header=None,skiprows=2,delim_whitespace=True)
#    
#    g.columns = ["x","y","z"]
#        
#    X,Y = np.mgrid[-100:101:1,-100:101:1]
#    dat = g.as_matrix()
#    z = dat[:,2].reshape(201,-1)
#        
#    im = plt.figure()
#    plt.pcolor(X,Y,z,cmap='RdBu', vmin=z.min(), vmax=z.max())
#    plt.colorbar()
#    plt.title("ebes")
#    plt.show()