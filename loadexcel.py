#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:55:01 2017

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
import math 
import re

def data_number(date,power,magfield):
    """
    日付とパワー、磁場の強さからpathリストを返す
    date = 170712
    power = 3
    magfield = 600
    みたいな感じ
    """
#    date = 170712
#    power = 3
#    magnetic = 600
    
    header = "/Volumes/kilimanjaroHD/experiment/shot_log.xlsm"
    
    loadf = pd.read_excel(header,sheet_name=str(date)[:-2],header=[0,1,2],skiprows=None)
    
    loadf = loadf.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')
    no = ("tomography","group6","No.")
    p = ("RF","power[kW]",'')
    m = ("magnetic field[G]",'','')
    
    df = loadf[[no,p,m]]
    df = df.loc[date]
    df = df[df[p].isin([power])]
    df = df[df[m].isin([magfield])].dropna()
    df = df.astype({no:int})
    return df[no].values.tolist()

def header_generator(shot=-1,subshot=-1,power=-1,magfield=-1,date=-1,group="6",extension='emission',home=False):
    """
    必要なヘッダを出力
    """
#===================== header =================================================
    header = "/Volumes/kilimanjaroHD/experiment/shot_log.xlsm"
    header_tomo = "/Volumes/kilimanjaroHD/experiment/"
    if home==True:
        header = "C:/Users/mamoru/Desktop/local_file/experiment/shot_log.xlsm"
        header_tomo="C:/Users/mamoru/Desktop/local_file/experiment/"
    opd = "output_data/"
    filen="/emission.dat"
    if extension == 'emission': 
        filen="/emission.dat"
    elif extension == 'eFB':
        filen="/eFB.dat"
    elif extension == 'ebes':
        filen="/ebes.dat"
    else:
        try:
            raise ValueError("emissionかeFBかebesと書いてください")
        except ValueError as e:
            raise
#==============================================================================

    
    loadf = pd.read_excel(header,sheet_name=str(date)[:-2],header=[0,1,2],skiprows=None)
    loadf = loadf.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')
    
    no6 = ("tomography","group6","No.") #各ヘッダーを除去
    no8 = ("tomography","group8","No.")
    nohioki = ("tomography","Hioki","No")#Noのドットなし
    p = ("RF","power[kW]",'')
    m = ("magnetic field[G]",'','')
    sh = ("probe shot No.","prime",'')
    subsh = ("probe shot No.","sub",'')
    
#    df = loadf[[no6,no8,nohioki,p,m,sh,subsh]]#必要なデータの選別
#    df.columns = ["group6 No.","group8 No.","Hioki No.","power","magfield","shot No.","subshot No."]
    
    if shot==-1 and subshot==-1:#データの選別
        if int(math.log10(abs(date))+1)!=6 or date<=0:#日付の正確な入力
            try:
                raise NameError("Write date like 170712")
            except NameError as e:
                raise
                
        if power==-1 | magfield==-1 | date ==-1:#各々の数字を記入するよう示す
            raise TypeError("Write power,magfield,date,group like 5,900,170712,\"6\"")
                
        else:
            df = loadf[[sh,subsh,no6,no8,nohioki,p,m]]
            df = df.loc[date]
            df = df[df[p].isin([power])]
            df = df[df[m].isin([magfield])]
            if len(df.index) == 0:#データなし
                raise ValueError("データがないです")
                
            head=[{"date":date,"power":power,"magfield":magfield}]
            
            if group == "6":#グループ６について
                df = df.drop([no8,nohioki], axis=1)
                df = df.dropna()
                df = df.astype({no6:int})
                s = df[sh].values.tolist()
                su = df[subsh].values.tolist()
                head[0].update({'shot':s[0],'subshot':su})
                for i in df[no6].values.tolist():
                   head.append(header_tomo+str(date)+"/group6/"+opd+"%.3d"%i+filen)    
                return head
            
            elif group == "8":
                df = df.drop([no6,nohioki], axis=1)
                df = df.dropna()
                df = df.astype({no8:int})
                s = df[sh].values.tolist()
                su = df[subsh].values.tolist()
                head[0].update({'shot':s[0],'subshot':su})
                for i in df[no8].values.tolist():
                   head.append(header_tomo+str(date)+"/group8/"+opd+"%.3d"%i+filen)    
                return head
            
            elif group == "hioki":
                df = df.drop([no6,no8], axis=1)
                df = df.dropna()
                df = df.astype({nohioki:int})
                s = df[sh].values.tolist()
                su = df[subsh].values.tolist()
                head[0].update({'shot':s[0],'subshot':su})
                for i in df[nohioki].values.tolist():
                   head.append(header_tomo+str(date)+"/group0/"+opd+"%.3d"%i+filen)    
                return head
            
            else:
                try:
                    raise NameError("Please write name within 6,8,hioki")
                except NameError as e:
                    raise
    
    else:   pass #shot,subshot版はあとで

def file_loader(path=[],stime=200000,etime=500000,file_number_list=[],reshape=False):#,extension='emission'):
    """
    stime:取得開始時間[μs]
    etime:取得終了時間[μs]
    ---extension---
    
    emission:(ファイル数*,時間長,121)の配列を返します
    eFB:(ファイル数,時間長,83)の配列を返します(eFBは全体で6万点)
    """
    extension = re.split('[./]', path[0])[-2]
    if extension == 'emission':
        #==============constant================
        diff = etime - stime 
        size = os.path.getsize(path[0])
        br_1 = 121 #1面データ個数
        data_1 = 8 #データ一つあたりのサイズ
        data_num = br_1*data_1 #１面データサイズ
        br_sum = size/data_num
        #======================================
        if file_number_list==[]:
            l = range(len(path))
        else:
            l = file_number_list
        print("read file number is {}".format(file_number_list))
        file=np.zeros((len(l),diff,br_1))
    
        for fn,i in enumerate(l):
            with open(path[i], "rb") as f:
                print(path[i])
                f.seek(stime*data_num)  #欲しいデータ番号の最初
                file[fn,:,:] = np.fromfile(f,np.float64,br_1*diff).reshape(diff,br_1)
        if reshape==True:
            file = file.reshape(len(l),diff,11,11)
        return file
    
    elif extension == 'eFB':
        #==============constant================
        stime,etime = stime//10,etime//10
        diff = etime - stime 
        size = os.path.getsize(path[0])
        br_1 = 83 #1面データ個数
        data_1 = 8 #データ一つあたりのサイズ
        data_num = br_1*data_1 #１面データサイズ
        br_sum = size/data_num
        #======================================
        if file_number_list==[]:
            l = range(len(path))
        else:
            l = file_number_list
        print("read file number is {}".format(file_number_list))
        file=np.zeros((len(l),diff,br_1))
        fn=0
    
        for i in l:
            with open(path[i], "rb") as f:
                print(path[i])
                f.seek(stime*data_num)  #欲しいデータ番号の最初
                file[fn,:,:] = np.fromfile(f,np.float64,br_1*diff).reshape(diff,br_1)
            fn+=1
        return file
        
    else:
        raise TypeError("emissionかeFBと書いてください")
        
                 
    
    
if __name__ == '__main__':
    print(data_number(170712,3,600))
    a = header_generator(power=5,magfield=600,date=170712)
    print(a)
    
    print(header_generator(power=4,magfield=60,date=170712))    





#loadf.columns = ["date","prime","sub","gr6filter","gr6No",
#                 "gr8filter","gr8No","hiokifilter","hiokiNo","bgsource",
#                 "bgmain","gassource","gasmain","gassccm"]
