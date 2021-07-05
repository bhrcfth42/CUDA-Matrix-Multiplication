# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:23:28 2021

@author: fatih
"""

#Kütüphaneler ekliyoruz
from numba import vectorize
import numpy as np

#Cuda paralelleştirmesi için dönüş tipi float64 ve parametre olarak 2 tane float64 gönderdiğimizi belirtiyoruz.
#Dönüş değeri olarak 2 parametrenin çarpımı olduğunu gösteriyoruz.
@vectorize(["float64(float64,float64)"],target="cuda")
def Carpma(a,b):
  return a*b

#N bizim dizi boyutumuz
N=10000000
  
#numpy random ile N boyutlu random float64 dizi oluşturuyoruz
A=np.random.rand(N)

#numpy random ile N boyutlu random float64 dizi oluşturuyoruz
B=np.random.rand(N)

#numpy 1 ile N boyutlu değerleri 1 olan ama float64 veri tipli dizi oluşturuyoruz.
#C dizisinde dönüş değerlerimizi tutacağız.
C=np.ones(N,dtype=np.float64)
  
#Cuda fonksiyonumuzu çağırarak C dizine atıyoruz.
C=Carpma(A,B)
  
#C dizimizi basıyoruz.
print(C)