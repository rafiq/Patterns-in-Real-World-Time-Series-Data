#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:22:57 2022

@author: kamaludindingle
"""
import numpy as np
import matplotlib.pyplot as plt

def BrownianMotion(years=15,record=1):
    #Brownian motion for (10) years, recorded every "record" years
    Gaussian = np.random.randn(years)
    Bm = np.cumsum(Gaussian)# Brownian motion
    if record==1:
        SaveBm = Bm
    elif record!=1:
         SaveBm = [Bm[j] for j in range(len(Bm)) if (j%record)==0]

    return SaveBm