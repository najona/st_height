#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:29:45 2020

@author: najona
"""
import numpy as np
from scipy.interpolate import griddata
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
import pdb
from math import atan2,degrees

# interpolate gridded temperature/salinity at given locations
def interpolateTS(x, y, str):
    # linear interpolation
    if str == "linear":
        f = interpolate.interp1d(x, y, "linear")
        
    # quadratic spline interpolation
    elif str == "quadratic":
        f = interpolate.interp1d(x, y, "quadratic")
    
    # cubic spline interpolation
    elif str == "cubic":
        f = interpolate.interp1d(x, y, "cubic")
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation
    # of zeroth, first, second or third order or integer specifying
    # the order of the spline interpolator to use. Default is ‘linear’.
    
    x_new = np.arange(np.min(x), np.max(x), 0.1)
    return f,x_new

def plotDepthST(T, PSAL, depth, lat, long):

    fig, ax1 = plt.subplots(figsize=(10,5))
    color = 'tab:red'
    ax1.set_ylabel('Temperature [°C]', color=color)
    ax1.set_xlabel('Depth [m]')
    ax1.plot(depth, T, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Salinity', color=color)  
    ax2.plot(depth, PSAL, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Argo profile (lat = '+str(round(lat,2))+'°'+', long = '+str(round(long,2))+'°)')

    fig.tight_layout()
    plt.show()
    
def getA(kappa_x, dkappa, k):
    # AUFSTELLUNG DER DESIGNMATRIX
    # INPUT:
    # kappa_x   Knotenmenge
    # dkappa    Knotenabstand
    # k         Stützstellen
    
    # Erweiterung der Knotenmenge an den Rändern
    kappa = np.append(kappa_x[0]-dkappa, kappa_x)
    kappa = np.append(kappa, kappa_x[-1]+dkappa)

    xD = np.zeros((len(k), len(kappa)))
    for i in range(len(k)):
        xD[i] = (k[i] - kappa)/dkappa

    # Designmatrix aufstellen
    A = np.zeros(xD.shape) # Initialisierung der Designmatrix

    # Besetze die Nicht-Null-Eintraege über logische Indizierung
    A[np.where(np.logical_and(xD>=-2, xD<-1))] = 1 
    A[np.where(np.logical_and(xD>=-2, xD<-1))] = ((xD[np.where(np.logical_and(xD>=-2, xD<-1))]+2)**3)/6
    A[np.where(np.logical_and(xD>=-1, xD<0))] = ((xD[np.where(np.logical_and(xD>=-1, xD<0))]+2)**3)/6 - 4*((xD[np.where(np.logical_and(xD>=-1, xD<0))]+1)**3)/6
    A[np.where(np.logical_and(xD>=0, xD<1))] = ((-xD[np.where(np.logical_and(xD>=0, xD<1))]+2)**3)/6 - 4*((-xD[np.where(np.logical_and(xD>=0, xD<1))]+1)**3)/6
    A[np.where(np.logical_and(xD>=1, xD<2))] = ((-xD[np.where(np.logical_and(xD>=1, xD<2))]+2)**3)/6
    return A

def getA_harm(t,omega,p):
    # Funktion zur Bestimmung der Designmatrix für eine polyharmonische Funktion der Form:
    # f(x) = c_0 + a*t + s_1*sin(omega*x) + c_1*cos(omega*x) + s_2*sin(2*omega*x) + c_p*cos(p*omega*x)
    # Input:
    # t ...     [nx1] double, Stuetzstellen der Beobachtungen mit n ... # Beobachtungen
    # omega ... double, Kreisfrequenz
    # p ...     double, Ordnung der polyharmonischen Funktion d
    #
    # Output:
    # A ...    [nx2*p+1] double, Designmatrix

    # Designmatrix initialisieren
    A = np.zeros((len(t), 2*(p+1)))
    A_ = np.zeros((len(t), 2))

    # konstanter Anteil
    A_[:,0] = 1

    # linearer Anteil
    A_[:,1] = t

    # Sinus- und Kosinusanteile pro Frequenz k*omega mit k = 1..p
    for k in range(1,p+1):
        A[:,2*k] = np.sin(k*omega*t)
        A[:,2*k+1] = np.cos(k*omega*t)
    A_new = np.concatenate((A_, A[:,2:]), axis=1)
    return A_new

def linearGMM(A,l):
    # Ausgeglichene Parameter
    xS = np.linalg.solve(A.T@A, (A.T).dot(l))

    # ausgegl. Beobachtungen
    lS = A.dot(xS)

    # Verbesserungen
    v = lS-l

    # Rechenprobe
    rp = (A.T).dot(v)
    return xS,lS,v,rp

def clustering(lat, long, degN, degE, obs, maxdiff):
    cluster = np.array([])
    idx = np.array([])
    for i in range(len(lat)):
        if cluster.size == 0: # check if array is empty
            mean_obs = np.mean(obs[np.where((np.abs(lat-degN) < 3) & (np.abs(long-degE) < 4))])
        else:
            mean_obs = np.median(cluster) 
            
        if ((np.abs(lat[i]-degN) < 3) and (np.abs(long[i]-degE) < 4) and (obs[i]-mean_obs < maxdiff)):
            cluster = np.append(cluster, obs[i])
            idx = np.append(idx, i)

    long_cluster = np.array([])
    lat_cluster = np.array([])
    for i in range(len(idx)):
        lat_cluster = np.append(lat_cluster, lat[int(idx[i])])
        long_cluster = np.append(long_cluster, long[int(idx[i])])
    return idx, cluster, lat_cluster, long_cluster

def autocovariance(T):
    meanT = np.mean(T)
    n = len(T)
    m = np.floor(n/10)
    
    #k = np.arange(0, int(m), 1)
    gamma = np.zeros(int(m))
    

    for k in range(int(m)):
        c = 0;
        for j in range(int(n-k)):
           c = c + (T[int(j)] - meanT) * (T[int(j+k)] - meanT)
        # Autokovarianzfunktion
        gamma[k] = 1/(n - k - 1) * c
    return gamma


#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return
    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)
    return

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)
    return