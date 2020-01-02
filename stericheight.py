# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:08:57 2019

@author: Nadja Jonas
"""

import numpy as np
from gsw import *
import gsw.conversions
from operator import *
from netCDF4 import *
import julian
import datetime


def readnc(data):
    ds = Dataset(data)

    # temperature (in degC)
    T = ds.variables['TEMP'][:]

    # practical salinity (in PSU)
    S = ds.variables['PSAL'][:]

    # pressure 
    P = ds.variables['PRES'][:]

    lat = ds.variables['LATITUDE'][:]
    long = ds.variables['LONGITUDE'][:]
    
    # julian day of the profile (JULD:units = "days since
    # 1950-01-01 00:00:00 UTC")
    t = ds.variables['JULD'][:]
    
    # reference date/time of profile
    dt = '1950-01-01 00:00:00'
    date_ref = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    jd_ref = julian.to_jd(date_ref, fmt='jd')
    time = np.array([julian.from_jd(t[i] + jd_ref) for i in range(len(t))])    
    return T,S,P,lat,long,time

def TSz_to_h_steric(T, S, depth, lat, long):
    # INPUT:
    # S = Salinity                               [g/kg]
    # T = Temperature                            [deg C]
    # z = depth                                  [m]
    # lat = latitude in decimal degrees north    [-90 ... +90]
    # long = longitude in decimal degrees        [ 0 ... +360 ]
    #                                         or [ -180 ... +180 ]
    
    # remove masked values
    S = S[~S.mask]
    T = T[~T.mask]
    depth = depth[~np.isnan(depth)]
    
    nd = len(depth)
    
    # calculate the according pressure at depth
    P = np.zeros(nd)
    for i in range(nd):
        p = gsw.p_from_z(depth[i],lat)
        P[i] = p
       
    # calculate the specific volume of standard sea water (at the 
    # Standard Ocean Salinty, SSO, and at a temperature of 0 degrees C)
    SSO = 35.16504*np.ones(P.shape)
    spvol0 = gsw.specvol(SSO,0,P)
        
    # create land/depth  mask
    tmp = np.copy(S)
    tmp[np.where(ge(tmp,9.9990*(10**19)))] = np.nan
    invspvol0 = lt(tmp,90000)/spvol0
    
    # additionally create a land mask (set to nan) 
    msk = np.ones(nd)
    msk[np.where(ge(sum(np.isnan(tmp)),17))] = np.nan
    
    # apply land mask
    invspvol0 = invspvol0*msk
            
    # calculate absolute salinity from PSU
    SA = gsw.SA_from_SP(S,P,long,lat)
            
    # calculate specific volume anomaly
    sptmp = gsw.specvol_anom_standard(SA,T,P)
                
    # apply mask and set other nan's to zero
    sptmp[np.where(np.isnan(sptmp))] = 0 # set nans to zero
    spvol = sptmp*msk
                
    # divide by the specific volume of the corresponding standard sea
    # water column (and apply a depth mask at the same time)
    spvol = spvol*invspvol0

    # integrate over the entire column
    # integrate density over depth with the trapezium rule
    sth = np.trapz(spvol,-depth)       
    return sth


# Version 2
def TSz_to_h_steric2(T, S, depth, lat, long):
    # INPUT:
    # S = Salinity                               [g/kg]
    # T = Temperature                            [deg C]
    # z = depth                                  [m]
    # lat = latitude in decimal degrees north    [-90 ... +90]
    # long = longitude in decimal degrees        [ 0 ... +360 ]
    #                                         or [ -180 ... +180 ]
    
    # remove masked values
    S = S[~S.mask]
    T = T[~T.mask]
    depth = depth[~np.isnan(depth)]
    
    nd = len(depth)
    
    # calculate the according pressure at depth
    P = np.zeros(nd)
    for i in range(nd):
        p = gsw.p_from_z(depth[i],lat)
        P[i] = p
    
    
    # calculate absolute salinity from PSU
    SA = gsw.SA_from_SP(S,P,long,lat)

    # Standard Ocean Salinty (SSO) [g/kg]
    SSO = 35.16504*np.ones(S.shape)
    
    # constant sea water density [kg/m³]
    rhoSea = 1029 
    
    rho = gsw.rho(SA,T,P)
    rho0 = gsw.rho(SSO,0,P)
    
    # calculate steric height changes: integrate over the entire column
    # integrate density over depth with the trapezium rule
    sth = -1/rhoSea * np.trapz(rho-rho0,-depth)
    return sth

