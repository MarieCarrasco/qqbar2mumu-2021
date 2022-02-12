import os
import math
import pickle
import awkward as ak
import pandas as pd
import ProjectPackage.Minv as minv

""" Marie """

def savedf(df,run):
    df.to_csv (r'./Saved/df'+str(run)+'.csv')

def readcsv(run):
    return pd.read_csv('./Saved/df'+str(run)+'.csv',header=[0], index_col=[0,1])

def read_dict_hist(filename,p=0):
    path = './Data/'
    if (p!=0):
        path = path + 'MC_'

    with open(path+filename+'_histograms.pkl', 'rb') as f:
        dict_hist = pickle.load(f)
    return dict_hist




