import awkward as ak
import numpy as np
import math
import scipy as sp
from scipy import integrate
from scipy.optimize import curve_fit as cf
from tqdm import tqdm
import ProjectPackage.fit as fit
import ProjectPackage.DataExtraction as de
import ProjectPackage.Kinematic as km
import ProjectPackage.Minv as minv
import ProjectPackage.errFit as errFit

def globalfunc(x, N, a, n, xb, sig, const1, slope1, const2, slope2, *argv):
    JPsi = fit.crystal_ball([N, a, n, xb, sig], x)
    Bg  = fit.expo([const1, slope1], x) + fit.expo([const2, slope2], x)
    return JPsi + Bg

def myCB(x,N, a, n, xb, sig,*argv):
    return fit.crystal_ball([N, a, n, xb, sig], x)

def my2CB(params, x):
  """ A Gaussian curve between two power-laws. Used in
  physics to model lossy processes.
  See http://en.wikipedia.org/wiki/Crystal_Ball_function
  Note that the definition used here differs slightly. At the time of this
  writing, the wiki article has some discrepancies in definitions/plots. This
  definition makes it easier to fit the function by using complex numbers
  and by negating any negative values for a and n.

  This version of the crystal ball is normalized by an additional parameter.
  params: N, a, n, xb, sig
  """
  x = x+0j # Prevent warnings...
  N, a1,a2, n1,n2, xb, sig = params
  if a1 < 0:
    a1 = -a1
  if n1 < 0:
    n1 = -n1
  if a2 < 0:
    a2 = -a2
  if n2 < 0:
    n2 = -n2
  aa1 = abs(a1)
  aa2 = abs(a2)
  A1 = (n1/aa1)**n1 * fit.exp(- aa1**2 / 2)
  B1 = n1/aa1 - aa1
  A2 = (n2/aa2)**n2 * fit.exp(- aa2**2 / 2)
  B2 = n2/aa2 - aa2
  total = 0.*x
  total += ((x-xb)/sig  > -a1)*((x-xb)/sig < a2) * N * fit.exp(- (x-xb)**2/(2.*sig**2))
  total += ((x-xb)/sig <= -a1) * N * A1 * (B1 - (x-xb)/sig)**(-n1)
  total += ((x-xb)/sig >= a2) * N * A2 * (B2 + (x-xb)/sig)**(-n2)
  try:
    return total.real
  except:
    return total
  return total

def my2CB2cf(x,N,a1,n1,a2,n2,xb,sig,*argv):
    return my2CB([N, a1, n1, a2, n2, xb, sig], x)

def globalfunc2CB(x, N, a1, a2, n1, n2, xb, sig, const1, slope1, const2, slope2, *argv):
    JPsi = my2CB([N, a1, a2, n1, n2, xb, sig], x)
    Bg  = fit.expo([const1, slope1], x) + fit.expo([const2, slope2], x)
    return JPsi + Bg

# Fit double crystal ball
def fit2CB(xdata,ydata,rangeinf,rangesup): 
    xf = xdata[ak.where(xdata >= rangeinf)]
    xf = xf[ak.where(xdata <= rangesup)]
    yf = ydata[ak.where(xdata >= rangeinf)]
    yf = yf[ak.where(xdata <= rangesup)]
    xf = xf[ak.where(yf>0)]
    yf = yf[ak.where(yf>0)]
    # Erreur sur les données à fiter
    sigmaf = np.sqrt(yf)
    sigmaf = sigmaf[ak.where(sigmaf>0)]
    xf = xf[ak.where(sigmaf>0)]
    yf = yf[ak.where(sigmaf>0)]
    # Fit Crystal ball
    paramsf, covf = cf(my2CB2cf,xdata=xf, ydata=yf,sigma=sigmaf)
    yfCB = my2CB(paramsf[0:7],xf)
    Nraw = integrate.simpson(yfCB,xf,dx=0.00001)
    paramsferr = np.sqrt(np.diag(covf))
    return xf,yf,sigmaf,yfCB,paramsf,covf,paramsferr,Nraw

# Fit sur les données Monte Carlo, reçoit et retourne un dictionnaire
def fitmcCB2dict(dict_hist):
    (rangeinf,rangesup)=(0.,0.)
    fitCBdata = {'xfmc': [None]*7, 'yfmc': [None]*7,'sigmafmc': [None]*7, 'yfCBmc': [None]*7, 'paramsfmc': [None]*7, 'covfmc': [None]*7, 'paramsferrmc': [None]*7, 'MJPsimc': [None]*7,'errMJPsimc': [None]*7, 'Nrawmc': [None]*7, 'errNrawmc': [None]*7}
    idx = 1
    for i in tqdm(range(0,7)):
        print(i)
        if i>5: idx = 2
        if i== 0:
            (rangeinf,rangesup) = (1.8,4.)
        if i==1: 
            (rangeinf,rangesup) = (1.8,3.9)
        if i==2:
            (rangeinf,rangesup) = (1.6,4.)
        if i==3: 
            (rangeinf,rangesup) = (1.6,3.6)
        if i==4: 
            (rangeinf,rangesup) = (1.6,3.6)
        if i==5: 
            (rangeinf,rangesup) = (1.6,3.6)
        if i==6: 
            (rangeinf,rangesup) = (2.9,3.3)
        fitCBdata['xfmc'][i],fitCBdata['yfmc'][i],fitCBdata['sigmafmc'][i],fitCBdata['yfCBmc'][i],fitCBdata['paramsfmc'][i],fitCBdata['covfmc'][i],fitCBdata['paramsferrmc'][i],fitCBdata['Nrawmc'][i] = fit2CB(dict_hist[(i,i+idx)][1][0:-1],dict_hist[(i,i+idx)][0],rangeinf,rangesup)
        fitCBdata['MJPsimc'][i] = fitCBdata['paramsfmc'][i][5]
        fitCBdata['errMJPsimc'][i] = fitCBdata['paramsferrmc'][i][5]
    return fitCBdata

# Fonction qui fait tourner le fit sur sur les données expérimentales pour chaque range de pt , reçoit un dictionnaire et les paramètres des CB issues des fits MC pour chaque range, retourne un dictionnaire
def fitCB2dict(dict_hist,paramsfmc):
    
    fitCBdata = {'xf': [None]*7, 'yf': [None]*7,'rangef': [None]*7,'sigmaf': [None]*7, 'yfGlob': [None]*7, 'yfCB': [None]*7, 'paramsfmc': paramsfmc, 'paramsf': [None]*7, 'covf': [None]*7, 'paramsferr': [None]*7, 'MJPsi': [None]*7,'errMJPsi': [None]*7, 'Nraw': [None]*7, 'errNraw': [None]*7}
    idx = 1
    i6 = 0
    for i in tqdm(range(0,7)):
        print(i)
        if i>5: idx = 2
        if i== 0:
            (rangeinf,rangesup) = (1.8,4.)
        if i==1: 
            (rangeinf,rangesup) = (1.8,3.9)
        if i==2:
            (rangeinf,rangesup) = (1.6,4.)
        if i==3: 
            (rangeinf,rangesup) = (1.6,3.6)
        if i==4: 
            (rangeinf,rangesup) = (1.6,3.6)
        if i==5: 
            (rangeinf,rangesup) = (1.6,4.)
        if i==6: 
            i6 = -1
            (rangeinf,rangesup) = (1.5,4.)
        fitCBdata['xf'][i],fitCBdata['yf'][i],fitCBdata['sigmaf'][i],fitCBdata['yfGlob'][i],fitCBdata['yfCB'][i],fitCBdata['paramsf'][i],fitCBdata['covf'][i],fitCBdata['paramsferr'][i],fitCBdata['Nraw'][i],fitCBdata['errNraw'][i] = fit2CB2exp(dict_hist[(i,i+idx)][1][0:-1],dict_hist[(i,i+idx)][0],rangeinf,rangesup,fitCBdata['paramsfmc'][i+i6],p1CB=0)
        fitCBdata['MJPsi'][i] = fitCBdata['paramsf'][i][5]
        fitCBdata['errMJPsi'][i] = fitCBdata['paramsferr'][i][5]
    return fitCBdata

# Fit des données expérimentales pour un jeu de données, reçoit les données, les bornes de fit et les paramètres d'entrée de la CB. Le paramètre p1CB devait servir à comparer les fit avec une one sided crystal ball
def fit2CB2exp(xdata,ydata,rangeinf,rangesup,paramsCB,p1CB=0):
    # Extraction des données sur le range du fit
    xf = xdata[ak.where(xdata > rangeinf)]
    xf = xf[ak.where(xdata < rangesup)]
    yf = ydata[ak.where(xdata > rangeinf)]
    yf = yf[ak.where(xdata < rangesup)]
    xf = xf[ak.where(yf>0)]
    yf = yf[ak.where(yf>0)]
    # Erreur sur les données à fiter
    sigmaf = np.sqrt(yf)
    sigmaf = sigmaf[ak.where(sigmaf>0)]
    xf = xf[ak.where(sigmaf>0)]
    yf = yf[ak.where(sigmaf>0)]
    # Initialisation des paramètres de fit
    ## Calcul grossier de la pente et de la constante de part et d'autre des points d'inflexion de la CB
    slope1 = math.log(yf[0]/yf[ak.where(xf<(paramsCB[5]-paramsCB[1]*paramsCB[6]))][-1])/(xf[0]-xf[ak.where(xf<(paramsCB[5]-paramsCB[1]*paramsCB[6]))][-1])
    slope2 = math.log(yf[ak.where(xf>=(paramsCB[5]-paramsCB[2]*paramsCB[6]))][0]/yf[-1])/(xf[ak.where(xf>=(paramsCB[5]-paramsCB[2]*paramsCB[6]))][0]-xf[-1])
    const1 = math.log(yf[0])-slope1*xf[0]
    const2 = math.log(yf[-1])-slope2*xf[-1]
    paramsExpo = [const1,slope1,const2,slope2]
    paramsCB[0] = yf.max()
    
    # One sided crystal ball ou double sided
    if p1CB != 0:
        paramsf, covf = cf(globalfunc,xf, yf, p0=[paramsCB[0], paramsCB[1], paramsCB[2], paramsCB[3], paramsCB[4], paramsExpo[0],paramsExpo[1],paramsExpo[2],paramsExpo[3]],sigma=sigmaf,method='dogbox')

        yfGlob = globalfunc(xf,paramsf[0],paramsf[1],paramsf[2],paramsf[3],paramsf[4],paramsf[5],paramsf[6],paramsf[7],paramsf[8])
        yfCB = fit.crystal_ball(paramsf[0:5],xf)
    else:
        paramsf, covf = cf(globalfunc2CB,xf, yf, p0=[paramsCB[0], paramsCB[1], paramsCB[2], paramsCB[3], paramsCB[4], paramsCB[5], paramsCB[6], paramsExpo[0],paramsExpo[1],paramsExpo[2],paramsExpo[3]],sigma=sigmaf,method='dogbox')

        yfGlob = globalfunc2CB(xf,paramsf[0],paramsf[1],paramsf[2],paramsf[3],paramsf[4],paramsf[5],paramsf[6],paramsf[7],paramsf[8],paramsf[9],paramsf[10])
        yfCB = my2CB(paramsf[0:7],xf)
    
    Nraw = integrate.simpson(yfCB,xf,dx=0.00001)
    paramsferr = np.sqrt(np.diag(covf))
    Nrawerr = errFit.errNraw(paramsf[0:7],paramsf[0:7],rangeinf,rangesup)
    return xf,yf,sigmaf,yfGlob,yfCB,paramsf,covf,paramsferr,Nraw,Nrawerr