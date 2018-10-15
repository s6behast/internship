# -*- coding: utf-8 -*-
"""
Copyright S. Eichstaedt, F. Schmaehling (PTB) 2013
sascha.eichstaedt@ptb.de

This software is licensed under the BSD-like license:

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the distribution.

This software was developed at Physikalisch-Technische Bundesanstalt.
The software is made available 'as is' free of cost. PTB assumes
no responsibility whatsoever for its use by other parties, and makes no
guarantees, expressed or implied, about its quality, reliability, safety,
suitability or any other characteristic. In no event will PTB be liable
for any direct, indirect or consequential damage arising in connection
with the use of this software.

Citation when using this software is:

Eichstaedt, Schmaehling, Wuebbeler, Anhalt, Buenger, Krueger & Elster
'Comparison of the Richardson-Lucy method and a classical approach for 
spectrometer bandwidth correction', Metrologia vol. 50, 2013
DOI: 10.1088/0026-1394/50/2/107


For guidance on using this software see the documentation files.
In case of questions or suggestions contact Sascha.Eichstaedt@ptb.de

This module contains the following functions:

* RichLucy
* EstMeas
* Interpolate2Equidist
* Interpolate2SameStep
* InterpolateBandpass
* LoadFromFile
* RichLucy
* RichLucyUncertainty
* RichLucyDeconv
* fix_inputs
* fix_raw_inputs
* browseRL

"""

version = 2.0

min_auto_iter = 4 # smallest number of iterations for automatic stopping
rel_tol_zero  = 1e-6 # relative tolerance for measured spectrum values for judgement of automatic stopping

import numpy as np
import sys
from scipy.interpolate import interp1d

try:
    from pyxll import xl_func,xl_macro
except ImportError:
    # pyXLL is not available - use dummy decorator functions instead    
    def xl_func(signature):
        def dummy(func):
            return func
        return dummy
    def xl_macro(signature):
        def dummy(func):
            return func
        return dummy


class stopping_rule(object):    
    "Base class to set up interface for use in RichLucy"
    
    crit_measure = None     # measure underlying the stopping rule  (e.g. rms)
    maxiter = None          # maximum number of RL iterations taken
    display = True          # whether to print text to the cmd line
    min_auto_iter = 5       # number of minimal required RL iterations
    name = "(name not defined)"
    
    def calc_stoppoint(self,SH):
        """
        Main function to be called for application of stopping rule. Returns
        integer value indicating index of optimal RichLucy result in SH.
        
        :param SH:  numpy.ndarray of shape (K,N) with K the number of RL iterations
                    and N the number of data points        
        :return ind: integer indicating index of best estimate in SH
        """
        return -1

    def output(self,text):
        "Function to print text to command line if display variable is True"
        if self.display:
            sys.stdout.write(text)
            
    def plot(self,fig_nr=1):
        "Function to plot result of stopping rule."        
        raise NotImplementedError("Plotting not implemented")


class smoothing(object):
    "Base class to set up interface for use in RichLucy"
    
    name = "(name not defined)"
    display = False
    tau = 0.0

    def smooth(self,Shat):
        """
        Main function to be called for application of the smoothing method. Returns
        a smoothed version of the RichLucy estimate.
        """
        return Shat
    
    def output(self,text):
        "Function to print text to command line if display variable is True"
        if self.display:
            sys.stdout.write(text)
            



###################################### The actual Richardson-Lucy method #################################

def RichLucy(b,delta,M,maxiter=500,autostop=True,display=True,stoprule=None,smooth_method=None,returnAll=False,initialShat=None):
    """
    Richardson Lucy iteration under the assumption that the sampling of b and M is equidistant and
    that the wavelength step size of b and M is equal.
    
    :param b: ndarray of shape (N,) with the bandpass function values
    :param delta: float, wavelength step size for b and M
    :param M: ndarray of shape (L,) with the measured spectrum values
    
    optional parameters
    
    :param maxiter: (optional) maximum number of iterations, default is 500
    :param autostop: (optional) boolean whether to automatically find optimal iteration number (default: True)
    :param display: (optional) boolean whether to print information to the command line (default: True)
    :param stoprule: (optional) a stopping_rule object to calculate the optimal iteration number (default: None)
    :param smooth_method: (optional) smoothing object to smooth each RichLucy iteration (e.g., Tikhonov)
    :param returnAll: (optional) boolean whether to return all intermediate estimates (default: False)
    :param initialShat: (optional) ndarray of shape (L,) with an initial estimate of the input spectrum (default: None)
    
    Output:    
    
    :returns Shat:, ndarray of shape (L,) - the estimated spectrum
    :returns SH: (if returnAll=True) ndarray of shape(maxiter+1,L) of all intermediate results
    """

    def output(text):
        if display:
            sys.stdout.write(text)
        
    output('\n -------------- Richardson-Lucy algorithm version ' + repr(version) + ' --------------\n')
    
    if np.count_nonzero(M<0)>0 or np.count_nonzero(b<0)>0:
        raise ValueError("Measured spectrum and bandpass function must not have negative values.")

    if np.abs(np.trapz(b,dx=delta)-1)>1e-4:
        raise ValueError("Line-spread function must be normalized.")

    if issubclass(type(stoprule),stopping_rule):
        calc_stop = True
        stoprule.display=display
        if maxiter < stoprule.min_auto_iter:
            output('Warning: Maximum number of iterations is too small. Automatic stopping not available.\n')
            autostop = False            
    else:
        output("No stopping rule defined. Taking maximum number of iterations instead.\n")
        calc_stop = False
        autostop = False
        
    if issubclass(type(smooth_method),smoothing):
        output("Smoothing the RL estimates using %s with tau=%g\n" % (smooth_method.name,smooth_method.tau))
        smooth = smooth_method.smooth
    else:
        smooth = lambda x: x
        
# transform continuous b to discrete (impulse invariance technique)
    bd = b*delta
    bdtilde=bd[::-1]
        
   
# adjust length of M and b for proper convolution
    if M.size < b.size:
        sys.stdout.write("The bandpass function seems to be defined over a larger wavelength region than the actual measurement.\n")
        sys.stdout.write("Padding the missing spectrum values with zeros, but results may be inaccurate\n")
        leng_diff = b.size - M.size        
        M = np.concatenate((np.zeros(leng_diff),M,np.zeros(leng_diff)))
        padded = True
    else:
        padded = False

# allocate computer memory 
    if autostop or calc_stop or returnAll:
        saveAll = True
    else:
        saveAll = False
    
    if saveAll:
        SH = np.zeros((maxiter+1,M.size))

# initial 'iteration'
    r = 0
    if isinstance(initialShat,np.ndarray):
        if len(initialShat) == M.size:
            if saveAll:
                SH[r,:] = initialShat
            else:
                SH = initialShat
        else:
            output("User-supplied initial estimate does not match measured spectrum and will be ignored.\n")
            if saveAll:
                SH[r,:] = M[:]
            else:
                SH = M[:]
    else:
        if saveAll:
            SH[r,:] = M[:];  
        else:
            SH = M[:]
  
# additional iterations
    if autostop:
        output('Richardson-Lucy method calculating optimal stopping point using ' + stoprule.name + "\n")
        output('Step 1: Carry out ' + repr(int(maxiter)) + ' iterations.\n')
    else:
        output('Richardson-Lucy method with ' + repr(int(maxiter)) + ' iterations\n')
#######################    original RL iterations
    while r < maxiter:
        r = r+1
    # actual RichLucy step
        if saveAll:
            tmp1 = np.convolve(SH[r-1,:],bdtilde,'same')
        else:
            tmp1 = np.convolve(SH,bdtilde,'same')
        tmp1[tmp1!=0] = M[tmp1!=0]/tmp1[tmp1!=0]
        
        # if anything went wrong during convolution - set to zero
        count_all = np.count_nonzero(tmp1==0)
        tmp1[np.isnan(tmp1)] = 0; tmp1[np.isinf(tmp1)] = 0   
        count_bad = np.count_nonzero(tmp1==0)-count_all
        if display and count_bad>0:
            sys.stdout.write('After convolution in RL iteration %d, %d estimated values were set to zero. \n' % (r,count_bad))
        
        tmp2 = np.convolve(tmp1,bdtilde[::-1],'same')
        if saveAll:            
            Shat_new = smooth(SH[r-1,:])*tmp2
        else:
            Shat_new = smooth(SH)*tmp2
               
        Shat_new[np.isnan(Shat_new)]=0; Shat_new[np.isinf(Shat_new)]=0
        if saveAll:
            SH[r,:]=Shat_new
        else:
            SH = Shat_new
        if np.mod(r,maxiter*0.05)==0 and display:
          sys.stdout.write(".")
######################
    if padded:
        if saveAll:
            SH = SH[:,leng_diff:-leng_diff]
        else:
            SH = SH[leng_diff:-leng_diff]
 
    output(' Done.\n')
    if autostop:
        output('Step 2: Calculating optimal stopping point\n')
            
    if calc_stop:
        indopt = stoprule.calc_stoppoint(SH)
    else:
        indopt = maxiter

    if indopt<0 and autostop:
        output('Warning: Calculation of optimal stopping failed. Using measured spectrum as best estimate.\n')
        Shat = M[:]
    else:
        if autostop:
            if display:
                print('Optimal number of iterations = ' + repr(int(indopt)+1))
            Shat = np.squeeze(SH[indopt-1,:])
        else:
            if saveAll:
                Shat = SH[-1,:]
            else:
                Shat = SH
           
    if returnAll:
        return Shat, SH
    else:
        return Shat
    
    
    
    
    
    
    
######################### some helper functions for several tasks    
    

def LoadFromFile(filename):
    """
    Load wavelength scale axis, measured values and uncertainties from file.
    Output of this method is the matrix "data".
    
    :param filename: string, name of the file to load (including full path)
    
    :returns: data matrix
    
    :raises: `ValueError` if data file could not be loaded.
    """
   
    try:
        data = np.loadtxt(filename,comments='#')
    except ValueError:
        try:
            data = np.loadtxt(filename,comments='#',delimiter=',')
        except ValueError:
            try:
                data = np.loadtxt(filename,comments='#',delimiter=';')
            except ValueError:
                raise ValueError("Data file could not be loaded." \
                "Please provide data column-wise and separated by whitespaces. " \
                "For further details see the documentation files.\n")
    return data
    

@xl_func("numpy_column lambda, numpy_column values: numpy_array")
def InterpolateBandpass(lamb_axis,values):
    """
    Interpolation of bandpass values (line spread function). This is just
    a wrapper to allow for a convenient function call from EXCEL.
    """
    newAx, newVals = Interpolate2Equidist(lamb_axis,values)[:2]

    result = np.vstack((newAx,newVals))
    result = result.transpose()
    return result
    

@xl_func("numpy_column lambda_b, numpy_column b: var")
def checkBandpass(lambda_b,b,print_output=False):
    """
    For given bandpass values b and corresponding wavelengths lambda_b, this
    function determines whether::
    * the lambda axis is strictly monotone increasing
    * the lambda axis is equidistant
    * the bandpass is normalized to integrate to 1
     
    :param lambda_b: ndarray of shape (N,) with wavelengths for the bandpass function
    :param b: ndarray of shape (N,) with corresponding bandpass values
    
    :returns: result - a list: [lambda monotonically increasing?, is equidistant?, is normalized?, delta_b]
    """
    
    #adjust range of values
    inds = np.nonzero(lambda_b>0.0)
    lambda_b = lambda_b[inds]
    b = b[inds]
    
    #check monotonicity
    if not np.all(np.sort(lambda_b) == lambda_b) or np.count_nonzero((np.diff(lambda_b)<1e-14)) > 0:
        check_mono = "False"
    else:
        check_mono = "True"
    #check equidistance
    if np.max(np.diff(lambda_b)) - np.min(np.diff(lambda_b)) > 1e-5:
        check_equi = "False"
    else:
        check_equi = "True"
    #check normalization
    bn = np.trapz(b,lambda_b)
    if np.abs(1-bn) > 1e-5:
        check_norm = "False (integral = %1.4e)" % bn
    else:
        check_norm = "True"
        
    delta = np.min(np.abs(np.diff(lambda_b)))
    
    if print_output:
        print "\nResult of bandpass check:"
        print "wavelength scale values are strictly monotonically increasing?\t" + repr(check_mono)
        print "wavelength scale values are spaced equidistantly?\t" + repr(check_equi)
        print "bandpass is normalized?\t" + repr(check_norm)
        print "wavelength scale axis (minimum) spacing is:\t %g" %delta
         
    result = [[check_mono],[check_equi],[check_norm],[delta]]
    return result
    
    

@xl_func("numpy_column lambda_M, numpy_column M: var")
def checkSpectrum(lambda_M,M,print_output=False):
    """
    For given vector of measured spectrum values and corresponding wavelengths
    this function determines whether::
    * the lambda axis is strictly monotonically increasing
    * the lambda axis is equidistant
     
    :param lambda_M: ndarray of shape (N,) with wavelengths for the measured spectrum
    :param M: ndarray of shape (N,) with corresponding spectrum values
    
    :returns: result - the list: [lambda_M monotonically increasing?, is equidistant?, delta_M]
    """
    
    #adjust range of values
    inds = np.nonzero(lambda_M>0.0)
    lambda_M = lambda_M[inds]
    M = M[inds]

    #check monotonicity
    if not np.all(np.sort(lambda_M) == lambda_M) or np.count_nonzero((np.diff(lambda_M)<1e-14)) > 0:
        check_mono = "False"
    else:
        check_mono = "True"
    #check equidistance
    if np.max(np.diff(lambda_M)) - np.min(np.diff(lambda_M)) > 1e-5:
        check_equi = "False"
    else:
        check_equi = "True"
        
    delta = np.min(np.diff(lambda_M))        
    
    if print_output:
        print "\nResult of checkSpectrum:"
        print "wavelength scale values are strictly monotonically increasing?\t" + repr(check_mono)
        print "wavelength scale values are spaced equidistantly?\t" + repr(check_equi)
        print "wavelength scale (minimal) spacing: \t %g" % delta 
        
    result = [[check_mono],[check_equi],[delta]]
    return result
       



def Interpolate2Equidist(lamb_axis,values,display=True,interp_kind='cubic',tol_equi=1e-5):
    """
    Interpolation of measured values if wavelength scale axis is not equidistant.
    Input of this function are the original measured values and the corresponding
    wavelength scale. The output is the interpolated version with the new
    scale, a flag whether interpolated or not and a tuple (numNaN,numNeg) 
    with the number of NaN and negative values after interpolation.
    
    :param lamb_axis: ndarray of shape (N,) with wavelength values
    :param values: ndarray of shape (N,) with measured values
    
    optional parameters
    
    :param display: boolean whether to print information to the command line (default:True)
    :param interp_kind: string which is handed over to the scipy interp1d function (default:'cubic')
    :param tol_equi: float, tolerance value to judge whether wavelength axis is equidistant (default:1e-5)
    
    Outputs    
    
    :returns: newAx, ndarray of shape (L,) with equidistant wavelength scale
    :returns: newVals, ndarray of shape (L,) with corresponding interpolated spectrum values
    :returns: did_interp, boolean whether interpolation was actually necessary 
    :returns: numNaN, numNeg, ints - the number of NaN and negative values after interpolation
    
    """    

    if display:
        print "\nCheck for equidistant wavelength scale axis "

    if not np.all(np.sort(lamb_axis) == lamb_axis) or np.count_nonzero(lamb_axis)<1e-14>0:
        print "Error in Interpolate2Equidist: Wavelength scale is not strictly monotonically increasing!"
        return

    if np.max(np.diff(lamb_axis)) - np.min(np.diff(lamb_axis)) > tol_equi:
        delta = min(np.diff(lamb_axis))        
        if display:
            print "Wavelength scale is not equidistant."\
                  "New wavelength scale step size is %1.4f. Interpolating..." % delta
            
        newAx = np.arange(lamb_axis[0],lamb_axis[-1],delta)
        Intfunc = interp1d(lamb_axis,values,kind=interp_kind,bounds_error=False)
        newVals = Intfunc(newAx)
        numNaN = np.count_nonzero(np.isnan(newVals))
        numNeg = np.count_nonzero(newVals<0)
        newVals[np.nonzero(np.isnan(newVals))]=0.0
        newVals[newVals<0] = 0.0
        if display:
            print "Interpolation resulted in %d NaN(s) and %d negative value(s)." % (numNaN,numNeg)
            if numNaN+numNeg>0:
                print "All set to zero."
        did_interp = True
    else:
        if display:
            print "Wavelength scale axis is equidistant. No interpolation necessary."
        newAx = lamb_axis
        newVals  = values
        numNaN = 0
        numNeg = 0
        did_interp = False
    if display:   
        print ""        
    
    return newAx, newVals, did_interp, numNaN, numNeg
    
    
    
def Interpolate2SameStep(M,lambda_M,delta_b,display=True,interp_kind='cubic',tol_equi=1e-5):
    """
    Interpolation of measured spectrum if its wavelength step size is not equal to that
    of the bandpass function or if it is not equidistant. The output is the interpolated version with the new scale, 
    a flag whether interpolated or not and a tuple (numNaN,numNeg) with the number of NaN and negative values after interpolation.
    
    :param M: ndarray of shape (N,) with the measured spectrum values
    :param lambda_M: ndarray of shape (N,) with the corresponding wavelengths
    :param delta_b: float, the wavelength step size for the bandpass function
    
    optional parameters
    
    :param display: boolean, whether to print information to the command line
    :param interp_kind: string, which is passed to the scipy interp1d function
    :param tol_equi: float, tolerance for judgin whether the wavelength step sizes are equal
    
    Outputs
    
    :returns: lm_interp - ndarray of shape (K,) of new wavelength values for measured spectrum
    :returns: M_interp - ndarray of shape (K,) of corresponding interpolated spectrum values
    :returns: did_interp - boolean whether interpolation was actually necessary
    :returns: numNaN,numNeg - ints, the number of NaN and negative values after interpolation
    
    """    
    if display:
        print "\nCheck for equidistant wavelength scale axis"

    if not np.all(np.sort(lambda_M) == lambda_M) or np.count_nonzero((np.diff(lambda_M)<1e-14)) >0:
        raise ValueError("Error in Interpolate2SameStep: Wavelength scale is not strictly monotonically increasing!")
    
    deltaM = min(np.diff(lambda_M))   
    
    if np.abs(deltaM-delta_b) > tol_equi or np.max(np.diff(lambda_M)) - np.min(np.diff(lambda_M)) > tol_equi:
        if display:
            print "\nWavelength step size in measured spectrum requires interpolation.\n"\
                  "New step size is %1.4f. Interpolating..." % delta_b
        lm_interp = np.arange(lambda_M[0],lambda_M[-1],delta_b)
        spline_M   = interp1d(lambda_M,M,bounds_error=False,kind=interp_kind)        
        M_interp = spline_M(lm_interp)
        numNaN = np.count_nonzero(np.isnan(M_interp))
        numNeg = np.count_nonzero(M_interp<0)
        M_interp[np.nonzero(np.isnan(M_interp))] = 0.0
        M_interp[M_interp<0] = 0.0
        if display:
            print "Interpolation resulted in %d NaN(s) and %d negative value(s)." % (numNaN,numNeg)
            if numNaN+numNeg>0:
                print "All set to zero."
        did_interp = True
    else:
        if display:
            print "Wavelength scale step size equal and equidistant. No interpolation necessary."
        lm_interp = lambda_M
        M_interp  = M
        numNaN = 0
        numNeg = 0
        did_interp = False
    if display:
        print ""
        
    return lm_interp,M_interp,did_interp,numNaN,numNeg        
    
    
def fix_bandpass(lambda_bandpass,bandpass,display=True):    
    """
    
    :param lambda_bandpass: ndarray of wavelength scale values for bandpass
    :param bandpass: ndarray of bandpass values
    
    :returns lb,b: 
    :returns changes: an array of 2 boolean (bandpass interpolated?, measured spectrum interpolated?) 

    """    
    
    _ = checkBandpass(lambda_bandpass,bandpass,print_output=display)
    
    if display:
        print "\nFixing input data:"
    
    if any(bandpass<0):
        if display:
            print "Some values in bandpass are negative...will be set to zero."
        bandpass[bandpass<0] = 0.0
        
    lb,b,interpolated_b = Interpolate2Equidist(lambda_bandpass,bandpass,display=display)[:3]
    deltab = lb[1]-lb[0]        

    intb = np.trapz(b,dx=deltab)
    if np.abs(1-intb)>1e-4:
        b = b/intb
    
    return lb,b,interpolated_b
    
def fix_inputs(lambda_bandpass,bandpass,lambda_spectrum,spectrum,display=True,returnDiagnostics=False):
    """
    For given wavelength scales (bandpass and measurement each), this functions
    verifies whether interpolation of the data is necessary in order to carry
    out the Richardson-Lucy method and returns the interpolated results.
    
    If no interpolation is necessary, then the original data is returned.
    
    :param lambda_bandpass: ndarray of wavelength scale values for bandpass
    :param bandpass: ndarray of bandpass values
    :param lambda_spectrum: ndarray of wavelength scale values for measured spectrum
    :param spectrum: ndarray of measured spectrum amplitudes
    
    :returns lb,b,lM,M: each corrected (interpolated) if necessary
    :returns changes: an array of 2 boolean (bandpass interpolated?, measured spectrum interpolated?) 

    """    
    
    _ = checkBandpass(lambda_bandpass,bandpass,print_output=display)
    _ = checkSpectrum(lambda_spectrum,spectrum,print_output=display)
    
    if display:
        print "\nFixing input data:"
    
    if any(bandpass<0):
        if display:
            print "Some values in bandpass are negative...will be set to zero."
        bandpass[bandpass<0] = 0.0
        
    lb,b,interpolated_b = Interpolate2Equidist(lambda_bandpass,bandpass,display=display)[:3]
    deltab = lb[1]-lb[0]        

    intb = np.trapz(b,dx=deltab)
    if np.abs(1-intb)>1e-4:
        b = b/intb
        
    if any(spectrum<0):
        if display:
            print "Some value in measured spectrum are negative...will be set to zero."
        spectrum[spectrum<0]=0.0
        
    lM,M,interpolated_M = Interpolate2SameStep(spectrum,lambda_spectrum,deltab,display=display)[:3]
    
    if len(M)<len(b):
        if display:
            print "Length of measured spectrum larger than that of bandpass. Is adjusted by zero-padding."
        leng_diff = len(b)-len(M)
        M = np.hstack((np.zeros(int(leng_diff/2)),M))
        M = np.hstack((M,np.zeros(len(b)-len(M))))
        lM= np.hstack((np.arange(lM[0]-((leng_diff/2)+1)*deltab,lM[0],deltab),lM,np.arange(lM[-1]+deltab,lM[-1]+(leng_diff/2)*deltab,deltab)))
    
    if display:
        print "\nFixing of input data finished\n"
        
    if returnDiagnostics:    
        return lb,b,lM,M,[interpolated_b,interpolated_M]
    else:
        return lb,b,lM,M
    
    
    

def fix_raw_inputs(bdata,Mdata,**kwargs):
    """
    Same functionality as fix_inputs, but taking as inputs the matrices which result from reading
    the data text files
    
    :param bdata: ndarray of shape (N,2) or (N,3) with N the number of bandpass values
    :param Mdata: ndarray of shape (K,2) or (K,3) with K the number of spectrum values
    
    :returns lb,b,lM,M: wavelength scales and values interpolated if necessary
    :return changes: an array of 2 boolean (bandpass interpolated?, measured spectrum interpolated?)
    
    """
        
    return fix_inputs(bdata[:,0],bdata[:,1],Mdata[:,0],Mdata[:,1],**kwargs)

    
    
    
def EstMeas(Shat,b,lambda_S,lambda_b,display=False,interpolate_back=True):
    """
    Calculate the measurement result estimated from an estimated input spectrum
    and the bandpass function.
    
    If necessary, the bandpass is interpolated to obtain an equidistantly sampled sequence.
    If necessary, the estimated input spectrum Shat is interpolated to obtain 
    the same step size as for the bandpass
    """
       
    if display:
        print "---- Calculating estimated measurement from bandpass and RichLucy estimate ----- "
       
    lb,b,ls,S,interpolated= fix_inputs(lambda_b,b,lambda_S,Shat,display=display,returnDiagnostics=True)
    deltab = lb[1]-lb[0]
        
    Mhat = np.convolve(S,b[::-1]*deltab,'same')
    
    if interpolated[1] and interpolate_back:
        if display:
            print "Interpolating back to original wavelength scale"
        Mhat_interp = interp1d(ls,Mhat,kind='linear',bounds_error=False,fill_value=0.0)
        Mhat = Mhat_interp(lambda_S)
    else:
        lambda_S = ls

    if display:
        print "-----------------------------------------------------------"

    Mhat[Mhat<0] = 0.0
    
    return Mhat,lambda_S
    


def browseRL(lM,M,lS,SH,iter0=10):
    """
    Opens a new figure window to browse the RichLucy estimates
    
    :param lM: ndarray of shape (N,) of wavelength scale values in measured spectrum
    :param M: ndarray of shape (N,) of measured spectrum amplitudes
    :param lS: ndarray of shape (K,) of wavelength scale values in estimated spectrum
    :param SH: ndarray of shape (maxiter,K) of Richardson-Lucy estimates
    
    """
    from pylab import figure,axes,show,draw
    from matplotlib.widgets import Slider
    from matplotlib.gridspec import GridSpec    
    
    gs = GridSpec(2,1,height_ratios=[1,0.1])
    fig = figure()
    ax = fig.add_subplot(gs[0])
    ax.plot(lM,M,'bo',alpha=0.6,label="measured spectrum")
    iter_text = ax.text(lM[0]+5,max(M)*0.9,"iter %d" % iter0,fontsize=16)
    ax_slide = axes([0.15, 0.1, 0.7, 0.03])
    ax.set_xlabel("wavelength")
    ax.set_ylabel("spectrum amplitude")
    ax.set_ylim(0,M.max()*1.2)
    slider = Slider(ax_slide, 'iter', int(0), int(SH.shape[0]-1), valinit=iter0)
    lineS = ax.plot(lS,SH[iter0,:],'r',label="RichLucy estimate",linewidth=1)
    ax.legend()
    
    def update(value):
        curr_iter = int(value)        
        if curr_iter > SH.shape[0]-1:
            curr_iter = SH.shape[0]-1
        iter_text.set_text("iter %d" % curr_iter)
        lineS[0].set_data(lS,SH[curr_iter,:])
        draw()
        
    slider.on_changed(update)
    show()
    return slider


##############################################################

    

def RichLucyUncertainty(bhat,b_unc,lambda_b,Mhat,M_unc,lambda_M,\
            runs=100,maxncpu=1,maxiter=500,display=False,stoprule=None,smooth_method=None):
    """
    This function carries out uncertainty evaluation for the Richardson-Lucy
    deconvolution by using the GUM Supplement 2 Monte Carlo method. It assumes
    that knowledge about the input quantities (i.e., bandpass and measured spectrum)
    is available in terms of an estimate and a vector of associated standard uncertainties.
    
    TODO: Extend the uncertainty evaluation to arbitrary type of input knowledge.
    
    :param bhat: ndarray of shape (N,) with estimate of bandpass values
    :param b_unc: ndarray of shape (N,) with associated standard uncertainties
    :param lambda_b: ndarray of shape (N,) with corresponding wavelengths
    :param Mhat: ndarray of shape (L,) with measured spectrum values
    :param M_unc: ndarray of shape (L,) with associated standard uncertainties
    :param lambda_M: ndarray of shape (L,) with corresponding wavelengths
    
    optional arguments
    
    :param runs: int - the number of Monte Carlo runs for uncertainty propagation
    :param maxncpu: int - the maximum number of parallel workers for the Monte Carlo
    
    :returns: Shat, UShat - ndarrays of shape (L,) with the mean estimated spectrum and its associated uncertainty, respectively
    
    """
    
    from time import time
    from multiprocessing import Queue,Process,cpu_count
    
    fun_args = (bhat,b_unc,lambda_b,Mhat,M_unc,lambda_M,maxiter,display,stoprule,smooth_method)

    SH = np.zeros((len(Mhat),runs))
    
    """
    The actual call of the RichLucy method has to be in seperate methods
    in order to work with the Queue concept of multiprocessing (functions can't be pickled)
    """

    t0 = time()
    
    task_Queue = Queue()
    result_Queue = Queue()

    ncpu = min( max(cpu_count()-2,1), maxncpu)
    sys.stdout.write('\nStarting ' + repr(runs) + ' Monte Carlo trials with ' + str(ncpu) + ' parallel workers\n')
    
    tasks = [(__callRLparallel, fun_args) for k in range(runs)]

    for task in tasks:
        task_Queue.put(task)

    for i in range(ncpu):
        Process(target=__QueueWorker, args=(task_Queue,result_Queue)).start()
    
    results = [result_Queue.get() for task in tasks]
    
    for i in range(ncpu):
        task_Queue.put('STOP')
    

    dauer = time()-t0

    SH = np.array(results)
    
    Shat = np.mean(SH,axis=0)
    UShat = np.cov(SH.T)
    
   
    if len(Shat) != len(Mhat):
        raise ValueError("Something went wrong in the Monte Carlo method. Nothing will be returned.")
        return

    print "-----------------------------------------------------------"
    print "All done. Monte Carlo with %d runs took about %1.2f seconds" % (runs,np.round(dauer))
    print "-----------------------------------------------------------"
        
    return Shat, UShat


################ helper functions for the Monte Carlo ##################

def __QueueWorker(input,outQueue):
        for func,fun_args in iter(input.get,'STOP'):
            result = func(*fun_args)
            outQueue.put(result)
            
def __callRLparallel(bhat,b_unc,lambda_b,Mhat,M_unc,lambda_M,maxiter,display,stoprule,smooth_method):
        """
         Carries out the individual Monte Carlo trials
        """
        
        rst = np.random.RandomState()        
        
        # draw from the original distributions (without interpolation)
        if len(M_unc.shape)==1 or M_unc.shape[1]==1:
            M = Mhat + rst.randn(Mhat.size)*M_unc.flatten()
        else:
            M = rst.multivariate_normal(Mhat,M_unc)
        if len(b_unc.shape)==1 or b_unc.shape[1]==1:
            b = bhat + rst.randn(bhat.size)*b_unc.flatten()
        else:
            b = rst.multivariate_normal(bhat,b_unc)
        
        lb,b = Interpolate2Equidist(lambda_b,b,display=display)[:2]
        deltab = lb[1]-lb[0]
        lM,M,interpolated = Interpolate2SameStep(M,lambda_M,deltab,display=display)[:3]
        
        M[M<0] = 0.0
        b[b<0] = 0.0

        # normalization of bandpass            
        b = b/np.trapz(b,dx=deltab)
        
        if interpolated:
            if display:
                print "Running RichLucy with interpolated data..."
            Shat_interp = RichLucy(b,deltab,M,display=display,stoprule=stoprule,maxiter=maxiter,smooth_method=smooth_method)
            spline_Shat = interp1d(lM,Shat_interp,bounds_error=False,kind='linear')
            Shat = spline_Shat(lambda_M)
            Shat[np.nonzero(np.isnan(Shat))] = 0.0
            Shat[Shat<0.0] = 0.0            
        else:
            if display:
                print "Running RichLucy..."
            Shat = RichLucy(b,deltab,M,display=display,stoprule=stoprule,maxiter=maxiter,smooth_method=smooth_method)
        return Shat
        
##########################################

def __Excel2NDArray(tupleofTuples):
    # flatten the tuple of tuples to a 1D list
    all_vals = [element for tup in tupleofTuples for element in tup]
    # throw away the None values (corresp. to empty cells in Excel)
    vals = filter(None,all_vals)
    # convert to ndarray
    return np.asarray(vals)
    


@xl_macro("float dummy")
def __RLD_macro(dummy=None):
    import pyxll
    import logging    
    _log = logging.getLogger(__name__)
    
    try:
        import win32com.client
    except ImportError:
        _log.warning("*** win32com.client could not be imported          ***")
        _log.warning("*** the RichLucy Macro is working only under Microsoft Windows  ***")
        
    xl_window = pyxll.get_active_object()
    xl_app = win32com.client.Dispatch(xl_window).Application

    # it's helpful to make sure the gen_py wrapper has been created
    # as otherwise things like constants and event handlers won't work.
    win32com.client.gencache.EnsureDispatch(xl_app)

    lambda_bandpass = __Excel2NDArray(xl_app.Range("A2:A1000").Value)
    bandpass = __Excel2NDArray(xl_app.Range("B2:B1000").Value)
    lambda_spectrum = __Excel2NDArray(xl_app.Range("C2:C1000").Value)
    spectrum = __Excel2NDArray(xl_app.Range("D2:D1000").Value)
    
    val_range = "E2:E%d" % (len(spectrum)+1)
    
    max_it = xl_app.Range("maxiter").Value
    if max_it==None:
        maxiter = 500
    else:
        maxiter = int(xl_app.Range("maxiter").Value)
    curv_range = "F2:F%d" % (maxiter+1)
    
    xl_app.Range(val_range).Value = np.zeros((len(spectrum),1))
    xl_app.Range(curv_range).Value = np.zeros((maxiter,1))

    Shat,RMS,curv = RichLucyDeconv(lambda_bandpass,bandpass,lambda_spectrum,spectrum,maxiter,returnAll=True)

    xl_app.Range(val_range).Value = np.reshape(Shat,(len(Shat),-1))
    
    
    xl_app.Range(curv_range).Value = np.reshape(curv,(len(curv),-1))
    
    

@xl_func("numpy_column lambda_bandpass, numpy_column bandpass, numpy_column lambda_spectrum, numpy_column spectrum, int maxiter: numpy_column")
def RichLucyDeconv(lambda_bandpass,bandpass,lambda_spectrum,spectrum,display=True,**kwargs):
    """
    Application of Richardson-Lucy deconvolution for provided bandpass (line spread function)
    and measured spectrum. This method carries out interpolation if the sampling of bandpass
    and spectrum does not meet the RichLucy assumptions. If necessary, the estimation result
    is returned interpolated to the original wavelength values.
    
    :param lambda_bandpass: ndarray of shape (N,) with wavelengths for the bandpass function
    :param bandpass: ndarray of shape (N,) with corresponding bandpass values
    :param lambda_spectrum: ndarray of shape (L,) with wavelengths for the measured spectrum
    :param spectrum: ndarray of shape (L,) with corresponding measured spectrum values
    
    :returns: Shat - ndarray of shape (L,) with the estimated spectrum values
   
    """    
    if display:
		print '\n\n-------------------------------------------------------'
		print 'Richardson-Lucy algorithm for bandpass correction'   
		print 'S. Eichstaedt, F Schmaehling (PTB) 2013'
		print 'Usage of this software is without any warranty.'
		print '\n-------------------------------------------------------'
    

    lb,b,lm,M,interpolated = fix_inputs(lambda_bandpass,bandpass,lambda_spectrum,spectrum,display=display,returnDiagnostics=True)    
    interpolated_b,interpolated_M = interpolated
   
    deltab = lb[1]-lb[0]        

    centroid = np.trapz(b*lb,dx=deltab)
    if display:
		print 'central wavelength of bandpass function is %1.4f' % (centroid)
    
    lambda_central = np.mean(lb)
    if display:
		print 'central wavelength of provided wavelength scale is %1.4f' % (lambda_central)
    
    shift = centroid - lambda_central
    if shift>1e-1*deltab and display:
        print 'Suggested shift of wavelength scale is %1.4f'% (shift)

    deltam = lm[1]-lm[0]
	
    if display:
		print 'bandpass function:'
		print '   wavelength scale: [%1.3f,%1.3f]' % (lb[0],lb[-1])
		print '   wavelength step: %1.4f' % (deltab)
		print 'measured spectrum:'
		print '   wavelength scale: [%1.3f,%1.3f]' % (lm[0],lm[-1])
		print '   wavelength step: %1.4f' % (deltam)
       
    _ = kwargs.pop("returnAll",False)
    
    if interpolated_M:
        if display:
            print('Measured spectrum required interpolation to meet wavelength step of bandpass.')
        Shat_interp = RichLucy(b,deltab,M,display=display,returnAll=False,**kwargs)
        if display:
            print "RichLucy finished. Interpolating estimate back to original wavelength scale."        
        spline_Shat = interp1d(lm,Shat_interp,bounds_error=False,kind='linear')
        Shat = spline_Shat(lambda_spectrum)
        Shat[np.nonzero(np.isnan(Shat))]=0.0
    else:                
        Shat = RichLucy(b,deltab,M,display=display,returnAll=False,**kwargs)
        
    return Shat


@xl_macro("float dummy")
def __RLDUnc_macro(dummy=None):
    import pyxll
    import logging    
    _log = logging.getLogger(__name__)
    
    try:
        import win32com.client
    except ImportError:
        _log.warning("*** win32com.client could not be imported          ***")
        _log.warning("*** the RichLucy Macro is working only under Microsoft Windows  ***")
        
    xl_window = pyxll.get_active_object()
    xl_app = win32com.client.Dispatch(xl_window).Application

    # it's helpful to make sure the gen_py wrapper has been created
    # as otherwise things like constants and event handlers won't work.
    win32com.client.gencache.EnsureDispatch(xl_app)

    lb = __Excel2NDArray(xl_app.Range("A2:A1000").Value)
    b  = __Excel2NDArray(xl_app.Range("B2:B1000").Value)
    Ub = __Excel2NDArray(xl_app.Range("C2:C1000").Value)
    lM = __Excel2NDArray(xl_app.Range("D2:D1000").Value)
    M  = __Excel2NDArray(xl_app.Range("E2:E1000").Value)
    UM = __Excel2NDArray(xl_app.Range("F2:F1000").Value)
    
    val_range = "G2:G%d" % (len(M)+1)
    unc_range = "H2:H%d" % (len(M)+1)
    
    max_it = xl_app.Range("MC_maxiter").Value
    if max_it==None:
        maxiter = 500
    else:
        maxiter = int(max_it)
        
    MC = xl_app.Range("MCruns").Value
    if MC==None:
        MCruns = 100
    else:
        MCruns = int(MC)
        
    xl_app.Range(val_range).Value = np.zeros((len(M),1))
    xl_app.Range(unc_range).Value = np.zeros((len(M),1))
    
    Res = RichLucyDeconvWithUncertainty(lb,b,Ub,lM,M,UM,MCruns,maxiter)

    xl_app.Range(val_range).Value = np.reshape(Res[:,0],(len(M),-1))
    xl_app.Range(unc_range).Value = np.reshape(Res[:,1],(len(M),-1))
    


@xl_func("numpy_column lambda_b, numpy_column b, numpy_column Ub, numpy_column lambda_M, numpy_column M, numpy_column M, int MCruns, int maxiter: numpy_array")
def RichLucyDeconvWithUncertainty(lb,b,Ub,lm,M,UM,MCruns=100,**kwargs):
    """
    Application of Richardson-Lucy deconvolution for provided bandpass (line spread function)
    and measured spectrum. This method carries out interpolation if the sampling of bandpass
    and spectrum does not meet the RichLucy assumptions. If necessary, the estimation result
    is returned interpolated to the original wavelength values.
    Evaluation of uncertainty associated with the estimation result is carried out using
    GUM-S2 Monte Carlo.
    """  
    
    print '\n\n-------------------------------------------------------'
    print 'Richardson-Lucy algorithm for bandpass correction command line tool'    
    print 'S. Eichstaedt, F Schmaehling (PTB) 2013'
    print 'Usage of this software is without any warranty.'
    print '\n-------------------------------------------------------'

    print 'Uncertainties will be calculated using %d Monte Carlo trials' % MCruns
    print('-------------------------------------------------------\n')


    sys.executable = "C:\Python27\python.exe"

    Shat,UShat = RichLucyUncertainty(b,Ub,lb,M,UM,lm,\
                        runs=MCruns,**kwargs)
    
    uShat = np.sqrt(np.diag(UShat))
   
    result = np.vstack((Shat,uShat))
    
    return result.transpose()



@xl_macro("string x: int")
def py_strlen(x):
    """returns the length of x"""
    return len(x)


@xl_macro("numpy_column lambda_bandpass, numpy_column bandpass, numpy_column lambda_spectrum, numpy_column spectrum, int maxiter: numpy_column")
def RichLucyDeconvMacro(lambda_bandpass,bandpass,lambda_spectrum,spectrum,**kwargs):
    """
    Application of Richardson-Lucy deconvolution for provided bandpass (line spread function)
    and measured spectrum. This method carries out interpolation if the sampling of bandpass
    and spectrum does not meet the RichLucy assumptions. If necessary, the estimation result
    is returned interpolated to the original wavelength values.
    """    

    print '\n\n-------------------------------------------------------'
    print 'Richardson-Lucy algorithm for bandpass correction command line tool'    
    print 'S. Eichstaedt, F Schmaehling (PTB) 2013'
    print 'Usage of this software is without any warranty.'
    print '\n-------------------------------------------------------'
    
    lb,b,interpolated = Interpolate2Equidist(lambda_bandpass,bandpass)[:3]
    deltab = lb[1]-lb[0]        

    intb = np.trapz(b,dx=deltab)
    if np.abs(1-intb)>1e-4:
        print 'Bandpass function is not normalized...doing it now.'
        b = b/intb
        
    centroid = np.trapz(b*lb,dx=deltab)
    print 'central wavelength of bandpass function is %1.4f' % (centroid)
    
    lambda_central = np.mean(lb)
    print 'central wavelength of provided wavelength scale is %1.4f' % (lambda_central)
    
    shift = centroid - lambda_central
    if shift>1e-3:
        print 'Suggested shift of wavelength scale is %1.4f'% (shift)


    lm,M,interpolated_M = Interpolate2SameStep(spectrum,lambda_spectrum,deltab)[:3]
    deltam = lm[1]-lm[0]
    
    print 'bandpass function:'
    print '   wavelength scale: [%1.3f,%1.3f]' % (lb[0],lb[-1])
    print '   wavelength step: %1.4f' % (deltab)
    print 'measured spectrum:'
    print '   wavelength scale: [%1.3f,%1.3f]' % (lm[0],lm[-1])
    print '   wavelength step: %1.4f' % (deltam)
        
    if interpolated_M:
        print('Measured spectrum required interpolation.')
        print('Using spline interpolation for measurement data and wavelength\n step length of bandpass measurement.')
        Shat_interp = RichLucy(b,deltab,M,**kwargs)
        print "RichLucy finished. Interpolating estimate back to original wavelength scale."
        spline_Shat = interp1d(lm,Shat_interp,bounds_error=False,kind='linear')
        Shat = spline_Shat(lambda_spectrum)            
    else:                
        Shat = RichLucy(b,deltab,M,**kwargs)
        
    Shat[np.nonzero(np.isnan(Shat))]=0.0      
        
    return Shat












######################## demonstration of RichLucy - use from Python shell ###############

## generate simulated bandpass
def sim_bandpass(bandwidth,sampledist,skew=1,noise=None):
        """
        Generate a triangular bandpass function with given bandwidth
        and skewnes (skew) on the wavelength axis [-bandwidth,bandwidth]
        
        Relative measurement noise can be simulated using noise and noise_low
        as maximum and minimum relative noise levels.
        
        :param bandwidth: float - the bandwidth of the bandpass function
        :param sampledist: float - the wavelength step size
        :param skew: float - skewness of the triangle (left-skewed for <1.0, right-skewed for >1.0)
        :param noise: float - additive noise
        
        :returns: hat, lamb - the bandpass function and the corresponding wavelengths
                
        """
        
        # definition of x-axis
        lamb = np.arange(-bandwidth,bandwidth+sampledist,sampledist) + bandwidth
        ind_kl = np.nonzero(lamb < skew*bandwidth)  # indices for increasing slope
        ind_gr = np.nonzero(lamb >= skew*bandwidth) # indices for decreasing slope
    
        # definition of bandpass function    
        hat = np.zeros_like(lamb)
        hat[ind_kl] = lamb[ind_kl]/(skew*bandwidth**2)
        hat[ind_gr] = -lamb[ind_gr]/( (2.0-skew)*bandwidth**2 ) + 2.0/( (2-skew)*bandwidth )

        # definition of noise/measurement uncertainty
        if isinstance(noise,float): # interpret noise as i.i.d. normal
            noise_vals = np.random.randn(len(lamb))*noise
            hat += noise_vals
        
        lamb = lamb - bandwidth        
        hat[np.nonzero(hat<0)] = 0.0
        hat = hat/np.trapz(hat,dx=sampledist)
        
        return hat,lamb




## generate simulated measurement
def sim_measurement(Sfun,lamb,b,bandwidth,delta_b,noise=None):
        """
        Generate a Gaussian-shaped spectrum by simulating a measurement of a
        clean Gaussian-shaped function Sfun defined on the wavelength axis lamb
        with a triangular bandpass function b of given bandwidth and wavelength
        step size delta_b.
        
        Note that Sfun needs to be a function object.
               
        Measurement noise can be simulated using the relative noise levels
        noise_low and noise_up
        
        :param Sfun: a function S(lambda) returning the spectrum at wavelengths lambda
        :param lamb: ndarray of shape (N,) with wavelengths for measurement
        :param b: ndarray of shape (L,) - the bandpass function values
        :param bandwidth: float - the bandwidth of the bandpass function
        :param delta_b: float - the wavelength step size of the bandpass function
        :param noise_low: float - lower limit for the relative measurement uncertainty
        :param noise_up: float - upper limit for the relative measurement uncertainty
        
        :returns: M - ndarray of shape (N,) with simulated measured spectrum values
        """
        
        if hasattr(Sfun,'__call__'):        
           # simulation of the measurement
            M = np.zeros_like(lamb)
            for ii in range(len(lamb)):
                lambda_int = lamb[ii] + np.arange(-bandwidth,bandwidth+delta_b,delta_b)                
                M[ii] = np.trapz(Sfun(lambda_int)*b,dx=delta_b)
            # add measurement noise
            if isinstance(noise,float):
                M += np.random.randn(len(M))*noise
                M[M<0] = 0.0
    
            return M
        else:
            raise ValueError('The first input variable -Sfun- needs to be of type <function>.')




############################## demo of RL with simulated data ################

def demo(bandwidth=40,skew=1,FWHM=10,maxiter=2000): 
    """
    Demonstrate usage of the Richardson-Lucy method for bandpass correction.
    A triangular bandpass of given bandwidth and skewness is used to simulate
    a measurement of a Gaussian-shaped spectrum of given FWHM value.
    
    The maximum number of Richardson-Lucy iterations is given by maxiter.
    
    Two different automatic stopping criteria are employed and the results
    are compared.    
    """
    from matplotlib.pyplot import figure,plot,clf,xlabel,ylabel,legend,show
    from stopping_rules import discrepancy
        
    noiseM = 1e-3
    noiseb = 1e-4
    
    delta_b = .1      # sampledist for monochromator (in nm)
    delta_M = delta_b # sampledist for measurement points (in nm)
    mu    = 500      # mean of Gaussian pulse (simulated input spectrum)
    sigma  = np.sqrt(FWHM**2/(6*np.log(2)))
    
    lambda_min = 400    # min wavelength in spectrum measurement
    lambda_max = 600    # max wavelength in spectrum measurement
    lambda_Mess = np.arange(lambda_min+bandwidth,lambda_max-bandwidth,delta_M)
    
    Sfun = lambda lam: np.exp(-(lam-mu)**2/(2*sigma**2))
    S = Sfun(lambda_Mess);

  # generation of bandpass function
    b,lambda_b = sim_bandpass(bandwidth,delta_b,skew,noiseb)
    figure (1);clf()
    plot(lambda_b,b)    
    
  # simulation of the measurement          
    M = sim_measurement(Sfun,lambda_Mess,b,bandwidth,delta_b,noiseM)

### plot intermediate results
    figure(2);clf()
    plot(lambda_Mess,S,lambda_Mess,M)
    
### application of Richardson-Lucy
    stoprule = discrepancy(M,b[::-1]*delta_b,noiseM)
    Shat = RichLucy(b,delta_b,M,maxiter=maxiter,stoprule=stoprule)

### plot results         
    figure(3);clf()
    plot(lambda_Mess,S,lambda_Mess,M,lambda_Mess,Shat)
    xlabel('wavelength in nm')
    ylabel('spectrum in a.u.')
    legend(('input spectrum','measured spectrum', \
            'Rich-Lucy estimate'),loc='best')

    stoprule.plot(fig_nr=4)
    
    show()
    
    return b,lambda_b,M,Shat,lambda_Mess
    
    


#########################################################################
    
    
    
############################# main function for command line use of RL    

def __main():
    from scipy.interpolate import interp1d
    
    print '\n\n-------------------------------------------------------'
    print 'Richardson-Lucy algorithm for bandpass correction command line tool'    
    print 'S. Eichstaedt, F Schmaehling (PTB) 2013'
    print 'Usage of this software is without any warranty.'
    print '\n-------------------------------------------------------'

    if len(sys.argv)<4:
        print 'Not enough input arguments! \nNeed at least [bandpass file] [measured spectrum file] and [output file]'
        print 'Additional optional arguments are [maxiter] and [Monte Carlo runs]'
        return
    else:
        bandpassfile = unicode(sys.argv[1])
        measfile = unicode(sys.argv[2])
        outputfile = unicode(sys.argv[3])
        if len(sys.argv)>4:
            maxiter = int(sys.argv[4])
            if len(sys.argv)>5:
                calcUnc = 1
                MCruns = int(sys.argv[5])
            else:
                calcUnc = 0
        else: # default values
            maxiter = 500
            calcUnc = 0
    
        print 'RichLucy deconvolution parameters:'
        print 'input files:\n ' + bandpassfile + '\n ' + measfile
        print 'output file:\n ' + outputfile
        print 'maximum number of iterations: %d' % maxiter
        if calcUnc:
            print 'Uncertainties will be calculated using %d Monte Carlo trials' % MCruns
        print('-------------------------------------------------------\n')
    
    
        print "---- Loading bandpass file:"
        bdata = LoadFromFile(bandpassfile)
                    
        lb= bdata[:,0]
        b = bdata[:,1]        
        
        if calcUnc:
            m,n = bdata.shape
            if n<3:
                print "Bandpass data file does not contain uncertainties. Assigning zeros instead."
                Ub = np.zeros_like(b)
            else:
                Ub = np.abs(bdata[:,2])

        lb,b,interpolated = Interpolate2Equidist(lb,b)[:3]
        deltab = lb[1]-lb[0]        

        if interpolated:                
            bout_interp = bandpassfile[:-4] + "_interp" + bandpassfile[-4:]
            print "Saving interpolated bandpass to " + bout_interp
            datei = open(bout_interp,'w')
            header = "# Interpolated bandpass function\n" + "# wavelengths | interpolated bandpass \n"
            datei.write(header)
            Data = np.vstack((lb,b,))
            Data = Data.transpose()
            np.savetxt(datei,Data)
            if calcUnc:
                print "Note that for uncertainty evaluation the original data will be used."
        
        intb = np.trapz(b,dx=deltab)
        if np.abs(1-intb)>1e-4:
            print 'Bandpass function is not normalized...doing it now.'
            b = b/intb
            
        centroid = np.trapz(b*lb,dx=deltab)
        print 'central wavelength of bandpass function is %1.4f' % (centroid)
        
        lambda_central = np.mean(lb)
        print 'central wavelength of provided wavelength scale is %1.4f' % (lambda_central)
        
        shift = centroid - lambda_central
        if shift>1e-3:
            print 'Suggested shift of wavelength scale is %1.4f'% (shift)


        print "\n---- Loading measured spectrum file:"
    
        mdata = LoadFromFile(measfile)    
    
        lm= mdata[:,0]
        M = mdata[:,1]
        
        if calcUnc:
            m,n = mdata.shape
            if n<3:
                print "Spectrum data file does not contain uncertainties. Assigning zeros instead."
                UM = np.zeros_like(M)
            else:
                UM = np.abs(mdata[:,2])
        
        lm,M,interpolated_M = Interpolate2SameStep(M,lm,deltab)[:3]
        deltam = lm[1]-lm[0]
        
        if interpolated_M:
            Mout_interp = measfile[:-4] + "_interp" + measfile[-4:]
            print "Saving interpolated spectrum to " + Mout_interp
            datei = open(Mout_interp,'w')
            header = "# Interpolated measured spectrum\n" + "# wavelengths | interpolated bandpass \n"
            datei.write(header)
            Data = np.vstack((lm,M,))
            Data = Data.transpose()
            np.savetxt(datei,Data)
            if calcUnc:
                print "Note that for the uncertainty evaluation the original data will be used."
        
        
        print 'Input data loaded successfully'
        print 'bandpass function:'
        print '   wavelength scale: [%1.3f,%1.3f]' % (lb[0],lb[-1])
        print '   wavelength step: %1.4f' % (deltab)
        print 'measured spectrum:'
        print '   wavelength scale: [%1.3f,%1.3f]' % (lm[0],lm[-1])
        print '   wavelength step: %1.4f' % (deltam)
        
        if calcUnc:
            m1,n1 = bdata.shape
            m2,n2 = mdata.shape
            if n1<3 and n2<3:
                print "No uncertainties assigned to the input data. Skipping uncertainty evaluation."
                calcUnc = False
                
        if calcUnc:
            Shat,UShat = RichLucyUncertainty(bdata[:,1],Ub,bdata[:,0],\
                        mdata[:,1],UM,mdata[:,0],\
                        runs=MCruns,maxiter=maxiter)
        else:
            if interpolated_M:
                print('Measured spectrum required interpolation.')
                print('Using spline interpolation for measurement data and wavelength\n step length of bandpass measurement.')
                interp_outputfile = outputfile[:-4] + '_interp' + outputfile[-4:]                
                Shat_interp = RichLucy(b,deltab,M,maxiter)[0]
                print "RichLucy finished. Interpolating estimate back to original wavelength scale."
                spline_Shat = interp1d(lm,Shat_interp,bounds_error=False,kind='linear')
                Shat = spline_Shat(mdata[:,0])            
            else:                
                Shat = RichLucy(b,deltab,M,maxiter)[0]
    
        header1 = "# Result of Richardson-Lucy deconvolution calculated using the RichLucy command line version " + repr(version) + "\n"
        if interpolated_M:
            header2 = "# Estimation did require interpolation. deltab = " \
            + repr(deltab) + ", deltaM = " + repr(deltam) + "\n"\
            "# Interpolation results are stored in " + interp_outputfile + "\n"            
        else:
            header2 = ""
        header3 = "# bandpass function centroid was at " + repr(centroid) \
        + " whereas central wavelength value was " + repr(lambda_central) + "\n"\
        + "# Suggested shift of wavelength scale is " + repr(shift) + "\n"
        if calcUnc:
            header4 = "# wavelengths | estimated spectrum | uncertainty (covariance matrix)\n"
        else:
            header4 = "# wavelengths | estimated spectrum\n"
        header = header1 + header2 + header3 + header4
        
        print "Saving results to file..."
        datei = open(outputfile,'w')        
        datei.write(header)
        if calcUnc:
            Results = np.vstack((mdata[:,0],Shat,UShat))
        else:
            Results = np.vstack((mdata[:,0],Shat))
        Results = Results.transpose()
        np.savetxt(datei,Results)   
            
        print 'Saved result to ' + outputfile
        print 'Format is: ' + header4

        if interpolated_M:        
            datei = open(interp_outputfile,'w')
            header = "# Interpolation result" + header1[8:] + header4
            datei.write(header)
            Data = np.vstack((lm,Shat_interp,))
            Data = Data.transpose()
            np.savetxt(datei,Data)
            print 'Saved interpolated result to ' + interp_outputfile
              
            
        print 'All done'
            




########################################################################
    
    
####################################################################

if __name__ == '__main__':
        __main()