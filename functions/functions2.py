import numpy as np
import pandas as pd


def read_dat2file(f):
    """ imports a dat22 file into pandas dataframe. f is full path to file """
    model_name= f.split('/')[-1].rsplit('.',1)[0]
    wanted_cols=['1:t[s]', '2:M/Msun', '3:Teff[K]', '4:log(L/Lsun)','5:R/Rsun','6:log(Mdot)[Msun/yr]',
                 '8:v_crit[km/s]', '9:v_surf[km/s]', '11:H','12:He','26:H_massfr', '27:He_massfr']

    if "-0" in str(f): # if file is for NON rotating star, remove vrot and crit vel from col names 
            col_names= ['1:t[s]', '2:M/Msun', '3:Teff[K]', '4:log(L/Lsun)', '5:R/Rsun', '6:log(Mdot)[Msun/yr]', '7:logg[cgs]', 
                        '10:P[days]', '11:H', '12:He', '13:Li', '14:Be', '15:B', '16:C', '17:N', '18:O', 
                    '19:F', '20:Ne', '21:Na', '22:Mg', '23:Al', '24:Si', '25:Fe', '26:H_massfr', '27:He_massfr']
    else: 
        col_names= ['1:t[s]', '2:M/Msun', '3:Teff[K]', '4:log(L/Lsun)', '5:R/Rsun', '6:log(Mdot)[Msun/yr]', '7:logg[cgs]', '8:v_crit[km/s]', 
                '9:v_surf[km/s]', '10:P[days]', '11:H', '12:He', '13:Li', '14:Be', '15:B', '16:C', '17:N', '18:O', 
                '19:F', '20:Ne', '21:Na', '22:Mg', '23:Al', '24:Si', '25:Fe', '26:H_massfr', '27:He_massfr']
     
    df= pd.read_csv(f, delim_whitespace=True,comment='#', names=col_names)#, usecols=wanted_cols)    
    
    if "-0" in str(f): # for non-rotating star, add 0 vsurf and vcrit columns 
        df['8:v_crit[km/s]'] =0       
        df['9:v_surf[km/s]'] =0 
    
    return df[wanted_cols], model_name


def MS_finder2(f):
    #find terminal age MS time with criteria X_H > 1e-3
    df, nam= read_dat2file(f)
    
    #FIND MS MODELS. based on when the star's radius first starts to shrink.
    ##########################################################################
    #if star is non-rotating, find first point when radius contracts 
    if df['9:v_surf[km/s]'].iloc[0] ==0:
        y=np.array(df['5:R/Rsun'])
        t=np.array(df['1:t[s]'])
        dydt= np.gradient(y, t)
        u=lambda x : x< -1e-6

        #if model leaves ms...
        if np.unique(u(dydt), return_counts=True)[1][0] !=len(dydt):
                index=np.nonzero(dydt ==(filter(u, dydt)[0]))[0].min()
                t_endMS= t[index]
                df=df[df['1:t[s]'] < float(t_endMS)] 
                #plt.plot(df['1:t[s]'], df['9:v_surf[km/s]']/df['8:v_crit[km/s]'])
                #plt.title(nam)
                #plt.show()
                return df, t_endMS
            
        else: 
                t_endMS= df['1:t[s]'].max()
                df=df[df['1:t[s]'] < float(t_endMS)] 
                #plt.plot(df['1:t[s]'], df['9:v_surf[km/s]']/df['8:v_crit[km/s]'])
                #plt.title(nam)
                #plt.show()
                return df, t_endMS
   
    #if star is rotating, find point when v/vcrit drops 
    else:
            
    
        u=lambda x : x < 0.05

        x=(df['9:v_surf[km/s]']/df['8:v_crit[km/s]'])
        #if model leaves ms...
        if (x < 0.05).any()==True:

            #print filter(u,x)
            index=np.nonzero(x ==(filter(u,x))[0])[0].min()
            t_endMS= df['1:t[s]'].iloc[index]
            df=df[df['1:t[s]'] < float(t_endMS)] 

        else:
            t_endMS= df['1:t[s]'].max()
        
    #df=df[((df['9:v_surf[km/s]']/df['8:v_crit[km/s]']) >0.05) & (df['9:v_surf[km/s]']/df['8:v_crit[km/s]'] !=0 )]
    #t_endMS= df['1:t[s]'].max()
                     
    #plt.plot(df['1:t[s]'], df['9:v_surf[km/s]']/df['8:v_crit[km/s]'], 'k-')
    #plt.title(nam)
    #plt.show()
    
    #df_cen, _= read_cen1file(f)
    #TAMS_t= (df_cen['2:t[s]'][df_cen['3:H1'] > 5e-2].max())
    #print TAMS_t
    
    #plt.plot(df_cen['2:t[s]'],df_cen['3:H1'])
    #plt.axvline(t_endMS)
    #plt.show()
        
    return df, t_endMS





def M_MAX_calc2(data_df, t_val, v_max_value):
    dm=pd.DataFrame()
    d22t=pd.DataFrame()
    for mi in np.unique(data_df['M']):
        d=data_df[data_df['M']==mi]
        for vi in np.unique(d['V']):
            di= d[d['V']==vi]
            #ti=(di['t/Tms']-(1)).abs().argmin()
            #print ti, vi
            #print  di[di['t/Tms'] == 0.0]
            dn= di.iloc[-1]#di[di['t/Tms'] == ti]
            dm=dm.append(dn)
    #if ms lifetime of largest mass in dataframe rotating at highest velocity is less than time, give the largest mass
    
    #print f(v_max_value, dm['M'].max())
    if f(v_max_value, dm['M'].max()) >=t_val:
        return dm['M'].max()

    df=dm

    grid_x, grid_y = np.mgrid[0:t_val:1000j, 0: v_max_value:1000j]
    values = np.array(df['M'].astype(float))
    points=np.array(df[['1:t[s]','V_init' ]])


    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')

    #print grid_y[0][950]
    #print grid_x[900][0]

    #look for values in range t/Tms >0.95 and v > 0.5 v_max 
    M_min_candidates=[]
    for x in range(995,1000-1):
        for y in range(995,1000-1):
            M_min_candidates=np.append(M_min_candidates,grid_z0[x][y])


    return M_min_candidates.max()


def integrate_vdist(lower, upper, v_dist, v_max_val):
    """"integrates a polynomial object v_dist between lower and upper 
    v_max_val is largest v value for v_dist """
    # vdistribution as found earlier in polynomial form
    #v_dist=np.poly1d([ 4.96085349e-13, -4.27023037e-10,  6.65435944e-08,  6.46442910e-06
    #,1.93953157e-03])
    # max value of v that polynomial takes. for values greater than v, pdf is 0 
    #v_max_val=430
    
    if lower  > v_max_val:
        return 0
    
    if upper > v_max_val:
        upper= v_max_val
        
    int_func=np.polyint(v_dist)
     
    return int_func(upper) -int_func(lower)



def integrate_imf(lower, upper, alpha, eta0):
    """integrates initial mass function from power law imf with power -alpha, constant eta0.
    m is input masses. integral of imf"""
    a= -alpha +1
    
    if lower == 0 :
        
        return -(eta0 /a * upper**a)
    
    
    else:
        int_upper=  (eta0/ a) * upper**a
        int_lower= (eta0/ a) *lower**a
        integ_result= int_upper - int_lower
    
        return integ_result
    