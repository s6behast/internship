import glob
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy import interpolate ,optimize, integrate
from scipy.interpolate import UnivariateSpline

def MVT_interp_setup(data_df, mass_list):
    df=data_df[data_df['1:t[s]']== 0]

    v_T_interp={}
    for m in mass_list: 
        x=np.array(df['V_init'][df.M==m])#[-20:]
        y=np.array(df['M'][df.M==m])#[-20:]
        z=np.array(df['T_MS'][df.M==m])#[-20:]

        func=interpolate.interp1d(x,z,bounds_error=False, assume_sorted=True)
        v_T_interp[m]=func
    return v_T_interp, mass_list

def M_V_T_interp(M, V, v_T_interp_obj):
    v_T_interp=v_T_interp_obj[0]
    mass_list= v_T_interp_obj[1]
    x= mass_list
    y=map((lambda u: v_T_interp[u](V)),x)#v_T_interp[m](v)

    func2=interpolate.interp1d(x,y,bounds_error=False, assume_sorted=True)

    return func2(M)
    
    
def v_over_vcrit_interp(data_df, m_i, v_init, v_T_interp_obj, plot):
    '''takes a m_i on the grid and a velocity v_init not on the grid and produces by 
    interpolation the v/vcrit against t or t/T curve'''
    v_T_interp=v_T_interp_obj[0]
    mass_list= v_T_interp_obj[1]
    plot_df=pd.DataFrame()
    dm=data_df[data_df.M ==m_i]
   

    for t_frac in np.linspace(0,0.99, 199):
        dg=pd.DataFrame()
        for vi in np.unique(dm['V_init'])[1:]:
            di= dm[(dm['V_init']==vi)  ]
            ti=((di['t/Tms']-(t_frac)).abs()).argmin()
            dn= di.iloc[ti]
            dg=dg.append(dn)

            
                #straight_line1=interpolate.interp1d( dg['V/Vcrit'],dg['V_init'],
                #                                       bounds_error=False, assume_sorted=True)
        straight_line1=interpolate.interp1d(dg['V_init'], dg['V/Vcrit'],
                                                       bounds_error=True, assume_sorted=True)

        v_over_vcrit=straight_line1(v_init)
        
        if t_frac == 0 :
			Vvcrit_init = v_over_vcrit

        plot_df=plot_df.append(pd.DataFrame([[m_i,v_init,t_frac,v_over_vcrit, Vvcrit_init]], 
                                               columns=['M','V_init', 't/T', 'V/Vcrit', 'V/Vcrit_init']))

        
    plot_df.reset_index(drop=True, inplace=True)
    
    
    #interpolate MS lifetime of star 
    T= M_V_T_interp(m_i, v_init, v_T_interp_obj)
    
    # insert real time into dataframe 
    plot_df['t']= plot_df['t/T']* T
    
    #plot=False
    if plot== True:
        plt.plot( plot_df['t/T'],plot_df['V/Vcrit'], 'r-')
        plt.title(str(v_init)+'   '+str(m_i))
        plt.xlabel('t/T')
        plt.ylabel('V/Vcrit')
        plt.show()

    return plot_df

def interp_vcrit_min(m_i, B_i, data_df,v_critmin_boundary, vmax_dict ):
#m_i=5.6
#B_i=0.7
    plot=False # if plot is true, plots of the interpolated curve and grid models are made 
    m1= data_df[data_df['M'] <=  m_i].M.max()
    m2= data_df[data_df['M'] >  m_i].M.min()

    interp_lines={}
    plt.clf()
    for M in [m1,m2]:#np.sort(v_crit_min_master[B].keys()):
        
        for B in [B_i]:#np.sort(v_crit_min_master.keys()):
            #print v_critmin_boundary.keys(), B,M
            if (B,M) in v_critmin_boundary.keys():
                critmin_df=v_critmin_boundary[B,M]

            else :
                continue
            #crit_rots_df= crit_rot_data[M]

                #x=critmin_df['t/T']
                #y=critmin_df['Vcrit_min']

            x=critmin_df['t/T']#[(critmin_df['t/T'] < critmin_df['t/T'].max()) & 
                               #     (critmin_df['t/T'] > critmin_df['t/T'].min())]
            y=critmin_df['Vcrit_min']#[(critmin_df['t/T'] < critmin_df['t/T'].max()) & 
                                    #(critmin_df['t/T'] > critmin_df['t/T'].min())]

            straight_line=interpolate.interp1d(x,y,bounds_error=True, assume_sorted=True)

            interp_lines[M]= [straight_line, [x.min(),x.max()]]
            #print x.min(),x.max()
            #plt.plot(np.linspace(x.min(),x.max(),100), straight_line(np.linspace(x.min(),x.max(),100)))

            if plot:
                plt.plot(critmin_df['t'],critmin_df['Vcrit_min'], '.',label=M )
            

    a=interp_lines[m1]
    b=interp_lines[m2]
    mins=[a[1][0],b[1][0] ]
    maxs=[a[1][1],b[1][1] ]
    
    t_vals=np.linspace(max(mins), min(maxs), 1e3)

    a=interp_lines[m1][0]
    b=interp_lines[m2][0]
    res_df=pd.DataFrame()
    for t_i in t_vals:
        y=[a(t_i), b(t_i)]

        straight_line=interpolate.interp1d([m1,m2],y,bounds_error=True, assume_sorted=True)

        T= M_V_T_interp(m_i,straight_line(m_i), v_T_interp)

        res_df=res_df.append(pd.DataFrame([[t_i*T,straight_line(m_i), t_i]], columns=['t', 'V_init', 't/T']))


    res_df.reset_index(drop=True, inplace=True)
    # insert values with t/T =1 on leading edge of plot 
    
    res_df=res_df[:-1]

    v_vals=np.linspace(res_df['V_init'].iloc[res_df['t'].argmax()], vmax_dict[m1], 100)
    for v_i in v_vals: 
        T=M_V_T_interp(m_i,v_i, v_T_interp)

        res_df=res_df.append(pd.DataFrame([[0.99*T,v_i, 0.99]], columns=['t', 'V_init', 't/T']))

    res_df.reset_index(drop=True, inplace=True)
    if plot:
        plt.plot(res_df['t'],res_df['V_init'], 'r.' , label=m_i)
        plt.legend(loc='lower left')
        #plt.plot(x,y, '.',label=M )
        plt.title(str(B) + str('$V/V_{crit}$'))

        plt.xlabel('t/Myr')
        plt.ylabel('V/$kms^{-1}$')
        
        plt.show()

    return res_df
