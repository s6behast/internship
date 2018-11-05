import glob
import numpy as np
import pandas as pd
import sys

m_val=lambda x : int(x.split('/')[-1].split('.')[0].split('-')[0][1:])
v_val= lambda x : int(x.split('/')[-1].split('.')[0].split('-')[1])

def load_masses(data_dir, max_mass):
    # sort file names so that they are listed by mass, then rotation velocity 
    

    file_ext='.dat2'
    s1=sorted(glob.glob(data_dir +'*'+ file_ext), key=v_val)
    data_files=sorted(s1, key= m_val)


    #cut stars heavier than 40M, at they do not contribute to Be
    data_files=filter(lambda x: m_val(x) <= max_mass, data_files)
    #print data_files

    mass_list=[]
    for f in data_files:
        mass_list=np.append(mass_list, int(m_val(f)))

    mass_list=np.unique(mass_list)


    print 'masses of models :\n',mass_list
    
    return data_files, mass_list


def load_models(data_files):
	data_df=pd.DataFrame()
	i=1
	for fil in data_files:
	    message="working on file " +str(i) + " / "+ str(len(data_files)) #+ " :time loop "+str(i2) + " / "+ str(n_vals)
	    sys.stdout.write ('\r'+ message)
	    i=i+1
	    df_ms, T_ms= MS_finder2(fil)
	    

		# index of time closest to t 
		#T_indx=(df_ms['1:t[s]']-t).abs().argmin()
		#t_i=df_ms['1:t[s]'].loc[T_indx]
		#V= df_ms['9:v_surf[km/s]'].loc[T_indx] 
		#if star is rotating 
	    if v_val(fil) != 0:
		V_over_Vcrit=df_ms['9:v_surf[km/s]']/ df_ms['8:v_crit[km/s]']
		#print V_over_Vcrit
		V_initial=df_ms['9:v_surf[km/s]'][0:10].mean()
		V_crit_initial=df_ms['8:v_crit[km/s]'][0:10].mean()
		crit_frac= V_initial/V_crit_initial
		V_crit=df_ms['8:v_crit[km/s]']
	    else:
		V_over_Vcrit=0
		V_initial=0
		V=0
		crit_frac=0
		V_crit=0
	    ti_over_tms=df_ms['1:t[s]']/T_ms

	    #df_ms=df_ms.append(pd.DataFrame([[f_name,m,V_initial,V, T_ms,V_over_Vcrit, ti_over_tms]],
		                                              #columns=['f-name','M','V_initial', 'V', 'T_MS', 'V/Vcrit', 't_i/T_MS']))
	    #df_ms['V_crit']=V_crit
	    df_ms['V/Vcrit']=V_over_Vcrit
	    df_ms['t/Tms']=ti_over_tms
	    df_ms['V_init']=V_initial
	    df_ms['M']= m_val(fil)
	    df_ms['V']=v_val(fil)
	    df_ms['T_MS']= T_ms
	    df_ms['V/Vc_initial']= crit_frac
	    
	    data_df=data_df.append(df_ms)

	return data_df


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

