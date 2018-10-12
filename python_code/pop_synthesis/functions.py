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
     
    df= pd.read_csv(f, delim_whitespace=True,comment='#', names=col_names, usecols=wanted_cols)    
    
    if "-0" in str(f): # for non-rotating star, add 0 vsurf and vcrit columns 
        df['8:v_crit[km/s]'] =0       
        df['9:v_surf[km/s]'] =0 
    
    return df, model_name
