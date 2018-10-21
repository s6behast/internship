import pandas as pd 
import numpy as np
from scipy.interpolate import griddata
from skmonaco import mcimport

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
    
    return df, model_name


def read_cen1file(f):
    """ imports a cen1 file into pandas dataframe. f is full path to file.
    also converts from abundances to mass fractions a--> f by f=10**-a"""
    
    
    #some files have scientific notation with D instead of E. Eg 1E32 is written as 1D32
    #code below fixes this with converters in read_csv
    def D_to_E(s):
        if s.isdigit():
            return s #np.float(s.replace('D', 'E'))

        else:
            return s.replace('D', 'E')
    
    
    model_name= f.split('/')[-1].rsplit('.',1)[0]
    f= str(f.split('.')[0]) +'.'+  str(f.split('.')[1]) + '.cen1'
    #print f
    col_names= ['1:?','2:t[s]',  '3:H1', '4:H2', '5:He3', '6:He4', '7:Li6', '8:Li7', 
                '9:Be7', '10:Be9', '11:B8', '12:B10', '13:B11', '14:C11', '15:C12', '16:C12', '17:N12', '18:N14', 
                '19:N15', '20:O16', '21:O17', '22:O18', '23:Ne20', '24:Ne21', '25:Ne22', '26:Na23', '27:Mg24',
                '28:Mg25', '29:Mg26', '30:Al27', '31:Si28', '32:Si29', '33:Si30', '34: Fe56', '35:F19', '36:Al26']
     
    df= pd.read_csv(f, delim_whitespace=True,comment='#', names=col_names, 
                    converters={1:D_to_E,2:D_to_E,3:D_to_E,5:D_to_E})#, usecols=wanted_cols)    
    df = df.apply(pd.to_numeric, errors='coerce', axis=1) # convert all values to floats
    times=df['2:t[s]']
    df=df.drop(['2:t[s]','1:?' ], axis=1)
    #df=np.power(10, df)
    
    #do 10**-x of all values to get mass frac
    df=-df
    df=df.rpow(10)
    df['2:t[s]']= times #add times again 
    return df, model_name




def MS_finder(f):
    #find terminal age MS time with criteria X_H > 1e-3
    df_cen, _= read_cen1file(f)
    TAMS_t= (df_cen['2:t[s]'][df_cen['3:H1'] > 5e-2].max())
    #print TAMS_t
    
    #plt.plot(df_cen['2:t[s]'],df_cen['3:H1'])
    #plt.show()
    
    #load data and filter out only rows younger than TAMS
    df, _= read_dat2file(f)
    
    df=df[df['1:t[s]'] < TAMS_t]
    
    return df, TAMS_t


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
    
    
def MS_finder2(f):
    #find terminal age MS time with criteria X_H > 1e-3
    df, nam= read_dat2file(f)
    
    #FIND MS MODELS. based on when the star's radius first starts to shrink.
    ##########################################################################
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
        return df, None

def random_M_value(size):
    
    if type(size) == tuple:
        size=size[0]
    
    df=resdf1 
    ##################################
    # PERHAPS WRONG TO FILTER OUT ONLY MS STARS????
    ######################
    M_vals=np.unique(df['M'])[:-1]#[df['t_i/T_MS']]  <1.0])
    eta0=(-alpha+1) * (M_vals.max()**(-alpha+1)- M_vals.min()**(-alpha+1))**-1
    #add first mass manually 
    mass_bins=pd.DataFrame([[M_vals[0], M_vals[0],( -M_vals[0]+ M_vals[1])*0.5 + M_vals[0] ]],
                           columns=['M', 'lower_bin', 'upper_bin'])
    
    for i in range(1,len(M_vals)-1):
        l=( -M_vals[i]+ M_vals[i-1])*0.5 + M_vals[i]
        u= ( -M_vals[i]+ M_vals[i+1])*0.5 + M_vals[i]
        #print M_vals[i],'lower',( -M_vals[i]+ M_vals[i-1])*0.5 + M_vals[i],
        #print 'upper',( -M_vals[i]+ M_vals[i+1])*0.5 + M_vals[i]
        mass_bins=mass_bins.append(pd.DataFrame([[M_vals[i], l,u]],columns=['M', 'lower_bin', 'upper_bin'] ))
     
    # add last mass manually 
    l=( -M_vals[-2]+ M_vals[-1])*0.5 + M_vals[-2]
    u=  M_vals[-1]
    mass_bins=mass_bins.append(pd.DataFrame([[M_vals[-1], l,u]],columns=['M', 'lower_bin', 'upper_bin'] ))
    
    #do integration on IMF over bins
    p = lambda x : integrate_imf(x.lower_bin, x.upper_bin, alpha, 1 )
    mass_bins['p']= mass_bins.apply(p, axis=1)
    
    p_vals=mass_bins['p']/mass_bins['p'].sum()
    random_m_val= np.random.choice(mass_bins['M'], p=p_vals)# p normalised to sum of probs to avoid sum !=1 error
    
    bin_width= mass_bins['upper_bin'][mass_bins.M == random_m_val]-mass_bins['lower_bin'][mass_bins.M == random_m_val]
    
    rand_vals=[]
    for w in range(0,size):
        rand_vals=np.append(rand_vals, np.random.choice(mass_bins['M'], p=p_vals))
    
    return rand_vals#, dict(zip(M_vals, p_vals)), bin_width.values
    

#print random_M_value(resdf1, alpha, eta0)

def M_max_calc(df, v_max_value):
    
    #if non-rotating maximum mass star is still on MS, retun maximum mass star mass
    if df['t_i/T_MS'][(df.M == df['M'].max()) & (df.V == 0)].values[0] < 1.0 :
        return df['M'].max()
    
    
    # x is t/t_ms, y is V
    grid_x, grid_y = np.mgrid[0.9:1:1000j, 0: v_max_value:1000j]
    values = np.array(df['M'].astype(float))
    points=np.array(df[['t_i/T_MS','V' ]])


    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')

    #print grid_y[0][950]
    #print grid_x[900][0]

    #look for values in range t/Tms >0.95 and v > 0.5 v_max 
    M_min_candidates=[]
    for x in range(500,1000-1):
        for y in range(500,1000-1):
            M_min_candidates=np.append(M_min_candidates,grid_z0[x][y])

    #plot interpolation
    #plt.imshow(grid_z0, extent=(0.9,1,0.9,1), origin='lower')
    #plt.colorbar()
    #plt.show()
    
    M_min= np.nanmax( M_min_candidates)
    return M_min




def V_min_calc(M, df):
    # if M is not a value from the data, force it to be the closest value 
    if M not in df.M.values: 
        M=df['M'][(df['M'] -M).abs().argmin()]
    
    # fractional age of non-rotating model
    non_rot_tovertms= df['t_i/T_MS'][(df.M==M) & (df.V==0)].values
    
    #if all models of mass M are post MS, return an infinite V_min 
    if (df['t_i/T_MS'][(df.M==M) ].values  > 1).all(): 
        return 10000
    
    #if non rotating model is NOT MS
    if non_rot_tovertms > 1.0:
        # find 2 points eitherside of 1.0 and then linearly interpolate to get V_min 
        y=df['t_i/T_MS'][(df.M==M)]
        x=df['V'][(df.M==M)]
        
        #print df[(df.M==M)]
        #y.reset_index(drop=True, inplace=True)
        #print y[(y-1).abs().argmin()] , x[(y-1).abs().argmin()]
        #print y[(-y+1).argmin()] , x[(-y+1).argmin()]
        
        # get indexes of points with t/Tms straddling t/Tms =1 
        p_1_indx=df['t_i/T_MS'][(df.M==M) & (df['t_i/T_MS'] >1)].argmin()
        m_1_index=df['t_i/T_MS'][(df.M==M) & (df['t_i/T_MS'] <1)].argmax()
        
        #print  y[df['t_i/T_MS'][(df.M==M) & (df['t_i/T_MS'] >1)].argmin()]#(y-1)#.abs()#.argmin()
        #print 'XXXXX'
        #print  y[df['t_i/T_MS'][(df.M==M) & (df['t_i/T_MS'] <1)].argmax()]#(y-1)#.abs()#.argmin()
        
        
        #interpolate between two previously found points to find V at t/Tms=1 
        straight_line=np.poly1d(np.polyfit( [y[p_1_indx], y[m_1_index]],
                                            [x[p_1_indx], x[m_1_index]],
                                            1))
        
       
        #plot to check results 
        #plt.axhline( y[p_1_indx])
        #plt.axhline(y[m_1_index])
        #plt.plot(x,y,'ko')
        #plt.plot(straight_line(1.0), 1, 'go')
        #plt.title(M)
        #plt.show()
        
        V_min =straight_line(1.0)
        
        return V_min
    
    #if non-rotating model is on MS, V_min is 0 
    else: 
        return 0 
    
    
def V_min_calc_cont(M,df, v_max_val):

    id1= df[df['M'] <= M].M.max()
    id2= df[df['M'] > M].M.min()

    if V_min_calc(id2, df) == 10000:
        
        m_m=M_max_calc(df, v_max_val)
        x= [V_min_calc(id1, df),v_max_val]
        y=[id1,m_m]
    else:
        return 0 
        #x=[V_min_calc(id1, df),V_min_calc(id2, df)]
        #y=[id1,id2]

        
    #print x, y
    straight_line=np.poly1d(np.polyfit( y,x, 1))
    
    
    #l=np.linspace(0,10,10)
    #plt.plot(straight_line(l), l)
    #plt.plot(x,y, 'ko')
    #plt.show()


    return straight_line(M)
    
    

def V_critmin_calc(M, df, Be_val):
 
    
    #select stars still on the MS, or slightly past it
    df=df[(df['t_i/T_MS'] <1.1) & (df.M==M) ]
    y=df['V/Vcrit']
    x=df['V']
    #plt.plot(x,y,'ko')
    #plt.show()
    
    # if there is only one point, cannot do interpolaton, so return infinite value 
    if len(df) <=1 :
        return 1000000
    

    straight_line=np.poly1d(np.polyfit( y,x,1))
    #plot to check results 
    #plt.axhline( y[p_indx])
    #plt.axhline(y[m_index])
    #plt.plot(x,y,'ko')
    #plt.plot(straight_line(np.linspace(0.5,1,20)),np.linspace(0.5,1,20), 'g-')
    #plt.show()
        
    V_critmin =straight_line(Be_val)
        
    return V_critmin


def V_critmin_calc_cont(M, df, Be_val):
    id1= df[df['M'] <= M].M.max()
    id2= df[df['M'] > M].M.min()
    
    x=[id1,id2]
    y=[V_critmin_calc(id1,df, Be_val),V_critmin_calc(id2,df, Be_val) ]
    straight_line=np.poly1d(np.polyfit(x,y,1))
    
    
    #l=np.linspace(0,10,10)
    #plt.plot(l,straight_line(l))
    #plt.plot(x,y, 'ko')
    #plt.show()
    return straight_line(M)
    
def get_data(time, m_vals, data_files):
    t=time
    resdf1=pd.DataFrame(columns=['f-name','M','t','t_i', 'V', 'T_MS', 'V/Vcrit', 't_i/T_MS'])
    
    i=0
    #i2=i2+1
    m_val=lambda x : int(x.split('/')[-1].split('.')[0].split('-')[0][1:])
    v_val= lambda x : int(x.split('/')[-1].split('.')[0].split('-')[1])
    for m in m_vals:
        m_group_files = filter(lambda x : m_val(x)==m, data_files)
        
        for fil in m_group_files:
            f_name=str(m_val(fil)) + '-'+ str(v_val(fil))
            #print message
            #message="working on file " +str(i) + " / "+ str(len(data_files)) + " :time loop "+str(i2) + " / "+ str(n_vals)
            #sys.stdout.write ('\r'+ message)
            i=i+1
            
            #load in data and find points on the MS
            df_ms, T_ms= MS_finder2(fil)
            
            
            # index of time closest to t 
            T_indx=(df_ms['1:t[s]']-t).abs().argmin()
            t_i=df_ms['1:t[s]'].loc[T_indx]
            V= df_ms['9:v_surf[km/s]'].loc[T_indx] 
            #if star is rotating 
            if v_val(fil) != 0:
                V_over_Vcrit=V/ df_ms['8:v_crit[km/s]'].loc[T_indx]
                
            else:
                V_over_Vcrit=0
                
            ti_over_tms=t/T_ms
                
            resdf1=resdf1.append(pd.DataFrame([[f_name,m,t,t_i ,V, T_ms,V_over_Vcrit, ti_over_tms]],
                                                  columns=['f-name','M','t','t_i', 'V', 'T_MS', 'V/Vcrit', 't_i/T_MS']))
    
    resdf1.reset_index(drop=True, inplace=True)
    return resdf1    
    
    
    

def sample_distribution_shape(df, alpha, eta0):
    ##################################
    # PERHAPS WRONG TO FILTER OUT ONLY MS STARS????
    ######################
    M_vals=np.unique(df['M'][df['t_i/T_MS'] <1.0])
    
    #add first mass manually 
    mass_bins=pd.DataFrame([[M_vals[0], M_vals[0],( -M_vals[0]+ M_vals[1])*0.5 + M_vals[0] ]],
                           columns=['M', 'lower_bin', 'upper_bin'])
    
    for i in range(1,len(M_vals)-1):
        l=( -M_vals[i]+ M_vals[i-1])*0.5 + M_vals[i]
        u= ( -M_vals[i]+ M_vals[i+1])*0.5 + M_vals[i]
        #print M_vals[i],'lower',( -M_vals[i]+ M_vals[i-1])*0.5 + M_vals[i],
        #print 'upper',( -M_vals[i]+ M_vals[i+1])*0.5 + M_vals[i]
        mass_bins=mass_bins.append(pd.DataFrame([[M_vals[i], l,u]],columns=['M', 'lower_bin', 'upper_bin'] ))
     
    # add last mass manually 
    l=( -M_vals[-2]+ M_vals[-1])*0.5 + M_vals[-2]
    u=  M_vals[-1]
    mass_bins=mass_bins.append(pd.DataFrame([[M_vals[-1], l,u]],columns=['M', 'lower_bin', 'upper_bin'] ))
    
    #do integration on IMF over bins
    p = lambda x : integrate_imf(x.lower_bin, x.upper_bin, alpha, eta0 )
    mass_bins['p']= mass_bins.apply(p, axis=1)
    mass_bins.reset_index(drop=True, inplace=True)
    #print mass_bins
    #print mass_bins['p'].sum()
    p_vals=mass_bins['p']/mass_bins['p'].sum()
    #random_m_val= np.random.choice(mass_bins['M'], p=p_vals)# p normalised to sum of probs to avoid sum !=1 error
    
    #bin_width= mass_bins['upper_bin'][mass_bins.M == random_m_val]-mass_bins['lower_bin'][mass_bins.M == random_m_val]
    
    return dict(zip(M_vals, p_vals)), mass_bins
    #return random_m_val, dict(zip(M_vals, p_vals)), bin_width.values
    
#print sample_distribution_shape(resdf1, alpha, eta0)


def sample_distribution(x):
    if x < M_min or x > M_max:
        return 0  
    p_dict=s[0] 
    
    lower_bins= s[1].lower_bin
    upper_bins= s[1].upper_bin
    
    #print (x-lower_bins).abs().argmin()
    #print (upper_bins -x).abs()
    
    l=lower_bins[(x-lower_bins).abs().argmin()]
    u=upper_bins[ (upper_bins -x).abs().argmin()]
    
    
    
    return p_dict[s[1]['M'].iloc[(x-lower_bins).abs().argmin()]]
    
#s= sample_distribution_shape(resdf1, alpha, eta0)


#sample_distribution(5.8)