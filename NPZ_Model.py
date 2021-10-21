#!/usr/bin/env python
# coding: utf-8

# # Ecosystem Modelling
# 
# NP and NPZ models
# 


import numpy as np

#######################    NP     ############################

def eco_NP(P_t0, N_t0):
    
    ## Standard input paramenters

    v_max = 1.4     # Maximum Growth Rate Phyto (day-1)
    k_n = 0.1       # Half saturation constant Phyto (mmol/m3)
    Lambda_p = 0.05 # Phyto mortality rate (day-1) 
    mu_p = 1.0      # Remineralized fraction Phyto 
    supp = 0.0      # Nutrient supply (mmol/day)

    ## Initialisation parameters

    t0 = 0                           # Initial time (days)
    tf = 100                         # final time (days)
    dt = 0.01                           # time step (days)
    nstep = int(tf/dt)               # number of time steps (hourly)
    t = np.linspace(t0,tf,nstep+1)   # timestep array (days)


    ## Calculated input parameters

    P = np.zeros(len(t)+1)
    N = np.zeros(len(t)+1)
    time = np.zeros(len(t)+1)
    
    
    ## Variable input parameters
    
    P[0] = P_t0
    N[0] = N_t0


    for i in range(1,len(t)):
        i = int(i)
        
        # Equation 4.3.6 in Sarmiento & Gruber
        dP = (P[i-1] * (v_max * N[i-1] / (k_n + N[i-1])  # Uptake/Photosynthesis 
            - Lambda_p))                                 # Mortality               
        
        # Equation 4.3.7 in S&G
        dN = (P[i-1] * (-v_max * N[i-1] / (k_n + N[i-1]) # Phyto uptake
            + mu_p*Lambda_p)                             # Remin Phyto
            + supp)                                      # Supply through mixing/advection 

        P[i] = P[i-1] + dP * dt
        N[i] = N[i-1] + dN * dt
        time[i] = t[i]


        #print(dP, dN)
    Total_N = P + N
    
    return(P,N,Total_N,time)




#######################    NPZ     ############################

def eco_NPZ(P_t0, N_t0, Z_t0):

    ## Standard input paramenters

    v_max = 1.4     # Maximum Growth Rate Phyto (day-1)
    g_max = 2       # Maximum Growth Rate Zoo (day-1)
    k_n = 0.1       # Half saturation constant Phyto (mmol/m3)
    k_p = 2.8       # Half saturation constan Zoo (mmol/m3)
    Lambda_p = 0.05 # Phyto mortality rate (day-1) 
    Lambda_z = 0.12 # Zoo mortality rate (day-1)
    mu_p = 1.0      # Remineralized fraction Phyto
    mu_z = 1.0      # Remineralized fraction Zoo
    gamma_z = 0.4   # Assimilation Efficiency Zoo 
    supp = 0.0      # Nutrient supply (mmol/day)

    ## Initialisation parameters

    t0 = 0 # Initial time (days)
    tf = 100 # final time (days)
    dt = 0.01 #(days)
    nstep = int(tf/dt) # number of time steps
    t = np.linspace(t0,tf,nstep+1) # (days)


    ## Calculated input parameters

    P = np.zeros(len(t)+1)
    N = np.zeros(len(t)+1)
    Z = np.zeros(len(t)+1)
    time = np.zeros(len(t)+1)
   

    ## Variable input parameters
    
    P[0] = P_t0
    N[0] = N_t0
    Z[0] = Z_t0


    for i in range(1,len(t)):
        i = int(i)
        
        # Equation 4.3.11 in S&G
        dP = (P[i-1] * v_max * N[i-1] / (k_n + N[i-1])       # Uptake/Photosynthesis
            - P[i-1]*Lambda_p                                # Phyto mortality
            - Z[i-1]*g_max*P[i-1]/(k_p+P[i-1]))              # Grazing by Zoo

       
        # Equation 4.3.12 in S&G
        dZ = (Z[i-1] * gamma_z*g_max*P[i-1] / (k_p+P[i-1])     # Uptake by grazing
            - Z[i-1]*Lambda_z)                                 # Zoo Mortality
       
       # Equation 4.3.12 in S&G
        dN = (-P[i-1] * v_max * N[i-1]/(k_n + N[i-1])                # Phyto Uptake
            + P[i-1] * mu_p * Lambda_p                               # Remin of Phyto
            + Z[i-1] * mu_z * Lambda_z                               # Remin of Zoo
            + Z[i-1] * mu_z*(1-gamma_z)*g_max*P[i-1] / (k_p+P[i-1])  # Remin of Zoo excretions
            + supp)                                                  # Supply through mixing/advection


        P[i] = P[i-1] + dP * dt
        N[i] = N[i-1] + dN * dt
        Z[i] = Z[i-1] + dZ * dt
        time[i] = t[i]


        
    Total_N = P + N + Z
    
    return(P,N,Z,Total_N,time)




####################### Deep Ocean two-box model ##########################


def eco_NP_deep(P_t0, Ns_t0, Nd_t0, seasons):

    
    ## Standard input paramenters

    v_max = 1.4     # Maximum Growth Rate Phyto (day-1)
    k_n = 0.1       # Half saturation constant Phyto (mmol/m3)
    Lambda_p = 0.05 # Phyto mortality rate (day-1) 
    mu_p = 0.5      # Remineralized fraction Phyto 
    #supp = 0.01    # Nutrient supply (mmol/day)
    mixing = 0.01   # fraction of mixing between boxes (fraction/day)

    ## Initialisation parameters

    t0 = 0                           # Initial time (days)
    tf = 365*2                       # final time (days)
    dt = 0.01                        # time step (days)
    nstep = int(tf/dt)               # number of time steps (hourly)
    t = np.linspace(t0,tf,nstep+1)   # timestep array (days)


    ## Calculated input parameters

    P = np.zeros(len(t)+1)
    Ns = np.zeros(len(t)+1)
    Nd = np.zeros(len(t)+1)
    time = np.zeros(len(t)+1)
    
    
    ## Variable input parameters
    
    P[0] = P_t0
    Ns[0] = Ns_t0
    Nd[0] = Nd_t0


    for i in range(1,len(t)):
        i = int(i)
        
        # Equation 4.3.6 in Sarmiento & Gruber
        dP = (P[i-1] * (v_max * Ns[i-1] / (k_n + Ns[i-1])       # Uptake/Photosynthesis 
            - Lambda_p))                                        # Mortality               
        
        if seasons==True:
            # Equation 4.3.7 in S&G
            dNs = (P[i-1] * (-v_max * Ns[i-1] / (k_n + Ns[i-1])                     # Phyto uptake
                + mu_p*Lambda_p)                                                    # Remin Phyto
                + mixing * abs(np.sin(t[i-1]*0.009)) * (Nd[i-1] - Ns[i-1]))         # Mixing with the deep box 


            dNd = ((1 - mu_p)*Lambda_p*P[i-1]                                    # Export of phyto
                + mixing * abs(np.sin(t[i-1]*0.009)) * (Ns[i-1] - Nd[i-1]))      # Seasonal Mixing with the surface box


        if seasons==False:
            
            dNs = (P[i-1] * (-v_max * Ns[i-1] / (k_n + Ns[i-1])   # Phyto uptake
                + mu_p*Lambda_p)                                  # Remin Phyto
                + mixing * (Nd[i-1] - Ns[i-1]))                   # Mixing with the deep box 


            dNd = ((1 - mu_p)*Lambda_p*P[i-1]                     # Export of phyto
                + mixing * (Ns[i-1] - Nd[i-1]))                   # Mixing with the surface box
            


        P[i] = P[i-1] + dP * dt
        Ns[i] = Ns[i-1] + dNs * dt
        Nd[i] = Nd[i-1] + dNd * dt
        time[i] = t[i]


    Total_N = P + Ns + Nd 
    
    return(P,Ns,Nd,Total_N,time)

