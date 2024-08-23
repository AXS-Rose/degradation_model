import pandas as pd
import numpy as np

def log_a(a,x):
    y = np.log(x)/np.log(a)
    return y

def nml_factors(factor_dict,eta_0):
    # Ahora normalizamos los factores
    sr_numeric_0 = 100
    # normal_factors = []
    # eq_cycles = []

    for sr_k, factors in factor_dict.items():
        # obtenemos el sr de cada caso
        sr_range = [float(x) for x in sr_k.split("-")]
        sr_numeric = sr_range[0] - sr_range[1]
        # normlizamos el factor
        eta_k = factors[0]*eta_0
        new_factor = (eta_k**(sr_numeric_0/sr_numeric))/eta_0
        # eq_cycle = 1/(log_a(EOL_0,eta_k**(sr_numeric_0/sr_numeric)))
        ## actualizamos los nuevos valores
        # eq_cycles.append(eq_cycle)
        # normal_factors.append(new_factor)
        factors[0] = new_factor

    # factors_column = factors_column + "_nml"
    # cycles_column = cycles_column + "_nml"
        
    # df[factors_column] = normal_factors
    # df[cycles_column] = eq_cycles

def adap_factors(factor_dict,new_cycles,old_cycles,new_EOL):
    ## Adaptamos los factores a una nueva celda
    new_eta0 = new_EOL**(1/new_cycles)

    # adaptamos para factores normalizados
    # adapted_factors = []
    # adapted_cycles = []

    for sr_k, factors in factor_dict.items():
        adap_factor = factors[0]**(old_cycles/new_cycles)
        adap_etak = adap_factor*new_eta0
        adap_cycle = 1/(log_a(new_EOL,adap_etak))
        # actualizamos los nuevos valores
        # adapted_cycles.append(adap_cycle)
        # adapted_factors.append(adap_factor)
        factors[0] = adap_factor
        
    # factors_column = factors_column + "_adap"
    # cycles_column = cycles_column + "_adap"

    # df[factors_column] = adapted_factors
    # df[cycles_column] = adapted_cycles

def unnml_factors(df,new_cycles,new_EOL,factors_column,cycles_column):
    new_eta0 = new_EOL**(1/new_cycles)
    sr_numeric_0 = 100
    unnoml_adapted_factors = []
    unnoml_adapted_cycles = []
    for sr_k, factor in zip(df.SR, df.Factors_nml_adap):
        # obtenemos el sr de cada caso
        sr_range = [float(x) for x in sr_k.split("-")]
        sr_numeric = sr_range[0] - sr_range[1]  # SR
        # normlizamos el factor
        eta_k = factor*new_eta0
        unnml_eta_k = eta_k**(sr_numeric/sr_numeric_0)
        unnml_factor = unnml_eta_k/new_eta0
        eq_cycle = 1/(log_a(new_EOL,unnml_eta_k))
        # actualizamos los nuevos valores
        unnoml_adapted_cycles.append(eq_cycle)
        unnoml_adapted_factors.append(unnml_factor)

    factors_column = factors_column + "_unnml"
    cycles_column = cycles_column + "_unnml"
    df[factors_column] = unnoml_adapted_factors
    df[cycles_column] = unnoml_adapted_cycles
    return df

def linear_factors(factor_dict,EOL_0,cycles_0):
    eta_0 = EOL_0**(1/cycles_0)
    sr_numeric_0 = 100
    for sr_k, factors in factor_dict.items():
        # obtenemos el sr de cada caso
        sr_range = [float(x) for x in sr_k.split("-")]
        sr_numeric = sr_range[0] - sr_range[1]
        lin_cycles = cycles_0*sr_numeric_0/sr_numeric
        # normlizamos el factor
        eta_k = EOL_0**(1/lin_cycles)
        new_factor = eta_k/eta_0
        # eq_cycle = 1/(log_a(EOL_0,eta_k**(sr_numeric_0/sr_numeric)))
        ## actualizamos los nuevos valores
        # eq_cycles.append(eq_cycle)
        # normal_factors.append(new_factor)
        factors[0] = new_factor

    # factors_column = factors_column + "_nml"
    # cycles_column = cycles_column + "_nml"
        
    # df[factors_column] = normal_factors
    # df[cycles_column] = eq_cycles

def temp_factor(temp):
    # definimos los coeficientes ya fiteados
    coef = np.array([-1.25736777e-11,  4.79001576e-10,  5.01975901e-08,  5.04209738e-07,
              -3.61735311e-04,  1.12614410e-02,  1.02131291e+00])
    # creamos la funcion generadora de factores por temperatura
    polynomial = np.poly1d(coef)
    # Calculamos el factor ponderador por temperatura
    factor = polynomial(temp)
    return factor