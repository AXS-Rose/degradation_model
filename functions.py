def eta_subciclo(extremos, modelo):
    ind = 0
    eta = 1
    while ind < len(extremos):
        subciclo=[extremos[ind],extremos[ind+1]]
        eta_k = modelo.get_factor(subciclo)
        print("calculo para subciclo: ",subciclo, "eta = ",eta_k)
        eta *= eta_k 
        ind += 2
    return eta

def extremos_soc(soc):
    extremos = []
    soc_val = soc.SoC.values
    step_ind_val = soc.step_ind.values
    extremos.append(soc_val[0]) # agregamos el punto inicial (si o si máximo)

    # obtenemos los extremos intermedios
    for ind, elem in enumerate(soc_val[:-1]):
        # En caso de que sea minimo
        if ((step_ind_val[ind] in [13, 14]) and (step_ind_val[ind+1] not in [13, 14])):  
            extremos.append(soc_val[ind])
        # En caso de que sea maximo
        elif ((step_ind_val[ind] not in [13, 14]) and (step_ind_val[ind+1] in [13, 14])):  
            extremos.append(soc_val[ind])
    # agregamos el punto final (si o si mínimo)
    extremos.append(soc_val[-1])
    return extremos