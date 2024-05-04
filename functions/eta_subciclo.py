def eta_subciclo(extremos, modelo):
    ind = 0
    eta = 1
    while ind < len(extremos):
        subciclo=[extremos[ind],extremos[ind+1]]
        eta_k = modelo.get_factor(subciclo,True)
        print("calculo para subciclo: ",subciclo, "eta = ",eta_k)
        eta *= eta_k 
        ind += 2
    return eta