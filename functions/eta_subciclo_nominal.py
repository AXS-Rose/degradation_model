def eta_subciclo_nominal(extremos, modelo):
    ind = 0
    eta = 1
    while ind < len(extremos):
        subciclo=[extremos[ind],extremos[ind+1]]
        eta_k = modelo.get_factor(subciclo,False) # False usa eta nominal
        print("calculo para subciclo nominal: ",subciclo," eta = ",eta_k)
        eta *= eta_k 
        ind += 2
    return eta