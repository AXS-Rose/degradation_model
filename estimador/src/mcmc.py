import numpy as np
import numpy.random as rnd
from tqdm import tqdm


def metropolis_weighted_deltas(
    data,
    model,
    n_iter: float,
    sigma_p: float,
    prior_range: tuple = (0, 100),
    bar=True,
    sigma_e=1,
):
    """
    Algoritmo de Metropolis para samplear de la posterior considerando:
        - Prior uniforme: Prior range
        - Observaciones de la conjunta distribuyen como error Gaussiano: sigma_e
        - El error de las observaciones es iid.
        - Distribución de propuesta Gaussiana: sigma_e
        - Modelo: model(data, x) (debe estar vectorizado)
    """
    # Inicialización
    chain = []
    # prior_range = [r**expo for r in prior_range]
    x_ = rnd.uniform(*prior_range)
    den, _ = model(x_, data)
    cnt = 0
    while abs(den) < 1e-128:
        if 100 < cnt:
            break
        x_ = rnd.uniform(*prior_range)
        den, _ = model(x_, data)
        cnt += 1
    # x_ = rnd.normal(*prior_range)
    # print(den)

    for _ in tqdm(range(int(n_iter)), disable=not bar):
        # Generamos una propuesta
        # prop = 0
        # while not prop:
        x = rnd.normal(x_, sigma_p)
        # if (
        #     prior_range[0] <= x and x <= prior_range[1]
        # ):  # Nos aseguramos que el proposal está en el soporte
        #     prop = 1

        u = rnd.uniform()
        # print(x)

        # Transformamos los datos de triadas a duplas

        # Generamos el factor de pesos en función del Ahc

        # Evaluamos con nuestras observaciones (Asumimos error gaussiano)
        num, _ = (
            model(x, data)
            if prior_range[0] < x and x < prior_range[1]
            else (np.inf, np.inf)
        )
        den, _ = model(x_, data)

        # print(x, num, x_, den)

        if num == np.inf or num == 0:
            continue  # assert False

        A = num / den  # -> toma la verosimilitud

        # print(A)
        A = min(1, A)

        if u <= A:  # Aceptamos
            chain.append(x)
            x_ = x
            # print('a')

    return chain
