from pyswarm import pso
import matplotlib.pyplot as plt
import numpy as np


def optimizar_con_pso(I, V, SoC, parametros_iniciales=None, limites=None, opts=None):
    """
    Optimiza un modelo de pseudo voltaje en circuito abierto (VoC) de una batería, incluyendo efectos de corrientes pequeñas,
    utilizando el algoritmo PSO (Particle Swarm Optimization). Esta función es útil para ajustar parámetros del modelo
    que mejor se alineen con datos experimentales bajo condiciones de corriente baja, ayudando a minimizar el error
    entre los datos reales y las predicciones del modelo.

    :param ruta_archivo: Ruta al archivo .mat que contiene los datos de la batería.
    :param parametros_iniciales: Lista opcional de valores iniciales para los parámetros del modelo.
    :param limites: Tupla de listas con límites inferiores y superiores para los parámetros del modelo.
    :param opts: Diccionario con opciones de configuración para el algoritmo PSO.

    :return: Diccionario con 'Parámetros optimizados' y 'Error Cuadrático Medio' del ajuste del modelo.
    """

    # Función para la curva
    def vk_function(SoC, I, V_L, V_0, Gamma, Alpha, Beta, R):
        return (
            V_L
            + (V_0 - V_L) * np.exp(Gamma * (SoC - 1))
            + Alpha * V_L * (SoC - 1)
            + (1 - Alpha) * V_L * (np.exp(-Beta) - np.exp(-Beta * np.sqrt(SoC)))
            - I * R
        )

    # Cargar los datos desde el archivo .mat
    # data = scipy.io.loadmat(ruta_archivo)
    # datos_crudos = data[list(data.keys())[-1]]
    # I = datos_crudos[:, 0]  # Corriente
    # V = datos_crudos[:, 1]  # Voltaje
    # capacidad_extraida = datos_crudos[:, 2]  # Capacidad extraída acumulada

    # Calcular SoC (Estado de Carga)
    # SoC = (max(capacidad_extraida) - capacidad_extraida) / max(capacidad_extraida)

    # Función objetivo para PSO
    def objective_function(x):
        V_L, V_0, Gamma, Alpha, Beta, R = x
        predicted_V = vk_function(SoC, I, V_L, V_0, Gamma, Alpha, Beta, R)
        return np.sum(
            (V - predicted_V) ** 2
        )  # Minimizar la suma de cuadrados de los errores

    # Establecer parámetros iniciales y límites si no se proporcionan
    if parametros_iniciales is None:
        parametros_iniciales = [1, 1, 1, 1, 1, 1]
    if limites is None:
        limites = ([0.001, 0.001, 0.001, 0.001, 0.001, 0.001], [5, 5, 1, 1, 5, 0.2])
    if opts is None:
        opts = {"swarmsize": 50, "maxiter": 100, "minstep": 1e-8, "minfunc": 1e-8}

    # Ejecutar PSO con opciones
    xopt, fopt = pso(objective_function, limites[0], limites[1], **opts)

    # Calcular el Error Cuadrático Medio (ECM)
    V_predicho = vk_function(SoC, I, *xopt)
    ECM = np.mean((V - V_predicho) ** 2)

    # Resultados
    resultados = {"Parámetros optimizados": xopt, "Error Cuadrático Medio": ECM}

    # Gráfico (opcional)
    plt.figure(figsize=(10, 6))
    plt.scatter(SoC, V, color="b", label="Datos reales")
    plt.scatter(SoC, V_predicho, color="r", label="Ajuste de curva con PSO")
    plt.xlabel("Estado de Carga (SoC)")
    plt.ylabel("Voltaje (V)")
    plt.legend()
    plt.show()

    return resultados


def optimizar_con_pso_operacional_esc(
    I, V, SoC, resultados_pso, factor_inferior, factor_superior, opts
):
    """
    Optimiza un modelo de voltaje en bornes de batería con datos operacionales usando PSO.

    Esta función ajusta un modelo de resistencia interna de la batería a datos operacionales,
    considerando un factor de escalamiento y un sesgo (bias) para la resistencia. Utiliza el
    algoritmo PSO para encontrar los parámetros que mejor se ajustan a los datos, basándose en
    el ECM.

    Argumentos:
    ruta_csv (str): Ruta al archivo CSV con los datos operacionales de la batería.
    valor_indicador (int): Valor del indicador para filtrar los datos.
    resultados_pso (array): Parámetros previamente optimizados del modelo.
    factor_inferior (float): Límite inferior para el factor de escalamiento.
    factor_superior (float): Límite superior para el factor de escalamiento.
    opts (dict): Opciones de configuración para el algoritmo PSO.

    Retorna:
    dict: Un diccionario con 'Parámetros optimizados' y 'Error Cuadrático Medio'.
    """

    # Coeficientes de la curva de resistencia para 5 amperios y límites para el bias
    p1, p2, p3, p4 = 0.3084, -0.2578, -0.05083, 0.1317
    bias_inferior, bias_superior = -0.05, 0.05  # Ajustar según sea necesario

    # Función de la curva de resistencia con bias
    def resistencia_curva(SoC, factor, bias):
        return factor * (p1 * SoC**3 + p2 * SoC**2 + p3 * SoC + p4) + bias

    def vk_op_function(SoC, I, V_L, V_0, Gamma, Alpha, Beta, factor, bias):
        R = resistencia_curva(SoC, factor, bias)
        return (
            V_L
            + (V_0 - V_L) * np.exp(Gamma * (SoC - 1))
            + Alpha * V_L * (SoC - 1)
            + (1 - Alpha) * V_L * (np.exp(-Beta) - np.exp(-Beta * np.sqrt(SoC)))
            + I * R
        )

    def objective_function(x):
        V_L, V_0, Gamma, Alpha, Beta, factor, bias = x
        predicted_V = vk_op_function(SoC, I, V_L, V_0, Gamma, Alpha, Beta, factor, bias)
        return np.mean((V - predicted_V) ** 2)

    # datos = pd.read_csv(ruta_csv, header=None)
    # datos_filtrados = datos[datos.iloc[:, 5].isin(np.atleast_1d(valor_indicador))]

    # I = datos_filtrados.iloc[:, 0]
    # V = datos_filtrados.iloc[:, 1]
    # SoC = datos_filtrados.iloc[:, -1]

    # Ajustar límites para incluir factor de escalamiento y bias
    limites_inferiores = np.append(
        np.array(resultados_pso[:5]) * factor_inferior, [factor_inferior, bias_inferior]
    )
    limites_superiores = np.append(
        np.array(resultados_pso[:5]) * factor_superior, [factor_superior, bias_superior]
    )

    xopt, fopt = pso(objective_function, limites_inferiores, limites_superiores, **opts)

    V_predicho = vk_op_function(SoC, I, *xopt)
    ECM = np.mean((V - V_predicho) ** 2)

    resultados_operacionales = {
        "Parámetros optimizados": xopt,
        "Error Cuadrático Medio": ECM,
    }

    plt.figure(figsize=(10, 6))
    plt.scatter(SoC, V, color="b", label="Datos operacionales filtrados")
    plt.scatter(SoC, V_predicho, color="r", label="Ajuste de curva con PSO")
    plt.xlabel("Estado de Carga (SoC)")
    plt.ylabel("Voltaje (V)")
    plt.legend()
    plt.show()

    return resultados_operacionales
