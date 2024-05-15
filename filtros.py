from estimador import Estimador2, Estimador_cuant
from estimador import *
import numpy as np


class FiltrosAnidados(Estimador2):
    def __init__(
        self,
        # Parámetros del filtro de autonomía
        # soc_0 = 1,
        sigma_autonomia=1,
        # Parámetros del filtro de autonomía
        # soc_0 = 1,
        sigma_capacidad=1,
        # Parámetros del estimador
        min_estimation_points: int = 0,
        conserved_points: int = 0,
        voc_thresh=0,
        voc_times=0,
        chain_burn: int = 1,
        chain_samples: int = 10,
        Model_kwargs={},
        Estim_kwargs=[],
    ):

        Estimador2.__init__(
            self,
            min_estimation_points,
            conserved_points,
            voc_thresh,
            voc_times,
            chain_burn,
            chain_samples,
            Model_kwargs,
            Estim_kwargs,
        )

        # parámetros del filtro de autonomía
        # self.soc_0 = soc_0
        self.sigma_autonomia = sigma_autonomia
        self.sigma_capacidad = sigma_capacidad
        self.Q_inst = self.modelo_th.parameters["Qmax"]

        # Forzamnos el setup de knn
        self.modelo_th.setup_knn()

    # ============================= Métodos del filtro de autonomía =============================
    # Función que implementa el filtro de partículas pra el soc

    def filtrar_soc(
        self,
        particulas: np.array,
        pesos: np.array,
        voltaje: float,
        corriente: float,
        corriente_: float,
        dt: float,
    ):

        # verificamos que las partículas entrantes no tengan valores de soc imposibles
        particulas = np.clip(particulas, 0, 1)

        # Proyectamos las partículas con la ecuanción dinámica
        particulas = (
            particulas
            + dt * corriente_ / (self.Q_inst * 3600)
            + np.random.normal(0, 0.005, particulas.shape)
        )  # TODO: Agregar el ruido

        # Obtenemos el voltaje a partir de la correinet entrante y las partículas proyectadas
        v_particulas = self.modelo_th.voc(
            particulas
        ) - corriente * self.modelo_th.calculate_R(particulas)

        # evaluamos la verosimilitud
        verosimilitud = voltaje - v_particulas
        pesos = (
            pesos
            * np.exp(verosimilitud**2 / (-2 * self.sigma_autonomia**2))
            / (np.sqrt(2 * np.pi * self.sigma_autonomia))
        )
        pesos = pesos / sum(pesos)  # Normalizamos los pesos

        if np.isnan(pesos).any():
            pesos = np.ones(len(particulas)) / len(particulas)

        # Resampling
        if 1 / sum(pesos**2) < 0.85 * len(particulas):
            particulas = np.random.choice(particulas, particulas.shape, p=pesos)
            pesos = np.ones(len(particulas)) / len(particulas)

        soc_ponderado = sum(particulas * pesos)

        # Retornamos las partículas y sus pesos
        return particulas, pesos, soc_ponderado, v_particulas

    # ============================= Métodos del filtro de capacidad =============================
    # Función que implementa el filtro para la capacidad en base al estimador
    def get_factor(self, soc):
        # Obtenermos el SSR y el ASSR
        ssr = max(soc) - min(soc)
        assr = (max(soc) + min(soc)) / 2
        sr_numeric_0 = 100

        # Calculamos el valor de eta
        knn_factor = self.modelo_th.knn.predict(
            np.array([[assr, ssr, self.modelo_th.parameters["degradation_percentage"]]])
        )
        eta = (self.modelo_th.parameters["degradation_percentage"]) ** (
                1 / self.modelo_th.parameters["life_cycles"])
        
        etak = knn_factor * eta
        etak_unnml = etak**(ssr/sr_numeric_0)
        print("eta0: ",eta," etak: ",etak," normalizado a: ",etak_unnml," para subciclo: ",soc)

        return etak_unnml
        # return knn_factor

    def filtrar_q(
        self,
        particulas: np.array,
        pesos: np.array,
        soc,
        q_estimador: float,
        std_dev: float,
    ):

        # Proyectamos las partículas de capacidad
        particulas = self.get_factor(soc) * particulas + np.random.normal(
            0, std_dev, particulas.shape
        )  # TODO: agregar ruido

        # evaluamos la verosimilitud
        verosimilitud = q_estimador - particulas
        # print(sum(pesos))
        pesos = (
            pesos
            * np.exp(verosimilitud**2 / (-2 * self.sigma_capacidad**2))
            / (np.sqrt(2 * np.pi * self.sigma_capacidad))
        )
        # if sum(pesos) == 0: print(pesos)
        # print(sum(pesos), '\n')
        pesos = pesos / sum(pesos)  # Normalizamos los pesos

        if np.isnan(pesos).any():
            pesos = np.ones(len(particulas)) / len(particulas)
            particulas = np.random.choice(particulas, particulas.shape, p=pesos)

        # Resampling
        if 1 / sum(pesos**2) < 0.85 * len(particulas):
            particulas = np.random.choice(particulas, particulas.shape, p=pesos)
            pesos = np.ones(len(particulas)) / len(particulas)

        capacidad_ponderada = sum(particulas * pesos)

        # Retornamos las partículas y sus pesos
        return particulas, pesos, capacidad_ponderada

class FiltrosAnidadosCuant(Estimador_cuant):
    def __init__(
        self,
        # Parámetros del filtro de autonomía
        # soc_0 = 1,
        sigma_autonomia=1,
        # Parámetros del filtro de autonomía
        # soc_0 = 1,
        sigma_capacidad=1,
        # Parámetros del estimador
        min_estimation_points: int = 0,
        conserved_points: int = 0,
        voc_thresh=0,
        voc_times=0,
        chain_burn: int = 1,
        chain_samples: int = 10,
        Model_kwargs={},
        Estim_kwargs=[],
    ):

        Estimador_cuant.__init__(
            self,
            min_estimation_points,
            conserved_points,
            voc_thresh,
            voc_times,
            chain_burn,
            chain_samples,
            Model_kwargs,
            Estim_kwargs,
        )

        # parámetros del filtro de autonomía
        # self.soc_0 = soc_0
        self.sigma_autonomia = sigma_autonomia
        self.sigma_capacidad = sigma_capacidad
        self.Q_inst = self.modelo_th.parameters["Qmax"]

        # Forzamnos el setup de knn
        self.modelo_th.setup_knn()

    # ============================= Métodos del filtro de autonomía =============================
    # Función que implementa el filtro de partículas pra el soc
    def filtrar_soc(
        self,
        particulas: np.array,
        pesos: np.array,
        voltaje: float,
        corriente: float,
        corriente_: float,
        dt: float,
    ):

        # verificamos que las partículas entrantes no tengan valores de soc imposibles
        particulas = np.clip(particulas, 0, 1)

        # Proyectamos las partículas con la ecuanción dinámica
        particulas = (
            particulas
            + dt * corriente_ / (self.Q_inst * 3600)
            + np.random.normal(0, 0.005, particulas.shape)
        )  # TODO: Agregar el ruido

        # Obtenemos el voltaje a partir de la correinet entrante y las partículas proyectadas
        v_particulas = self.modelo_th.voc(
            particulas
        ) - corriente * self.modelo_th.calculate_R(particulas)

        # evaluamos la verosimilitud
        verosimilitud = voltaje - v_particulas
        pesos = (
            pesos
            * np.exp(verosimilitud**2 / (-2 * self.sigma_autonomia**2))
            / (np.sqrt(2 * np.pi * self.sigma_autonomia))
        )
        pesos = pesos / sum(pesos)  # Normalizamos los pesos

        if np.isnan(pesos).any():
            pesos = np.ones(len(particulas)) / len(particulas)

        # Resampling
        if 1 / sum(pesos**2) < 0.85 * len(particulas):
            particulas = np.random.choice(particulas, particulas.shape, p=pesos)
            pesos = np.ones(len(particulas)) / len(particulas)

        soc_ponderado = sum(particulas * pesos)

        # Retornamos las partículas y sus pesos
        return particulas, pesos, soc_ponderado, v_particulas

    # ============================= Métodos del filtro de capacidad =============================
    # Función que implementa el filtro para la capacidad en base al estimador
    def get_factor(self, soc):
        # Obtenermos el SSR y el ASSR
        ssr = max(soc) - min(soc)
        assr = (max(soc) + min(soc)) / 2

        # Calculamos el valor de eta

        knn_factor = self.modelo_th.knn.predict(
            np.array([[assr, ssr, self.modelo_th.parameters["degradation_percentage"]]])
        )
        eta = (self.modelo_th.parameters["degradation_percentage"]) ** (
            1 / self.modelo_th.parameters["life_cycles"]
        )
        return knn_factor * eta
        # return knn_factor

    def filtrar_q(
        self,
        particulas: np.array,
        pesos: np.array,
        soc,
        q_estimador: float,
        std_dev: float,
    ):

        # Proyectamos las partículas de capacidad
        particulas = self.get_factor(soc) * particulas + np.random.normal(
            0, std_dev, particulas.shape
        )  # TODO: agregar ruido

        # evaluamos la verosimilitud
        verosimilitud = q_estimador - particulas
        # print(sum(pesos))
        pesos = (
            pesos
            * np.exp(verosimilitud**2 / (-2 * self.sigma_capacidad**2))
            / (np.sqrt(2 * np.pi * self.sigma_capacidad))
        )
        # if sum(pesos) == 0: print(pesos)
        # print(sum(pesos), '\n')
        pesos = pesos / sum(pesos)  # Normalizamos los pesos

        if np.isnan(pesos).any():
            pesos = np.ones(len(particulas)) / len(particulas)
            particulas = np.random.choice(particulas, particulas.shape, p=pesos)

        # Resampling
        if 1 / sum(pesos**2) < 0.85 * len(particulas):
            particulas = np.random.choice(particulas, particulas.shape, p=pesos)
            pesos = np.ones(len(particulas)) / len(particulas)

        capacidad_ponderada = sum(particulas * pesos)

        # Retornamos las partículas y sus pesos
        return particulas, pesos, capacidad_ponderada
