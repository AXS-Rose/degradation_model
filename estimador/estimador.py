from .src import (
    ThModel_Extended,
    Modelo_Th_cuant_full,
    metropolis_weighted_deltas,
)
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.optimize import minimize


class Estimador2:
    def __init__(
        self,
        min_estimation_points: int,
        conserved_points: int,
        voc_thresh,
        voc_times,
        chain_burn: int,
        chain_samples: int,
        Model_kwargs,
        Estim_kwargs,
    ) -> None:
        """

        Esta clase es distinta de estimador por que considera la resistencia interna variable en funcion del SOC
        la funcion add_data_chunks cambia para considerar como entrada el soc y asi hacer la correccion por Rint
        del Voltaje, ademas ThModel_Ah tambien cambia al modelo de Aramis.

        ## Parámetros de la estimación en general
        min_estimation_points: cantidad mínima de puntos con la que se gatilla una estimación
        conserved_points: puntos que se conservarán para la estimación en el paso siguiente
        voc_thresh: umbral de corriente con el que se detecta el voc
        voc_times: umbral del tiempo de descanso con el que se detecta el voc


        ## Parámetros de la cadena
        chain_burn: cantidad de puntos a usar para la estimación
        chain_samples: cantidad de puntos con los que formar la cadena

        ## Argumentos del modelo de la batería
        **Model_kwargs: kwargs para el modelo equivalente de Thévennin
            - parámetros del voc, ie:
                - vl
                - v0
                - alpha
                - beta
                - gamma
                - r_int
                - Qmax

            - Se usará el parámetro de capacidad nominal Qmax (Ah) para obtener un estimado del SoH

        ## Argumentos del estimador
        **Estim_kwargs:
            - Parámetros del MCMC con triadas:
                - sigma_e: varianza de la distribución de error
                - sigma_p: varianza de la distribución de propuestas
                - w: exponente del peso asignado a las muestras. Por ejemplo: w = 2 implica que el ponderador será lambda_i = 1 - V_i^w/sum_j-N(V_j^w)
                - expo: exponente con el que se hará la estimación. Por ejempo: expo = 2 implica inferir Ah^2 como capacidad
                - bar: variable booleana que indica si queremos una barra de carga (tqdm) al hacer la inferencia.
        """

        # Variables relevantes para la operación
        self.min_estimation_points = min_estimation_points
        self.voc_thresh = voc_thresh
        self.voc_times = voc_times

        # assert 0 < conserved_points, 'La mínima cantidad de puntos a conservar es 1'
        self.conserved_points = conserved_points

        # Variables relevantes para la estimación
        self.chain_burn = chain_burn
        self.chain_samples = chain_samples
        self.estim_kwargs = Estim_kwargs

        # Puntos detectados en operación
        self.detected_points = None

        # Modelos de la curva y sus inversa
        self.modelo_th = ThModel_Extended(**Model_kwargs)
        self.modelo_th.fit_inverse(False)

        # parametros del capturador de puntos
        self.v0 = None
        self.ahc0 = None
        self.v_memo = []
        self.soc_memo = []
        self.ahc_memo = []
        self.c_memo = []
        self.consistent = 0  # variable que indica cuántas muestras consecutivas están por debajo del umbral
        self.Ahc = 0  # Memoria de ampere hyora consumido para cada dato
        self.detected_points = None

    def verosimilitud(self, Q, data):
        """
        Función de verosimilitud no normalizada para los datos en formato de triadas.
        Se asume ruido exponencial iid.

        Q: capacidad sugerida
        data: datos detectados en formato de triadas (V_pivote, Delta Ahc, V )

        """
        data = data[data[:, 1] != 0]
        # wf = np.log(abs(data[:, 1] / (self.modelo_th.voc_inv(data[:, 0]))))
        # print(data.shape)
        # data = data[0 < wf]
        # print(data.shape)

        # wf = np.log(abs(data[:, 1] / (self.modelo_th.voc_inv(data[:, 0]))))
        # wf = wf / sum(wf)
        # wf2 = -1 * np.log(1 / data[:, 2])

        # Si uno de los datos de SoC supera 1, la verosimilitud de ese valor de q debería ser 0!!!!!
        soc_data = self.modelo_th.voc_inv(data[:, 0]) - data[:, 1] / Q
        if 1 < soc_data.any():
            return 0, -100000

        modelo_datos = sum(
            -1
            / (2 * self.estim_kwargs.get("sigma_e"))
            # * wf
            # * wf2
            * (data[:, 2] - self.modelo_th.voc(np.clip(soc_data, 0, 1))) ** 2
        )

        return np.exp(modelo_datos), modelo_datos

    def verosimilitud_opuesta(self, Q, data, log=0):
        vero, logvero = self.verosimilitud(Q, data)
        if log:
            return -1 * logvero

        return -1 * vero

    def estimate_MCMC(self, prior_range, inform_ratio=False):
        """
        Da el valor de la cadena (cortada) en Ah^expo. Usa los datos guardados en la memeoria del objeto
        """

        chain_complete = metropolis_weighted_deltas(
            self.detected_points,
            model=self.verosimilitud,
            prior_range=prior_range,
            n_iter=self.chain_samples,
            **self.estim_kwargs,
        )

        # print(chain_complete)
        if inform_ratio:
            print(
                "Ratio de aceptación: ", len(chain_complete) / self.chain_samples * 100
            )

        return chain_complete[-self.chain_burn :]

    def estimate_ML(self, bounds, log=0, method="Nelder-Mead"):
        """
        Estima por máxima verosimilitud la capacidad nominal en base al esquema de triadas
        """
        minim = minimize(
            self.verosimilitud_opuesta,
            x0=[(bounds[0] + bounds[1]) / 2],
            args=(self.detected_points, log),
            tol=1e-9,
            method=method,
            bounds=[bounds],
            options={"maxiter": 100000},
        )
        # print(minim)

        if not minim.success:
            print("¡La estimación no converge!")

        return minim.x.item()

    def plot_verosimilitud(self, range_q):
        """
        Función que hace un plot de la verosimilitud y log-verosimilitud para un rango de Q's con los datos guardados en la memoria del estimador
        """
        vero = []
        logvero = []
        for q in range_q:
            v, lv = self.verosimilitud(q, self.detected_points)
            vero.append(v)
            logvero.append(lv)

        plt.figure(figsize=(15, 5))
        plt.plot(range_q, vero)
        plt.title("Plot de la función de verosimilitud a maximizar")
        plt.xlabel("Capacidad [Ah]")
        plt.ylabel("Verosimilitud")
        plt.show()

        plt.figure(figsize=(15, 5))
        plt.plot(range_q, logvero)
        plt.title("Plot de la función de log-verosimilitud a maximizar")
        plt.xlabel("Capacidad [Ah]")
        plt.ylabel("Log-verosimilitud")
        plt.show()

    def detect_iterest_points(self, voltage, current, dt, mean=True):
        self.Ahc += -1 * current * dt / 3600

        if abs(current) < self.voc_thresh:
            self.v_memo.append(voltage)
            self.c_memo.append(current)
            self.soc_memo.append(self.modelo_th.voc_inv(voltage))
            self.ahc_memo.append(self.Ahc)
            self.consistent += 1

        else:
            # Caso en el que los puntos están consistenetemente debajo del umbral
            if self.voc_times <= self.consistent:
                if mean:
                    r = self.modelo_th.calculate_R(np.mean(self.soc_memo))
                    v_corregido = np.mean(self.v_memo) - abs(
                        np.mean(self.c_memo)
                    ) * r * np.sign(np.mean(self.c_memo))
                else:
                    r = self.modelo_th.calculate_R(self.soc_memo[-1])
                    v_corregido = self.v_memo[-1] - abs(self.c_memo[-1]) * r * np.sign(
                        self.c_memo[-1]
                    )

                if self.v0 is None:  # caso en el que no hay voltaje pivote
                    if mean:
                        self.ahc0 = np.mean(self.ahc_memo)
                        self.v0 = v_corregido
                    else:
                        self.ahc0 = self.ahc_memo[-1]
                        self.v0 = v_corregido

                # Guardamos los datos
                if self.detected_points is None:
                    if mean:
                        self.detected_points = np.array(
                            [self.v0, np.mean(self.ahc_memo) - self.ahc0, v_corregido]
                        ).reshape(1, -1)
                    else:
                        self.detected_points = np.array(
                            [self.v0, self.ahc_memo[-1] - self.ahc0, v_corregido]
                        ).reshape(1, -1)

                    # Reiniciamos la memoria
                    self.v_memo = []
                    self.c_memo = []
                    self.soc_memo = []
                    self.ahc_memo = []
                    self.consistent = 0
                    return 1
                else:
                    if mean:
                        self.detected_points = np.concatenate(
                            [
                                self.detected_points,
                                np.array(
                                    [
                                        self.v0,
                                        np.mean(self.ahc_memo) - self.ahc0,
                                        v_corregido,
                                    ]
                                ).reshape(1, -1),
                            ],
                            0,
                        )
                    else:
                        self.detected_points = np.concatenate(
                            [
                                self.detected_points,
                                np.array(
                                    [
                                        self.v0,
                                        self.ahc_memo[-1] - self.ahc0,
                                        v_corregido,
                                    ]
                                ).reshape(1, -1),
                            ],
                            0,
                        )
                    # Reiniciamos la memoria
                    self.v_memo = []
                    self.c_memo = []
                    self.soc_memo = []
                    self.ahc_memo = []
                    self.consistent = 0
                    return 1

            else:
                self.consistent = 0
                self.v_memo = []
                self.c_memo = []
                self.soc_memo = []
                self.ahc_memo = []

        return 0

    # Función que resetea las variables guardadas en el estimador
    def reset_detector(self, mode: list):
        """
        Función que resetea las variables guardadas en el estimador
        El modo indica qué vaiables se resetean.
        Las siguientes combinaicones están disponibles:
        - [1,0]: Resetea el contador de apere hora consumido, la memoria y los pivotes. Es conveniente usarlo cuando cambia el experimento, pero no pasa un ciclo equivalente
        - [0,1]: Resetea los puntos importantes guardados en la memoria del estimador. ütil cuando se hace una estimaicón y es necesario volver a recolectar puntos
        """

        if mode[0]:
            self.v0 = None
            self.ahc0 = None
            self.v_memo = []
            self.soc_memo = []
            self.ahc_memo = []
            self.c_memo = []
            self.consistent = 0  # variable que indica cuántas muestras consecutivas están por debajo del umbral
            self.Ahc = 0  # Memoria de ampere hora consumido para cada dato

        if mode[1]:
            self.detected_points = None

    def fit_batt_model(
        self,
        params=dict(
            vL=1.50386725,
            v0=4.14655208,
            gamma=0.13020218,
            alpha=0.30099609,
            beta=4.55196918,
            Rint=0.00709686,
        ),
    ):
        """
        Función que ajusta los parámetros de la curva de Voltje a partir de un set de parámetros.
        Los parámetros a ajustar deben venir en un diccionario con las misma llaves del modelo de la batería:
            - process_noise
            - measurement_noise
            - Qmax
            - Rint
            - vL
            - v0
            - alpha
            - beta
            - gamma
            - x0
        Nota: el diccionario no tiene que contener obligatoriamente toas las llaves, en caso de faltar parámetros, se usarán los que tiene el modelo por defecto

        """

        for k, v in params.items():
            if not self.modelo_th.parameters.get(k) is None:
                self.modelo_th.parameters[k] = v
            else:
                print(f'No se encontó la llave "{k}"')

        print("Parámetros del modelo:")
        pprint(self.modelo_th.parameters)


class Estimador_cuant(Estimador2):
    def __init__(
        self,
        min_estimation_points: int,
        conserved_points: int,
        voc_thresh,
        voc_times,
        chain_burn: int,
        chain_samples: int,
        Model_kwargs,
        Estim_kwargs,
    ) -> None:
        """

        Esta clase es distinta de estimador por que considera la resistencia interna variable en funcion del SOC
        la funcion add_data_chunks cambia para considerar como entrada el soc y asi hacer la correccion por Rint
        del Voltaje, ademas ThModel_Ah tambien cambia al modelo de Aramis.

        Esta clase también incorpora cunatización en el sensor de voltaje, por lo que el modelo pasa a ser Modelo_Th_cuant_full, hija de ThModel_Ah

        ## Parámetros de la estimación en general
        min_estimation_points: cantidad mínima de puntos con la que se gatilla una estimación
        conserved_points: puntos que se conservarán para la estimación en el paso siguiente
        voc_thresh: umbral de corriente con el que se detecta el voc
        voc_times: umbral del tiempo de descanso con el que se detecta el voc


        ## Parámetros de la cadena
        chain_burn: cantidad de puntos a usar para la estimación
        chain_samples: cantidad de puntos con los que formar la cadena

        ## Argumentos del modelo de la batería
        **Model_kwargs: kwargs para el modelo equivalente de Thévennin
            - parámetros del voc, ie:
                - vl
                - v0
                - alpha
                - beta
                - gamma
                - r_int
                - Qmax

            - Se usará el parámetro de capacidad nominal Qmax (Ah) para obtener un estimado del SoH

        ## Argumentos del estimador
        **Estim_kwargs:
            - Parámetros del MCMC con triadas:
                - sigma_e: varianza de la distribución de error
                - sigma_p: varianza de la distribución de propuestas
                - w: exponente del peso asignado a las muestras. Por ejemplo: w = 2 implica que el ponderador será lambda_i = 1 - V_i^w/sum_j-N(V_j^w)
                - expo: exponente con el que se hará la estimación. Por ejempo: expo = 2 implica inferir Ah^2 como capacidad
                - bar: variable booleana que indica si queremos una barra de carga (tqdm) al hacer la inferencia.
        """

        # Variables relevantes para la operación
        self.min_estimation_points = min_estimation_points
        self.voc_thresh = voc_thresh
        self.voc_times = voc_times

        # assert 0 < conserved_points, 'La mínima cantidad de puntos a conservar es 1'
        self.conserved_points = conserved_points

        # Variables relevantes para la estimación
        self.chain_burn = chain_burn
        self.chain_samples = chain_samples
        self.estim_kwargs = Estim_kwargs

        # Puntos detectados en operación
        self.detected_points = None

        # Modelos de la curva y sus inversa
        self.modelo_th = Modelo_Th_cuant_full(**Model_kwargs)
        # self.modelo_th.fit_inverse(False)

        # parametros del capturador de puntos
        self.v0 = None
        self.ahc0 = None
        self.v_memo = []
        self.soc_memo = []
        self.ahc_memo = []
        self.c_memo = []
        self.consistent = 0  # variable que indica cuántas muestras consecutivas están por debajo del umbral
        self.Ahc = 0  # Memoria de ampere hyora consumido para cada dato
        self.detected_points = None
