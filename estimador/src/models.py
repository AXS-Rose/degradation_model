from prog_models import PrognosticsModel
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import KBinsDiscretizer
from .newcell_functions import *



class ThModel_Extended(PrognosticsModel):
    is_vectorized = True
    inputs = ["i"]
    states = ["soc", "Ck", "i_", "etak", "SoCacumk"]
    outputs = ["v", "R"]
    events = ["EOD"]

    default_parameters = {
        "Qmax": 40,  # Carga nominal (Ah)
        "A_Rint": 0.3084,  # Coeficiente A de la fórmula de resistencia
        "B_Rint": -0.2578,  # Coeficiente B
        "C_Rint": -0.05083,  # Coeficiente C
        "D_Rint": 0.1317,  # Coeficiente D
        "Factor_Rint": 0.1879029,  # Factor multiplicativo
        "Bias_Rint": 0.01454392,  # Sesgo aditivo
        "vL": 4.971,
        "v0": 26.792,
        "alpha": -2.6314,
        "beta": 0.508,
        "gamma": 0.556,
        "degradation_percentage": 0.8,  # Por ejemplo, 80% de degradación
        "life_cycles": 1000,  # Por ejemplo, 1000 ciclos de vida
        "adapt_cell" : False, # True si se aplica la corrección para nueva celda
        "Factor_R_SOH0": 283.71948548 / 1000,
        "Factor_R_SOH1": -572.76721458 / 1000,
        "Factor_R_SOH2": 320.37195027 / 1000,
        "Factor_R_SOH3": 21.40399288 / 1000,
        # Datos de degradación
        "degradation_data": {
            "100-0":        [1.0],
            "100-25":       [1.00000266],
            "75-0":         [1.0000186],
            "100-50":       [0.99999203],
            "75-25":        [1.00001521],
            "50-0":         [1.00002874],
            "100-75":       [1.00002146],
            "75-50":        [1.00000881],
            "62.5-37.5":    [1.00000620],
            "50-25":        [1.00003347],
            "25-0":         [1.00004184],
        },
        "x0": {  # cond iniciales
            "soc": 1,
            "Ck": 40,  # Capacidad inicial
            "i_": 0,
            "etak": 1,  # Eta inicial
            "SoCacumk": 0,  # SoC acumulado inicial,
        },
    }

    def initialize(self, *args, eod_thresh=0.05, **kwargs):
        self.eod_thresh = eod_thresh
        self.memoria_soc = []
        super().initialize(
            *args, **kwargs
        )  # Llama a la inicialización de la clase base

        # if self.default_parameters["adapt_cell"]:
        #     self.adapt_degradation()
        #     print("modelo de degradación adapatado a nueva celda")

        self.setup_knn()  # Configura el modelo k-NN
        # Estado inicial
        initial_state = {
            "soc": 1,  # Estado de carga inicial
            "Ck": self.parameters["Qmax"],  # Capacidad inicial
            "i_": 0,  # Corriente inicial
            "etak": 1,  # Eta inicial
            "SoCacumk": 0,  # SoC acumulado inicial
        }
        return initial_state

    def voc(self, soc):
        soc = np.clip(soc, 0, 1)
        a = (self.parameters["v0"] - self.parameters["vL"]) * np.exp(
            self.parameters["gamma"] * (soc - 1)
        )
        b = self.parameters["alpha"] * self.parameters["vL"] * (soc - 1)
        c = (
            (1 - self.parameters["alpha"])
            * self.parameters["vL"]
            * (
                np.exp(-1 * self.parameters["beta"])
                - np.exp(-1 * self.parameters["beta"] * np.sqrt(soc))
            )
        )
        return self.parameters["vL"] + a + b + c

    def extended_sigmoid_cubic(self, x, L, k, x0, a, b, c):
        return (L / (1 + np.exp(-k * (x - x0)))) + (a * x + b) ** 3 + c

    def fit_inverse(self, plot=False):
        soc = np.linspace(0, 1, 10000)
        voc = self.voc(soc)
        self.inverse_params, _ = curve_fit(
            self.extended_sigmoid_cubic,
            voc,
            soc,
            p0=[1, 1, voc.mean(), 0, 1, 0],
            maxfev=1500000,
        )

        if plot:
            plt.figure(figsize=(15, 5))
            plt.plot(voc, soc, label="gt")
            plt.plot(
                voc,
                self.extended_sigmoid_cubic(voc, *self.inverse_params),
                label="estim",
            )
            plt.xlabel("Voc")
            plt.ylabel("SoC")
            plt.legend()
            plt.title("Inversa estimada de Voc(SoC)")
            plt.suptitle(
                f"Mae: {np.mean(abs(self.extended_sigmoid_cubic(voc, *self.inverse_params)-soc)):.3f}"
            )
            plt.plot()

    def voc_inv(self, voc):
        try:
            return np.clip(self.extended_sigmoid_cubic(voc, *self.inverse_params), 0, 1)
        except:
            assert (
                False
            ), "Hubo un error con la inversa....\nHint: Asegurate de haber fiteado los parámetros con self.fit_inverse()"

    def calculate_R(self, soc_op):
        A = self.parameters["A_Rint"]
        B = self.parameters["B_Rint"]
        C = self.parameters["C_Rint"]
        D = self.parameters["D_Rint"]
        factor = self.parameters["Factor_Rint"]
        bias = self.parameters["Bias_Rint"]

        R = factor * (A * soc_op**3 + B * soc_op**2 + C * soc_op + D) + bias
        return R

    def output(self, x):
        # Rint_actualizado = self.calculate_R(x["soc"], x["Ck"])
        Rint_actualizado = self.calculate_R(x["soc"])
        voc = self.voc(x["soc"])
        noise = rnd.normal(0, self.parameters["measurement_noise"]["v"])
        return self.OutputContainer(
            {"v": voc + Rint_actualizado * x["i_"] + noise, "R": Rint_actualizado}
        )

    def next_state(self, x, u, dt):
        soc_ = x["soc"] + u["i"] * dt / (x["Ck"] * 3600)
        i_barra = u["i"]

        # Acumular el cambio en SoC
        delta_soc = u["i"] * dt / (x["Ck"] * 3600)
        SoCacumk = x["SoCacumk"] - delta_soc if u["i"] < 0 else x["SoCacumk"]

        # Comprobar si se ha completado un ciclo
        if x["SoCacumk"] >= 1:
            # Calcular eta al final de un ciclo
            etak = self.calculate_eta()
            # Actualizar capacidad
            Ck = self.calculate_Cap(etak, x["Ck"])
            # Resetear SoCacumk y memoria_soc para el próximo ciclo
            SoCacumk = 0
            self.memoria_soc = []
        else:
            # Continuar acumulando SoC en memoria_soc durante el ciclo
            self.memoria_soc.append(soc_)
            # Mantener los valores actuales de eta y capacidad
            etak = x["etak"]
            Ck = x["Ck"]

        return self.StateContainer(
            {"soc": soc_, "Ck": Ck, "i_": i_barra, "etak": etak, "SoCacumk": SoCacumk}
        )

    def calculate_Cap(self, etak, Cap_prev):
        Ck = etak * Cap_prev
        return Ck

    def adapt_degradation(self):
        # degradation_percentage = self.parameters["degradation_percentage"]
        new_cycles = self.parameters["life_cycles"]
        new_EOL = self.parameters["degradation_percentage"]
        cycles_0 = 4000
        EOL_0 = 0.8
        eta_0 = EOL_0**(1/cycles_0)
        
        nml_factors(self.parameters["degradation_data"],eta_0)
        adap_factors(self.parameters["degradation_data"],new_cycles,cycles_0,new_EOL)

    def setup_knn(self):
        # Preparar los datos para el modelo k-NN
        X = []
        y = []

        for sr, factors in self.parameters["degradation_data"].items():
            sr_range = [float(x) for x in sr.split("-")]
            sr_numeric = sr_range[0] - sr_range[1]  # SR
            asr_numeric = sum(sr_range) / 2  # ASR
            for degradation_percentage, factor in zip([0.7, 0.8, 0.85], factors):
                X.append([asr_numeric, sr_numeric, degradation_percentage])
                y.append(factor)

        # Convertir a numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Entrenar el modelo k-NN
        self.knn = KNeighborsRegressor(n_neighbors=3, weights="distance")
        self.knn.fit(X, y)  # Entrenar el modelo k-NN

    def calculate_eta(self):
        # Calcular eta basado en la fórmula
        eta = (self.parameters["degradation_percentage"]) ** (
            1 / self.parameters["life_cycles"]
        )

        # Multiplicar eta por el factor de KNN
        knn_factor = self.calculate_knn_factor()
        etak = eta * knn_factor

        return etak

    def calculate_knn_factor(self):
        max_soc = max(self.memoria_soc) if self.memoria_soc else 1
        min_soc = min(self.memoria_soc) if self.memoria_soc else 0
        sr = max_soc - min_soc
        asr = (max_soc + min_soc) / 2

        # Utilizar el modelo k-NN para predecir el factor
        factor_predicted = self.knn.predict(
            np.array([[asr, sr, self.parameters["degradation_percentage"]]])
        )
        return factor_predicted[0]

    def event_state(self, x):
        return {"EOD": x["soc"]}

    def threshold_met(self, x):
        condition = x["soc"] < self.eod_thresh
        return {"EOD": condition}




# =========================================================================================
# Modelo cuantizado
# =========================================================================================

class Modelo_Th_cuant_full(PrognosticsModel):
    is_vectorized = True
    inputs = ["i"]
    states = ["soc", "Ck", "i_", "etak", "SoCacumk"]
    outputs = ["v", "R"]
    events = ["EOD"]

    default_parameters = {
        "Qmax": 40,  # Carga nominal (Ah)
        "A_Rint": 0.3084,  # Coeficiente A de la fórmula de resistencia
        "B_Rint": -0.2578,  # Coeficiente B
        "C_Rint": -0.05083,  # Coeficiente C
        "D_Rint": 0.1317,  # Coeficiente D
        "Factor_Rint": 0.1879029,  # Factor multiplicativo
        "Bias_Rint": 0.01454392,  # Sesgo aditivo
        "vL": 4.971,
        "v0": 26.792,
        "alpha": -2.6314,
        "beta": 0.508,
        "gamma": 0.556,
        "degradation_percentage": 0.8,  # Por ejemplo, 80% de degradación
        "life_cycles": 1000,  # Por ejemplo, 1000 ciclos de vida
        "Factor_R_SOH0": 283.71948548 / 1000,
        "Factor_R_SOH1": -572.76721458 / 1000,
        "Factor_R_SOH2": 320.37195027 / 1000,
        "Factor_R_SOH3": 21.40399288 / 1000,
        # Datos de degradación
        "degradation_data": {
            "100-0": [1.0, 1.0, 1.0],
            "100-25": [1.000003, 1.00000266, 1.00000193],
            "75-0": [1.000024, 1.0000186, 1.00001354],
            "100-50": [0.999989, 0.99999203, 0.9999942],
            "75-25": [1.000019, 1.00001521, 1.00001108],
            "50-0": [1.000037, 1.00002874, 1.00002093],
            "100-75": [1.000027, 1.00002146, 1.00001563],
            "75-50": [1.000011, 1.00000881, 1.00000642],
            "62.5-37.5": [1.000008, 1.00000620, 1.00000451],
            "50-25": [1.000043, 1.00003347, 1.00002438],
            "25-0": [1.000054, 1.00004184, 1.00003047],
        },
        "x0": {  # cond iniciales
            "soc": 1,
            "Ck": 40,  # Capacidad inicial
            "i_": 0,
            "etak": 1,  # Eta inicial
            "SoCacumk": 0,  # SoC acumulado inicial,
        },
    }

    def initialize(self,*args, eod_thresh=0.05, **kwargs):
        self.eod_thresh = eod_thresh
        self.memoria_soc = []
        super().initialize(
            *args, **kwargs
        )  # Llama a la inicialización de la clase base
        self.setup_knn()  # Configura el modelo k-NN

        # Estado inicial
        initial_state = {
            "soc": 1,  # Estado de carga inicial
            "Ck": self.parameters["Qmax"],  # Capacidad inicial
            "i_": 0,  # Corriente inicial
            "etak": 1,  # Eta inicial
            "SoCacumk": 0,  # SoC acumulado inicial
        }
        
        return initial_state


    def setup_cunti(self,n_bins, low, up):
        self.encoder = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="uniform", random_state=0
        )
        self.encoder.fit(np.linspace([low], [up], 100))
        




    def voc(self, soc):
        soc = np.clip(soc, 0, 1)
        a = (self.parameters["v0"] - self.parameters["vL"]) * np.exp(
            self.parameters["gamma"] * (soc - 1)
        )
        b = self.parameters["alpha"] * self.parameters["vL"] * (soc - 1)
        c = (
            (1 - self.parameters["alpha"])
            * self.parameters["vL"]
            * (
                np.exp(-1 * self.parameters["beta"])
                - np.exp(-1 * self.parameters["beta"] * np.sqrt(soc))
            )
        )
        voc = self.parameters["vL"] + a + b + c
        # print(encoder.inverse_transform(encoder.transform(voc.reshape(-1, 1)))[:,0].shape)
        return self.encoder.inverse_transform(
            self.encoder.transform(voc.reshape(-1, 1))
        )[:, 0]


    def extended_sigmoid_cubic(self, x, L, k, x0, a, b, c):
        return (L / (1 + np.exp(-k * (x - x0)))) + (a * x + b) ** 3 + c

    def fit_inverse(self, plot=False):
        soc = np.linspace(0, 1, 10000)
        voc = self.voc(soc)
        self.inverse_params, _ = curve_fit(
            self.extended_sigmoid_cubic,
            voc,
            soc,
            p0=[1, 1, voc.mean(), 0, 1, 0],
            maxfev=1500000,
        )

        if plot:
            plt.figure(figsize=(15, 5))
            plt.plot(voc, soc, label="gt")
            plt.plot(
                voc,
                self.extended_sigmoid_cubic(voc, *self.inverse_params),
                label="estim",
            )
            plt.xlabel("Voc")
            plt.ylabel("SoC")
            plt.legend()
            plt.title("Inversa estimada de Voc(SoC)")
            plt.suptitle(
                f"Mae: {np.mean(abs(self.extended_sigmoid_cubic(voc, *self.inverse_params)-soc)):.3f}"
            )
            plt.plot()

    def voc_inv(self, voc):
        try:
            return np.clip(self.extended_sigmoid_cubic(voc, *self.inverse_params), 0, 1)
        except:
            assert (
                False
            ), "Hubo un error con la inversa....\nHint: Asegurate de haber fiteado los parámetros con self.fit_inverse()"

    # def calculate_R(self, soc_op, Ck):
    #     # Cálculo original de la resistencia en función del SoC
    #     A = self.parameters["A_Rint"]
    #     B = self.parameters["B_Rint"]
    #     C = self.parameters["C_Rint"]
    #     D = self.parameters["D_Rint"]
    #     factor = self.parameters["Factor_Rint"]
    #     bias = self.parameters["Bias_Rint"]

    #     R_soc = factor * (A * soc_op**3 + B * soc_op**2 + C * soc_op + D) + bias

    #     soh = Ck / self.parameters["Qmax"]

    #     F0, F1, F2, F3 = (self.parameters[f"Factor_R_SOH{i}"] for i in range(4))

    #     derivative_soh = F1 + 2 * F2 * soh + 3 * F3 * soh**2

    #     R_adjusted = R_soc * (1 - derivative_soh)

    #     return R_adjusted

    def calculate_R(self, soc_op):
        A = self.parameters["A_Rint"]
        B = self.parameters["B_Rint"]
        C = self.parameters["C_Rint"]
        D = self.parameters["D_Rint"]
        factor = self.parameters["Factor_Rint"]
        bias = self.parameters["Bias_Rint"]

        R = factor * (A * soc_op**3 + B * soc_op**2 + C * soc_op + D) + bias
        return R

    def output(self, x):
        # print('b')
        voc = self.voc(x["soc"])
        # noise = rnd.normal(0, self.parameters["measurement_noise"]["v"])
        # print(voc)
        v_clean = voc + self.parameters["Rint"] * x["i_"]

        return self.OutputContainer(
            {
                "v": self.encoder.inverse_transform(
                    self.encoder.transform(np.array([v_clean]).reshape(1, -1))
                ),
                "v_": v_clean,
            }
        )

    def next_state(self, x, u, dt):
        soc_ = x["soc"] + u["i"] * dt / (x["Ck"] * 3600)
        i_barra = u["i"]

        # Acumular el cambio en SoC
        delta_soc = u["i"] * dt / (x["Ck"] * 3600)
        SoCacumk = x["SoCacumk"] - delta_soc if u["i"] < 0 else x["SoCacumk"]

        # Comprobar si se ha completado un ciclo
        if x["SoCacumk"] >= 1:
            # Calcular eta al final de un ciclo
            etak = self.calculate_eta()
            # Actualizar capacidad
            Ck = self.calculate_Cap(etak, x["Ck"])
            # Resetear SoCacumk y memoria_soc para el próximo ciclo
            SoCacumk = 0
            self.memoria_soc = []
        else:
            # Continuar acumulando SoC en memoria_soc durante el ciclo
            self.memoria_soc.append(soc_)
            # Mantener los valores actuales de eta y capacidad
            etak = x["etak"]
            Ck = x["Ck"]

        return self.StateContainer(
            {"soc": soc_, "Ck": Ck, "i_": i_barra, "etak": etak, "SoCacumk": SoCacumk}
        )

    def calculate_Cap(self, etak, Cap_prev):
        Ck = etak * Cap_prev
        return Ck

    def setup_knn(self):
        # Preparar los datos para el modelo k-NN
        X = []
        y = []

        for sr, factors in self.parameters["degradation_data"].items():
            sr_range = [float(x) for x in sr.split("-")]
            sr_numeric = sr_range[0] - sr_range[1]  # SR
            asr_numeric = sum(sr_range) / 2  # ASR
            for degradation_percentage, factor in zip([0.7, 0.8, 0.85], factors):
                X.append([asr_numeric, sr_numeric, degradation_percentage])
                y.append(factor)

        # Convertir a numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Entrenar el modelo k-NN
        self.knn = KNeighborsRegressor(n_neighbors=3, weights="distance")
        self.knn.fit(X, y)  # Entrenar el modelo k-NN

    def calculate_eta(self):
        # Calcular eta basado en la fórmula
        eta = (self.parameters["degradation_percentage"]) ** (
            1 / self.parameters["life_cycles"]
        )

        # Multiplicar eta por el factor de KNN
        knn_factor = self.calculate_knn_factor()
        etak = eta * knn_factor

        return etak

    def calculate_knn_factor(self):
        max_soc = max(self.memoria_soc) if self.memoria_soc else 1
        min_soc = min(self.memoria_soc) if self.memoria_soc else 0
        sr = max_soc - min_soc
        asr = (max_soc + min_soc) / 2

        # Utilizar el modelo k-NN para predecir el factor
        factor_predicted = self.knn.predict(
            np.array([[asr, sr, self.parameters["degradation_percentage"]]])
        )
        return factor_predicted[0]

    def event_state(self, x):
        return {"EOD": x["soc"]}

    def threshold_met(self, x):
        condition = x["soc"] < self.eod_thresh
        return {"EOD": condition}