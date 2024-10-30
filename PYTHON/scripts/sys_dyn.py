import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import minimize
import copy
from scipy.interpolate import interp1d

# Clase base para los elementos del diagrama (Nivel, Flujo, Retardo, VariableExogena)
class ElementoDiagrama:
    def __init__(self, nombre):
        self.nombre = nombre

    def actualizar(self, delta_t):
        pass

# Clase para los niveles (almacenan estado y son variables endógenas)
class Nivel(ElementoDiagrama):
    def __init__(self, nombre, valor_inicial=0):
        super().__init__(nombre)
        self.valor_inicial = valor_inicial
        self.valor = valor_inicial

    def actualizar(self, delta_t, flujos_entrantes, flujos_salientes):
        neto_flujos = sum(flujos_entrantes) - sum(flujos_salientes)
        self.valor += neto_flujos * delta_t

# Clase para los flujos (conectan niveles y permiten definir ecuaciones personalizadas)
class Flujo(ElementoDiagrama):
    def __init__(self, nombre, tasa_funcion):
        super().__init__(nombre)
        self.tasa_funcion = tasa_funcion  # Función general para definir cualquier tasa
    
    def calcular(self, niveles, parametros, variables_exogenas):
        return self.tasa_funcion(niveles, parametros, variables_exogenas)

# Clase para el retardo
class Retardo(ElementoDiagrama):
    def __init__(self, nombre, tiempo_retardo_func, flujo_entrante):
        super().__init__(nombre)
        self.tiempo_retardo_func = tiempo_retardo_func  # Guardar la función que calcula el tiempo de retardo
        self.flujo_entrante = flujo_entrante
        self.cola_retardo = deque()  # Inicializar sin longitud máxima, se ajustará dinámicamente
    
    def calcular(self, flujo_actual, niveles, parametros, variables_exogenas):
        # Evaluar el tiempo de retardo usando la función
        tiempo_retardo = int(self.tiempo_retardo_func(niveles, parametros, variables_exogenas))
        # Asegurar que la cola de retardo tenga la longitud correcta
        if len(self.cola_retardo) >= tiempo_retardo:
            self.cola_retardo.popleft()  # Eliminar el valor más antiguo si se cumple el tiempo de retardo
        self.cola_retardo.append(flujo_actual)
        # Devuelve el flujo con retardo, o 0 si aún no se cumple el tiempo de retardo
        if len(self.cola_retardo) >= tiempo_retardo:
            return self.cola_retardo[0]
        else:
            return 0  # Aún no se cumple el tiempo de retardo

# Clase para variables exógenas (su valor se define fuera del sistema)
class VariableExogena(ElementoDiagrama):
    def __init__(self, nombre, valor_inicial=0):
        super().__init__(nombre)
        self.valor = valor_inicial

    def actualizar(self, valor_nuevo):
        self.valor = valor_nuevo

# Clase del diagrama que coordina todo el sistema
class Diagrama:
    """
    Clase Diagrama para modelar y simular sistemas dinámicos.
    Atributos:
    ----------
    elementos : dict
        Almacena todos los niveles, flujos, retardos y variables exógenas.
    flujos : list
        Lista específica para flujos.
    niveles : list
        Lista específica para niveles.
    retardos : list
        Lista específica para retardos.
    variables_exogenas : dict
        Lista de variables exógenas.
    conexiones : dict
        Conexiones entre flujos y niveles.
    resultados : dict
        Almacena resultados de simulación.
    tiempo : float
        Tiempo actual de la simulación.
    delta_t : float
        Intervalo de tiempo para la simulación.
    Métodos:
    --------
    __init__(self, delta_t=0.1):
        Inicializa la clase Diagrama con un delta_t opcional.
    __str__(self):
        Retorna una representación en cadena del diagrama.
    copiar(self):
        Realiza una copia profunda del diagrama.
    set_delta_t(self, delta_t):
        Establece el intervalo de tiempo delta_t.
    agregar_elemento(self, elemento):
        Agrega un elemento al diagrama.
    agregar_conexiones(self, conexiones):
        Agrega conexiones entre flujos y niveles.
    resetear_resultados(self):
        Resetea los resultados de la simulación y reinicia los niveles.
    simular(self, tiempo_total, parametros, reset_resultados=True):
        Ejecuta la simulación del diagrama durante un tiempo total dado.
    graficar(self):
        Grafica los resultados de la simulación.
    obtener_resultados(self):
        Obtiene los resultados de la simulación en formato de lista.
    calcular_error_base(self, datos_observados, parametros, escala_func):
        Calcula el error entre los datos observados y simulados en puntos de tiempo específicos.
    calcular_error_nrmse(self, datos_observados, parametros):
        Calcula el error NRMSE entre los datos observados y simulados.
    calcular_error_mape(self, datos_observados, parametros):
        Calcula el error MAPE entre los datos observados y simulados.
    calcular_error_mse_escalado(self, datos_observados, parametros):
        Calcula el error MSE escalado entre los datos observados y simulados.
    ajustar_parametros(self, datos_observados, parametros_iniciales, convertir_parametros_func, method='L-BFGS-B', limites=None, error_metric='nrmse', verbose=True):
        Ajusta los parámetros del modelo para minimizar el error entre los datos observados y simulados.
    """
    def __init__(self, delta_t=0.1):
        self.elementos = {}  # Almacena todos los niveles, flujos, retardos y variables exógenas
        self.flujos = []     # Lista específica para flujos
        self.niveles = []    # Lista específica para niveles
        self.retardos = []   # Lista específica para retardos
        self.variables_exogenas = {}  # Lista de variables exógenas
        self.conexiones = {}  # Conexiones entre flujos y niveles
        self.resultados = {} # Almacena resultados de simulación
        self.tiempo = 0
        self.delta_t = delta_t

    def __str__(self):
        return f"Diagrama con {len(self.elementos)} elementos"
    
    # Método de copia profunda
    def copiar(self):
        return copy.deepcopy(self)
    
    def set_delta_t(self, delta_t):
        self.delta_t = delta_t  
    
    def agregar_elemento(self, elemento):
        self.elementos[elemento.nombre] = elemento
        
        if isinstance(elemento, Flujo):
            self.flujos.append(elemento)
        elif isinstance(elemento, Nivel):
            self.niveles.append(elemento)
            self.resultados[elemento.nombre] = []
        elif isinstance(elemento, Retardo):
            self.retardos.append(elemento)
        elif isinstance(elemento, VariableExogena):
            self.variables_exogenas[elemento.nombre] = elemento

    def agregar_conexiones(self, conexiones):
        self.conexiones = conexiones

    def resetear_resultados(self):
        self.resultados = {nivel.nombre: [] for nivel in self.niveles}
        self.tiempo = 0
        # Reiniciar los valores de los niveles
        for nivel in self.niveles:
            nivel.valor = nivel.valor_inicial

    def simular(self, tiempo_total, parametros, reset_resultados=True):

        if reset_resultados:
            self.resetear_resultados()

        for t in np.arange(0, tiempo_total, self.delta_t):
            flujos_calculados = {}

            # Calcular todos los flujos
            for flujo in self.flujos:
                flujos_calculados[flujo.nombre] = flujo.calcular(self.elementos, parametros, self.variables_exogenas)
            
            # Procesar retardos
            for retardo in self.retardos:
                flujo_actual = flujos_calculados.get(retardo.flujo_entrante, 0)  # Asegurarse de obtener el flujo, con valor por defecto 0
                flujos_calculados[retardo.nombre] = retardo.calcular(flujo_actual, self.elementos, parametros, self.variables_exogenas)
            
            # Actualizar niveles
            for nivel in self.niveles:
                # Identificar flujos que afectan a este nivel
                flujos_entrantes = [flujos_calculados.get(flujo, 0) for flujo in self.conexiones['flujos_entrantes'].get(nivel.nombre, [])]
                flujos_salientes = [flujos_calculados.get(flujo, 0) for flujo in self.conexiones['flujos_salientes'].get(nivel.nombre, [])]
                nivel.actualizar(self.delta_t, flujos_entrantes, flujos_salientes)
                # Guardar el estado del nivel
                self.resultados[nivel.nombre].append(nivel.valor)

            # Actualizar el tiempo
            self.tiempo += self.delta_t
    
    def graficar(self):
        for nivel in self.resultados:
            plt.plot(self.delta_t*np.arange(0, len(self.resultados[nivel])), self.resultados[nivel], label=nivel)
        plt.xlabel("Tiempo")
        plt.ylabel("Valores de los niveles")
        plt.legend()
        plt.show()

    # Método para obtener los resultados de la simulación en formato de lista
    def obtener_resultados(self):
        return self.resultados
    
    # def calcular_error_base(self, datos_observados, parametros, escala_func):
    #     """Método base para calcular el error entre los datos observados y simulados, con una escala definida."""
    #     tiempo_total = len(next(iter(datos_observados.values()))) * self.delta_t
    #     self.simular(tiempo_total, parametros=parametros)
        
    #     resultados_simulados = self.obtener_resultados()
    #     error = 0

    #     for nivel, observados in datos_observados.items():
    #         simulados = resultados_simulados[nivel][:len(observados)]
    #         escala = escala_func(observados)
            
    #         # Para evitar divisiones por cero en cada elemento de escala
    #         escala = np.where(escala == 0, 1, escala)
    #         error += np.mean(((np.array(simulados) - np.array(observados)) / escala) ** 2)


    #     return np.sqrt(error)
    
    def calcular_error_base(self, datos_observados, parametros, escala_func):
        """Método base para calcular el error entre los datos observados y simulados en puntos de tiempo específicos."""
        # Ejecutar la simulación para obtener los datos simulados completos
        tiempo_total = max(time for time, _ in datos_observados[list(datos_observados.keys())[0]])
        self.simular(tiempo_total, parametros=parametros)
        
        resultados_simulados = self.obtener_resultados()
        error = 0

        # Crear una lista de tiempos simulados basada en delta_t
        tiempos_simulados = np.arange(0, tiempo_total, self.delta_t)

        for nivel, observados in datos_observados.items():
            # Extraer tiempos y valores observados
            tiempos_observados, valores_observados = zip(*observados)

            # Interpolación de los valores simulados en los tiempos observados
            valores_simulados = interp1d(tiempos_simulados, resultados_simulados[nivel], kind='linear', fill_value="extrapolate")(tiempos_observados)
            
            # Calcular el error entre los valores observados y los valores simulados interpolados
            escala = escala_func(valores_observados)
            escala = np.where(escala == 0, 1, escala)  # Evitar división por cero
            error += np.mean(((np.array(valores_simulados) - np.array(valores_observados)) / escala) ** 2)

        return np.sqrt(error)

    def calcular_error_nrmse(self, datos_observados, parametros):
        # Función de escala para NRMSE basada en la amplitud de la serie observada
        return self.calcular_error_base(datos_observados, parametros, lambda obs: max(obs) - min(obs))

    def calcular_error_mape(self, datos_observados, parametros):
        # Función de escala para MAPE basada en el valor observado (evita división por 0 usando np.where)
        return self.calcular_error_base(datos_observados, parametros, lambda obs: np.where(obs == 0, 1, obs))

    def calcular_error_mse_escalado(self, datos_observados, parametros):
        # Función de escala para MSE Escalado basada en la desviación estándar de la serie observada
        return self.calcular_error_base(datos_observados, parametros, np.std)

    def ajustar_parametros(self, datos_observados, parametros_iniciales, convertir_parametros_func, method='L-BFGS-B', limites=None, error_metric='nrmse', verbose=True):

        if method not in ['L-BFGS-B', 'Nelder-Mead', 'Powell']:
            raise ValueError("El argumento 'method' solo puede tomar los valores: 'L-BFGS-B', 'Nelder-Mead', 'Powell'")
        
        if verbose:
            print(f"Ajustando parámetros con método: {method} y métrica de error: {error_metric}")
        
        # Selección de la función de error según el parámetro error_metric
        if error_metric == 'nrmse':
            error_func = self.calcular_error_nrmse
        elif error_metric == 'mape':
            error_func = self.calcular_error_mape
        elif error_metric == 'mse_escalado':
            error_func = self.calcular_error_mse_escalado
        else:
            raise ValueError("Métrica de error desconocida.")
        
        # Definición de la función de error
        def wrapper_error_func(parametros_array):
            parametros_dict = convertir_parametros_func(parametros_array)
            return error_func(datos_observados, parametros_dict)

        # Aplicación de límites con métodos compatibles
        if method != 'Nelder-Mead':
            resultado = minimize(wrapper_error_func, parametros_iniciales, method=method, bounds=limites)
        else:
            def restringir_parametros(parametros_array, limites):
                for i, (low, high) in enumerate(limites):
                    if parametros_array[i] < low or parametros_array[i] > high:
                        return float('inf')
                return None
            
            def wrapper_error_func_con_restriccion(parametros_array):
                penalizacion = restringir_parametros(parametros_array, limites)
                if penalizacion is not None:
                    return penalizacion
                return wrapper_error_func(parametros_array)
            
            resultado = minimize(wrapper_error_func_con_restriccion, parametros_iniciales, method=method)
        
        return resultado
    
    # # Método para calcular el error entre los datos observados y los simulados
    # def calcular_error(self, datos_observados, parametros):

    #     # print(f"Calculando error con parámetros: alpha = {parametros['alpha']}, beta = {parametros['beta']}, gamma = {parametros['gamma']}, delta = {parametros['delta']}")
        
    #     tiempo_total = len(next(iter(datos_observados.values()))) * self.delta_t  # Ajustar el tiempo total a los datos observados
    #     self.simular(tiempo_total, parametros=parametros)
        
    #     resultados_simulados = self.obtener_resultados()
    #     error = 0

    #     # print(f"Valores medios resultados simulados: presa ({np.mean(resultados_simulados['presa'])}), depredador ({np.mean(resultados_simulados['depredador'])})")
        
    #     for nivel, observados in datos_observados.items():
    #         simulados = resultados_simulados[nivel][:len(observados)]  # Tomar solo los puntos simulados correspondientes
    #         error += np.square(np.sum((np.array(simulados) - np.array(observados))))

    #     # print(f"Error total: {error}\n------------------------------------------------------------")
        
    #     return error

    # # Método para ajustar los parámetros
    # def ajustar_parametros(self, datos_observados, parametros_iniciales, convertir_parametros_func, method='L-BFGS-B', limites=None, verbose=True):
    #     # Verificar que el método proporcionado sea uno de los permitidos
    #     if method not in ['L-BFGS-B', 'Nelder-Mead', 'Powell']:
    #         raise ValueError("El argumento 'method' solo puede tomar los valores: 'L-BFGS-B', 'Nelder-Mead', 'Powell'")
        
    #     if verbose:
    #         print(f"Ajustando parámetros con método: {method}")
        
    #     if method != 'Nelder-Mead':
    #         # Definir función de error 
    #         def error_func(parametros_array):
    #             # Convertir el array de parámetros en un diccionario usando la función proporcionada
    #             parametros_dict = convertir_parametros_func(parametros_array)
    #             return self.calcular_error(datos_observados, parametros_dict)
            
    #         # Minimizar el error ajustando los parámetros
    #         resultado = minimize(error_func, parametros_iniciales, method=method, bounds=limites)
    #     else:
    #         # Definir la función para restringir los parámetros manualmente dentro de ciertos límites
    #         def restringir_parametros(parametros_array, limites):
    #             for i, (low, high) in enumerate(limites):
    #                 if parametros_array[i] < low or parametros_array[i] > high:
    #                     return float('inf')  # Devolver un valor alto para rechazar parámetros fuera de rango
    #             return None
            
    #         # Definir función de error 
    #         def error_func(parametros_array):
    #             # Aplicar restricciones manuales
    #             penalizacion = restringir_parametros(parametros_array, limites)
    #             if penalizacion is not None:
    #                 return penalizacion

    #             # Convertir los parámetros y calcular el error como antes
    #             parametros_dict = convertir_parametros_func(parametros_array)
    #             return self.calcular_error(datos_observados, parametros_dict)
            
    #         # Minimizar el error ajustando los parámetros con límites
    #         resultado = minimize(error_func, parametros_iniciales, method=method)
            
    #     return resultado
