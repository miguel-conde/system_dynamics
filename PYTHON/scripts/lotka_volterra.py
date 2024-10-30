from sys_dyn import Diagrama, Nivel, Flujo

### DEFINICIÓN DEL MODELO

# Parámetros del modelo
parametros = {
    "alpha": 0.1,    # Tasa de crecimiento de presas
    "beta": 0.002,   # Tasa de depredación
    "gamma": 0.2,    # Tasa de muerte de depredadores
    "delta": 0.0025  # Eficiencia de conversión de presas en depredadores
}

# Definir las funciones de los flujos según el modelo Lotka-Volterra
def tasa_crecimiento_presas(niveles, parametros, variables_exogenas):
    # La cantidad de presas aumenta de modo proporcional a su número
    return parametros["alpha"] * niveles["presa"].valor

def tasa_depredacion(niveles, parametros, variables_exogenas):
    # La cantidad de depredadores aumenta de modo proporcional a la cantidad de encuentros entre las dos especies
    return parametros["beta"] * niveles["presa"].valor * niveles["depredador"].valor

def tasa_crecimiento_depredadores(niveles, parametros, variables_exogenas):
    # La cantidad de depredadores aumenta de modo proporcional a la cantidad de presas encontradas
    return parametros["delta"] * niveles["presa"].valor * niveles["depredador"].valor

def tasa_muerte_depredadores(niveles, parametros, variables_exogenas):
    # La cantidad de depredadores disminuye de modo proporcional a su número
    return parametros["gamma"] * niveles["depredador"].valor

# Crear el diagrama de sistema
diagrama = Diagrama()

# Añadir los niveles para presas y depredadores
diagrama.agregar_elemento(Nivel("presa", valor_inicial=80))       # Población inicial de presas
diagrama.agregar_elemento(Nivel("depredador", valor_inicial=20))   # Población inicial de depredadores

# Añadir los flujos
diagrama.agregar_elemento(Flujo("crecimiento_presas", tasa_crecimiento_presas))
diagrama.agregar_elemento(Flujo("depredacion", tasa_depredacion))
diagrama.agregar_elemento(Flujo("crecimiento_depredadores", tasa_crecimiento_depredadores))
diagrama.agregar_elemento(Flujo("muerte_depredadores", tasa_muerte_depredadores))

# Definir conexiones entre niveles y flujos
conexiones = {
    "flujos_entrantes": {
        "presa": ["crecimiento_presas"],
        "depredador": ["crecimiento_depredadores"]
    },
    "flujos_salientes": {
        "presa": ["depredacion"],
        "depredador": ["muerte_depredadores"]
    }
}

diagrama.agregar_conexiones(conexiones)

diagrama_ajuste = diagrama.copiar()


### SIMULACIÓN

# Ejecutar la simulación
tiempo_total = 100
diagrama.set_delta_t(0.1)
# parametros.update(conexiones)  # Añadir conexiones a los parámetros
diagrama.simular(tiempo_total=tiempo_total, parametros=parametros)

# Graficar los resultados
diagrama.graficar()


### AJUSTE DE PARÁMETROS

datos_observados = diagrama.obtener_resultados()

# Parámetros iniciales para ajustar (alpha, beta, gamma, delta)
# parametros_iniciales = [0.11, 0.0022, 0.21, 0.0027]
# parametros_iniciales = [0.1, 0.001, 0.1, 0.001]
# parametros_iniciales = [0.1, 0.1, 0.1, 0.1]
parametros_iniciales = [0.5, 0.005, 0.5, 0.005]

# Función de conversión específica del modelo Lotka-Volterra
def convertir_parametros_lotka_volterra(parametros_array):
    return {
        "alpha": parametros_array[0],
        "beta": parametros_array[1],
        "gamma": parametros_array[2],
        "delta": parametros_array[3],
    }

# Limites para los parámetros (alpha, beta, gamma, delta)
limites = [(0, 1), (0.001, 0.01), (0, 1), (0.001, 0.01)]

# Llamada a ajustar_parametros con el modelo específico
resultado = diagrama_ajuste.ajustar_parametros(datos_observados, parametros_iniciales, convertir_parametros_lotka_volterra, limites=limites)

# Informa de los detalles de los resultados de la optimización
print(resultado)

# Mostrar los parámetros ajustados
print("Parámetros ajustados:", resultado.x)
