import numpy as np

# Parámetros de la red
learning_rate = 0.01
num_iterations = 100
desired_output = 15

# Inicialización de los pesos
weights = np.array([-1.12381267, 0.1017876, -1.22593916, -0.50194269, -0.77792243, -1.27609638])

# Función de activación (sigmoide) y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Función para calcular la salida
def calculate_output(X, weights):

    X3 = (X[0] * weights[0]) + (X[1] * weights[1])
    X4 = (X[0] * weights[2]) + (X[1] * weights[3])
    X5 = (X3 * weights[4]) + (X4 * weights[5])
    return X5

# Función de entrenamiento
def train(X, desired_output, learning_rate, num_iterations):
    global weights

    for _ in range(num_iterations):
        # Propagación hacia adelante
        X3 = (X[0] * weights[0]) + (X[1] * weights[1])
        X4 = (X[0] * weights[2]) + (X[1] * weights[3])
        X5 = (X3 * weights[4]) + (X4 * weights[5])
        
        # Error y gradientes
        error = desired_output - X5
        d_X5 = error
        d_W35 = X3 * d_X5
        d_W45 = X4 * d_X5
        d_X3 = weights[4] * d_X5
        d_X4 = weights[5] * d_X5
        d_W13 = X[0] * d_X3
        d_W23 = X[1] * d_X3
        d_W14 = X[0] * d_X4
        d_W24 = X[1] * d_X4
        
        # Actualización de pesos
        weights[4] += learning_rate * d_W35
        weights[5] += learning_rate * d_W45
        weights[0] += learning_rate * d_W13
        weights[1] += learning_rate * d_W23
        weights[2] += learning_rate * d_W14
        weights[3] += learning_rate * d_W24

    return weights

# Valores de entrada y entrenamiento
X = np.array([5, 5])
weights = train(X, desired_output, learning_rate, num_iterations)

# Calcular la salida con los pesos entrenados
output = calculate_output(X, weights)

# Imprimir resultados
print(f"Pesos finales: {weights}")
print(f"Salida final: {output}")
