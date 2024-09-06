
""" 
      ██████╗ ██████╗ ██████╗ ████████╗ ██████╗      ██████╗██╗██████╗  ██████╗██╗   ██╗██╗████████╗ ██████╗ 
    ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔═══██╗    ██╔════╝██║██╔══██╗██╔════╝██║   ██║██║╚══██╔══╝██╔═══██╗
    ██║     ██║   ██║██████╔╝   ██║   ██║   ██║    ██║     ██║██████╔╝██║     ██║   ██║██║   ██║   ██║   ██║
    ██║     ██║   ██║██╔══██╗   ██║   ██║   ██║    ██║     ██║██╔══██╗██║     ██║   ██║██║   ██║   ██║   ██║
    ╚██████╗╚██████╔╝██║  ██║   ██║   ╚██████╔╝    ╚██████╗██║██║  ██║╚██████╗╚██████╔╝██║   ██║   ╚██████╔╝
     ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝      ╚═════╝╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝   ╚═╝    ╚═════╝ 
 * ============================================================================
 *                      Código Desarrollado por Corto Circuito
 * ============================================================================
 * Desarrollador: Ángel Carrasquedo & johan garcia
 """ 
# Valores de entrada
X1 = 2
X2 = 3

# Pesos
W13 = 7
W23 = 6
W14 = 5
W24 = 4
W35 = 3
W45 = 8

# Cálculo de X3 y X4
X3 = (X1 * W13) + (X2 * W23)  # (2*7) + (3*6)
X4 = (X1 * W14) + (X2 * W24)  # (2*5) + (3*4)

# Cálculo de X5
X5 = (X3 * W35) + (X4 * W45)  # (32*3) + (23*8)

# Imprimir los resultados
print(f"X3 = {X3}")
print(f"X4 = {X4}")
print(f"X5 = {X5}")
