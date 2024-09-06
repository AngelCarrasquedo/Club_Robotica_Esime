import tkinter as tk
from tkinter import messagebox
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt

class NeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        root.title("Red Neuronal")

        # Crear widgets
        self.create_widgets()

        # Inicializar variable de salida
        self.output = None
        self.weights = None

    def create_widgets(self):
        # Etiquetas y entradas para datos
        tk.Label(self.root, text="Entrada 1:").grid(row=0, column=0)
        self.entry_X1 = tk.Entry(self.root)
        self.entry_X1.grid(row=0, column=1)

        tk.Label(self.root, text="Entrada 2:").grid(row=1, column=0)
        self.entry_X2 = tk.Entry(self.root)
        self.entry_X2.grid(row=1, column=1)

        tk.Label(self.root, text="Tasa de Aprendizaje:").grid(row=2, column=0)
        self.entry_learning_rate = tk.Entry(self.root)
        self.entry_learning_rate.grid(row=2, column=1)

        tk.Label(self.root, text="Número de Iteraciones:").grid(row=3, column=0)
        self.entry_num_iterations = tk.Entry(self.root)
        self.entry_num_iterations.grid(row=3, column=1)

        tk.Label(self.root, text="Valor Deseado para la Salida:").grid(row=4, column=0)
        self.entry_desired_output = tk.Entry(self.root)
        self.entry_desired_output.grid(row=4, column=1)

        # Botones
        tk.Button(self.root, text="Entrenar", command=self.train_network).grid(row=5, column=0)
        tk.Button(self.root, text="Borrar", command=self.clear_all).grid(row=5, column=1)
        tk.Button(self.root, text="Exportar PDF", command=self.export_pdf).grid(row=5, column=2)
        tk.Button(self.root, text="Graficar", command=self.plot_graph).grid(row=5, column=3)

    def train_network(self):
        try:
            X1 = float(self.entry_X1.get())
            X2 = float(self.entry_X2.get())
            learning_rate = float(self.entry_learning_rate.get())
            num_iterations = int(self.entry_num_iterations.get())
            desired_output = float(self.entry_desired_output.get())
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingresa valores válidos.")
            return

        X = np.array([X1, X2])
        desired_output = np.array([desired_output])
        
        # Inicialización de pesos
        self.weights = np.random.rand(6)  # Inicializamos pesos aleatoriamente
        
        for _ in range(num_iterations):
            # Propagación hacia adelante
            X3 = (X[0] * self.weights[0]) + (X[1] * self.weights[1])
            X4 = (X[0] * self.weights[2]) + (X[1] * self.weights[3])
            X5 = (X3 * self.weights[4]) + (X4 * self.weights[5])
            
            # Error
            error = desired_output - X5

            # Gradientes
            d_X5 = error
            d_W35 = X3 * d_X5
            d_W45 = X4 * d_X5
            d_X3 = self.weights[4] * d_X5
            d_X4 = self.weights[5] * d_X5
            d_W13 = X[0] * d_X3
            d_W23 = X[1] * d_X3
            d_W14 = X[0] * d_X4
            d_W24 = X[1] * d_X4
            
            # Actualización de pesos
            self.weights[4] += float(learning_rate * d_W35)
            self.weights[5] += float(learning_rate * d_W45)
            self.weights[0] += float(learning_rate * d_W13)
            self.weights[1] += float(learning_rate * d_W23)
            self.weights[2] += float(learning_rate * d_W14)
            self.weights[3] += float(learning_rate * d_W24)
        
        # Calcular la salida con los pesos entrenados
        self.output = self.calculate_output(X, self.weights)
        self.show_results(self.weights, self.output)

    def calculate_output(self, X, weights):
        X3 = (X[0] * weights[0]) + (X[1] * weights[1])
        X4 = (X[0] * weights[2]) + (X[1] * weights[3])
        X5 = (X3 * weights[4]) + (X4 * weights[5])
        return X5

    def show_results(self, weights, output):
        result_text = (
            f"Pesos Finales:\n"
            f"W13 = {weights[0]}\n"
            f"W23 = {weights[1]}\n"
            f"W14 = {weights[2]}\n"
            f"W24 = {weights[3]}\n"
            f"W35 = {weights[4]}\n"
            f"W45 = {weights[5]}\n"
            f"Salida Final: {output}"
        )
        messagebox.showinfo("Resultados", result_text)

    def clear_all(self):
        self.entry_X1.delete(0, tk.END)
        self.entry_X2.delete(0, tk.END)
        self.entry_learning_rate.delete(0, tk.END)
        self.entry_num_iterations.delete(0, tk.END)
        self.entry_desired_output.delete(0, tk.END)
        self.output = None
        self.weights = None
        plt.close('all')  # Cierra cualquier gráfico abierto

    def export_pdf(self):
        if self.output is None or self.weights is None:
            messagebox.showwarning("Exportar PDF", "Primero entrene la red neuronal.")
            return

        X1 = self.entry_X1.get()
        X2 = self.entry_X2.get()
        learning_rate = self.entry_learning_rate.get()
        num_iterations = self.entry_num_iterations.get()
        desired_output = self.entry_desired_output.get()

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.cell(200, 10, txt="Resultados de Entrenamiento de Red Neuronal", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Entrada 1: {X1}", ln=True)
        pdf.cell(200, 10, txt=f"Entrada 2: {X2}", ln=True)
        pdf.cell(200, 10, txt=f"Tasa de Aprendizaje: {learning_rate}", ln=True)
        pdf.cell(200, 10, txt=f"Número de Iteraciones: {num_iterations}", ln=True)
        pdf.cell(200, 10, txt=f"Valor Deseado: {desired_output}", ln=True)
        pdf.ln(10)

        # Exportar pesos como listas
        pdf.cell(200, 10, txt="Pesos Finales:", ln=True)
        pdf.cell(200, 10, txt=f"W13 = {self.weights[0]}", ln=True)
        pdf.cell(200, 10, txt=f"W23 = {self.weights[1]}", ln=True)
        pdf.cell(200, 10, txt=f"W14 = {self.weights[2]}", ln=True)
        pdf.cell(200, 10, txt=f"W24 = {self.weights[3]}", ln=True)
        pdf.cell(200, 10, txt=f"W35 = {self.weights[4]}", ln=True)
        pdf.cell(200, 10, txt=f"W45 = {self.weights[5]}", ln=True)
        pdf.cell(200, 10, txt=f"Salida Final: {self.output}", ln=True)
        
        pdf.output("resultados.pdf")
        messagebox.showinfo("Exportar PDF", "Los resultados se han exportado a 'resultados.pdf'.")

    def plot_graph(self):
        if self.output is None:
            messagebox.showwarning("Graficar", "Primero entrene la red neuronal.")
            return

        # Graficar la salida final
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, self.output], marker='o', linestyle='-', color='b')
        plt.title('Salida Final de la Red Neuronal')
        plt.xlabel('Iteración')
        plt.ylabel('Salida')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()





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
 
      
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡿⠉⠀⠉⢿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⢀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣄⣀⣶⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠉⣯⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⠀⢿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⠀⠀⠘⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠼⠯⠤⠤⠤⠿⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⣀⡬⠥⢥⣀⠀⠀⠀⠀⠀⢀⡠⠾⠷⢤⡀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⣾⡇⠀⠀⠀⠀⢀⡾⣋⣄⠀⠈⠉⢶⠀⠀⠀⣴⡟⣁⣀⠀⠀⠙⣆⠀⠀⠀⠀⠀⢸⣅⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⠉⠉⢹⡇⠀⠀⠀⠀⢼⣿⣿⣿⣿⣦⡀⢈⡇⠀⠀⣿⣿⣿⣿⣿⡄⠀⣿⠀⠀⠀⠀⠀⢸⡏⠉⢻⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⢸⡇⠀⠀⠀⠀⠈⢿⣿⣿⣿⠿⢀⡾⠁⠀⠀⠘⣿⣿⣿⡿⠁⣠⠏⠀⠀⠀⠀⠀⢸⡇⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢀⣴⠾⠷⠾⢶⡄⠀⠀⠀⠉⠛⠷⠒⠋⠀⠀⠀⠀⠀⠈⠙⠛⠒⠊⠁⠀⠀⠀⣠⡶⠶⠾⠷⣦⢨⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢼⣾⣿⠀⠀⠀⠀⠙⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⠀⠀⢠⣿⣸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢈⣿⠀⠀⠀⠀⠀⠀⠛⣦⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⠞⠋⠀⠀⠀⠀⠀⠘⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠋⠛⢻⣿⣿⡛⠛⠛⠛⠛⢻⣿⣿⣽⣭⣭⣭⣭⣭⣭⣿⣟⠋⠟⠛⢛⣿⣿⡟⠉⠉⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⣤⣭⣿⣤⣤⣤⣼⣯⣤⣴⣤⣤⣦⣤⣤⣤⣤⣿⣦⣤⣾⣯⣥⣆⠉⠛⢦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡏⠀⠀⠀⣿⣠⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⢿⡷⢶⡄⣿⠀⠀⠘⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⡇⠀⠀⠀⣿⣼⠀⠀⠀⢲⡆⠀⠀⠀⠀⠀⠀⠀⠀⣤⠄⠀⠀⠀⠀⢬⡇⢸⠀⠀⠀⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⠏⢳⣄⠀⢀⣿⣿⠀⠀⢠⡾⡇⠀⡀⠀⠀⠀⠀⠀⢰⣿⠀⠀⠀⠀⠀⣿⣿⣼⠀⢀⡼⠛⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⠃⠀⠀⠉⢻⣿⣿⣿⠀⠀⣼⠇⢿⣴⣷⠀⠀⠀⠀⣠⣾⣿⣄⣿⡄⠀⠀⢹⣿⣿⣟⠋⠀⠀⠈⢷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣇⠀⠀⠀⢰⡟⠉⣿⢻⠷⠶⠿⠀⠸⠏⠘⣦⠀⡿⠒⠛⠃⢸⣿⡟⠿⠿⠷⠿⡟⣿⠹⣆⠀⠀⠀⢈⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡿⠉⠓⢶⣴⠟⠁⠀⣿⣼⠀⠀⠀⠀⠀⠀⠀⠹⣾⠁⠀⠀⠀⠈⢿⠀⠀⠀⠀⢀⡇⣿⠀⠘⣷⣴⠖⠉⠙⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣤⡀⢠⡾⠃⠀⠀⠀⣿⢹⠀⠀⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢨⣧⣿⠀⠀⠈⢻⣄⠀⣠⣼⣧⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⣴⡏⠀⠀⠈⠻⣿⠁⠀⠀⠀⢀⣿⠘⢿⣿⡿⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⢿⣿⠟⠫⣽⠀⠀⠀⠀⣹⡟⠁⠀⠀⠙⢷⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⣴⡾⠿⠟⢻⡄⠀⠀⠀⢰⡿⠀⠀⠀⠀⠘⣿⣶⡾⠉⢻⣤⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⣠⡞⠉⠳⣶⣿⠀⠀⠀⠀⠈⣇⠀⠀⠀⠀⣸⡟⠲⠶⣦⡀⠀⠀⠀
⠀⢀⡾⠋⠀⠀⠀⠈⠳⣦⣤⡴⠟⠀⠀⠀⠀⠀⢈⣿⠈⢷⣤⡾⠏⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉⢿⣷⡾⠛⣿⠀⠀⠀⠀⠀⠙⢦⣤⣤⠴⠋⠁⠀⠀⠀⠻⣄⠀⠀
⣤⡟⠀⠀⣤⢦⣀⠀⠀⠀⢻⠀⠀⠀⠀⠀⠀⠀⠸⣿⣄⣤⣷⣤⣤⣤⣴⣼⣧⣶⣴⣦⣦⣤⣤⣤⣤⣤⣤⣦⣤⣠⣿⠀⠀⠀⠀⠀⠀⠀⣼⠁⠀⠀⢀⣴⢦⡀⠀⠹⣦⠀
⣿⡀⢠⠞⠁⢀⣽⠇⠀⣠⣾⠃⠀⠀⠀⠀⠀⠀⠀⠀⣾⠋⠀⠀⠀⠀⠀⠀⠈⣿⡀⠀⣾⡇⠀⠀⠀⠀⠀⠀⠀⣻⡄⠀⠀⠀⠀⠀⠀⠀⠻⣄⠀⠀⢿⡀⠀⠻⣄⠀⣿⡃
⠈⣷⠏⠀⣠⡿⠁⠀⣰⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣷⠤⠤⠤⠤⠤⠤⢾⣟⠁⠀⢈⡷⠦⠦⠤⠤⠤⠤⠶⣏⠁⠀⠀⠀⠀⠀⠀⠀⠀⠙⣧⠀⠀⠹⣆⠀⠈⣷⠏⠀
⠀⠀⠀⢴⣯⣤⣤⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⡁⠀⠀⠀⠀⠀⠀⠀⣽⠶⠚⢿⡅⠀⠀⠀⠀⠀⠀⠀⣽⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⢦⣤⣬⣷⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠟⠛⠛⠛⠛⠛⠛⠻⢯⣄⠀⣤⠟⠛⠛⠛⠛⠛⠛⠻⣧⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣄⣀⣀⣀⣠⣤⣤⣤⣿⠟⠈⠻⣦⣀⣀⣀⣀⣀⣀⣠⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⠯⠿⠿⠭⢯⣉⣉⣉⣿⡀⠠⣾⣏⣀⣀⡠⠶⠶⠿⠿⠷⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⠞⠋⠁⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⠀⠀⣿⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣄⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣟⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣸⡷⠀⣿⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣈⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠀⠀⠈⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠀
 """