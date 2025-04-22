# nyx_core/modelo.py
'''Modelo de la red neural de la inteligencia artificial NYX
    version : 0.1 
    Autor: Jefferson Arias Gutierrrez
    '''

'''Librerias utilizadas '''
import torch
import torch.nn as nn

# Creamos una red neuronal simple
class MiniRed(nn.Module):
    def __init__(self):
        super(MiniRed, self).__init__()
        self.perceptron = nn.Linear(2, 1)  # 2 entradas, 1 salida

    def forward(self, x):
        return torch.sigmoid(self.perceptron(x))  # función de activación sigmoide

# Creamos una instancia del modelo
modelo = MiniRed()

# Un dato de prueba: dos entradas
entrada = torch.tensor([0.5, 0.8])

# Pasamos los datos por la red
salida = modelo(entrada)

print("Salida:", salida.item())
print("Pesos:", modelo.perceptron.weight)
print("Bias:", modelo.perceptron.bias)
