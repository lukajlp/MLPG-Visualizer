import torch
import numpy as np
import pickle
from typing import List, Tuple, Dict, Any, Optional
from .neural_graph import NeuralGraph

class MLPGraph:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 learning_rate: float = 0.01, activation: str = 'ReLU'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_name = activation
        
        # Definir função de ativação
        self.activation = self._get_activation_function(activation)
        
        # Criar grafo neural
        self.graph = NeuralGraph()
        self.graph.build_mlp([input_size] + hidden_sizes + [output_size])
        
        # Funções de perda e otimização
        self.criterion = torch.nn.MSELoss()
        
    def _get_activation_function(self, name: str):
        activations = {
            'ReLU': torch.nn.ReLU(),
            'Sigmoid': torch.nn.Sigmoid(),
            'Tanh': torch.nn.Tanh(),
            'LeakyReLU': torch.nn.LeakyReLU(),
            'ELU': torch.nn.ELU()
        }
        return activations.get(name, torch.nn.ReLU())
    
    def save_model(self, filename: str):
        """Salva o modelo em um arquivo"""
        state = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'activation': self.activation_name,
            'weights': [edge.weight.data for edge in self.graph.edges],
            'biases': [[node.bias.data for node in layer if hasattr(node, 'bias')] 
                      for layer in self.graph.layers.values()]
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load_model(cls, filename: str) -> 'MLPGraph':
        """Carrega um modelo de um arquivo"""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(state['input_size'], state['hidden_sizes'], 
                   state['output_size'], state['learning_rate'],
                   state['activation'])
        
        # Restaurar pesos
        for edge, weight in zip(model.graph.edges, state['weights']):
            edge.weight.data = weight
        
        # Restaurar biases
        layer_idx = 1
        for layer, layer_biases in zip(model.graph.layers.values(), state['biases']):
            for node, bias in zip(layer, layer_biases):
                if hasattr(node, 'bias'):
                    node.bias.data = bias
            layer_idx += 1
        
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass com animação"""
        # Forward pass normal
        output = self.graph.forward(x)
        # Aplicar função de ativação
        return self.activation(output)
    
    def backward(self, gradient: torch.Tensor):
        """Backward pass com animação"""
        # Aplicar derivada da função de ativação
        if isinstance(self.activation, torch.nn.ReLU):
            gradient = gradient * (self.graph.last_output > 0).float()
        elif isinstance(self.activation, torch.nn.LeakyReLU):
            gradient = gradient * torch.where(self.graph.last_output > 0, 
                                           torch.ones_like(self.graph.last_output),
                                           torch.full_like(self.graph.last_output, 0.01))
        elif isinstance(self.activation, torch.nn.Sigmoid):
            sig_out = torch.sigmoid(self.graph.last_output)
            gradient = gradient * sig_out * (1 - sig_out)
        elif isinstance(self.activation, torch.nn.Tanh):
            tanh_out = torch.tanh(self.graph.last_output)
            gradient = gradient * (1 - tanh_out * tanh_out)
        elif isinstance(self.activation, torch.nn.ELU):
            elu_deriv = torch.where(self.graph.last_output > 0,
                                  torch.ones_like(self.graph.last_output),
                                  torch.exp(self.graph.last_output))
            gradient = gradient * elu_deriv
        
        # Backward pass normal
        self.graph.backward(gradient)
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, callback=None, current_epoch=None, total_epochs=None) -> float:
        """Um passo de treinamento com animação
        
        Args:
            x: Dados de entrada (batch_size, input_size)
            y: Alvos (batch_size, output_size)
            callback: Função chamada durante o treinamento para atualizar a visualização
            current_epoch: Época atual do treinamento (para visualização)
            total_epochs: Total de épocas do treinamento (para visualização)
            
        Returns:
            Valor da loss para este batch
        """
        # Forward pass
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        
        # Coletar ativações de todos os nós
        node_activations = {}
        for layer_idx, layer in self.graph.layers.items():
            for node in layer:
                if hasattr(node, 'last_output') and node.last_output is not None:
                    value = node.last_output
                    # Se for um tensor, pegue a média
                    if isinstance(value, torch.Tensor):
                        node_activations[node.id] = value.mean().item()
                    else:
                        node_activations[node.id] = float(value)
        
        if callback:
            callback('forward', outputs, loss.item(), node_activations)
        
        # Backward pass
        self.backward(2 * (outputs - y))  # Derivada do MSE
        
        # Coletar gradientes de todos os nós
        node_gradients = {}
        for layer_idx, layer in self.graph.layers.items():
            for node in layer:
                if hasattr(node, 'grad') and node.grad is not None:
                    value = node.grad
                    # Se for um tensor, pegue a média
                    if isinstance(value, torch.Tensor):
                        node_gradients[node.id] = value.mean().item()
                    else:
                        node_gradients[node.id] = float(value)
        
        if callback:
            callback('backward', outputs, loss.item(), node_gradients)
        
        # Atualizar parâmetros
        weights, biases = self.graph.get_parameters()
        
        # Atualizar pesos
        for edge in self.graph.edges:
            if edge.weight.grad is not None:
                edge.weight -= self.learning_rate * edge.weight.grad
                edge.weight.grad = None
        
        # Atualizar biases
        for layer_idx in range(1, len(self.graph.layers)):
            for node in self.graph.layers[layer_idx]:
                if hasattr(node, 'bias') and node.bias.grad is not None:
                    node.bias -= self.learning_rate * node.bias.grad
                    node.bias.grad = None
        
        return loss.item()
    
    def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int, 
              batch_size: int = 32, verbose: bool = True, callback=None) -> List[float]:
        """
        Treina a rede neural
        
        Args:
            x: Dados de entrada (n_samples, input_size)
            y: Alvos (n_samples, output_size)
            epochs: Número de épocas
            batch_size: Tamanho do batch
            verbose: Se True, mostra progresso do treinamento
            callback: Função chamada a cada época com (época, loss, pesos)
            
        Returns:
            Lista com o valor da perda em cada época
        """
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in dataloader:
                loss = self.train_step(batch_x, batch_y)
                epoch_loss += loss
                num_batches += 1
            
            epoch_loss /= num_batches
            losses.append(epoch_loss)
            
            # Chamar callback se fornecido
            if callback is not None:
                callback(epoch, epoch_loss, self.graph.edges)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        
        return losses
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Faz predições para os dados de entrada
        
        Args:
            x: Dados de entrada (n_samples, input_size)
            
        Returns:
            Predições (n_samples, output_size)
        """
        with torch.no_grad():
            return self.forward(x)
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """
        Avalia o modelo nos dados fornecidos
        
        Args:
            x: Dados de entrada (n_samples, input_size)
            y: Alvos (n_samples, output_size)
            
        Returns:
            Tupla com (loss, accuracy)
        """
        with torch.no_grad():
            outputs = self.predict(x)
            loss = self.criterion(outputs, y).item()
            
            # Para classificação binária
            if self.output_size == 1:
                predictions = (outputs > 0.5).float()
                accuracy = (predictions == y).float().mean().item()
            # Para classificação multiclasse
            else:
                predictions = outputs.argmax(dim=1)
                targets = y.argmax(dim=1)
                accuracy = (predictions == targets).float().mean().item()
                
            return loss, accuracy