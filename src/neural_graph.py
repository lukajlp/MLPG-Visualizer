import torch
import numpy as np
from typing import List, Dict, Optional, Tuple

class Node:
    def __init__(self, id: int, layer: int, activation_fn=torch.nn.ReLU()):
        self.id = id
        self.layer = layer
        self.value = None
        self.gradient = None
        self.activation_fn = activation_fn
        self.incoming_edges: List['Edge'] = []
        self.outgoing_edges: List['Edge'] = []
        self.last_output = None  # Armazenar última ativação
        self.grad = None  # Armazenar último gradiente

    def forward(self) -> torch.Tensor:
        if self.layer == 0:  # Input layer
            return self.value
        
        # Sum weighted inputs from incoming edges
        self.pre_activation = sum(edge.weight * edge.source.forward() for edge in self.incoming_edges)
        if hasattr(self, 'bias'):
            self.pre_activation += self.bias
            
        # Apply activation function and store output
        self.value = self.activation_fn(self.pre_activation)
        if isinstance(self.value, torch.Tensor):
            self.last_output = self.value.clone().detach()  # Armazenar cópia do tensor
        else:
            self.last_output = self.value
        return self.value

    def backward(self, gradient: torch.Tensor = None):
        if gradient is not None:
            self.gradient = gradient
        
        # If this is not an output node, compute gradient from outgoing edges
        if not self.outgoing_edges:
            return
            
        total_gradient = torch.zeros_like(self.value)
        for edge in self.outgoing_edges:
            # Get gradient flowing back from the target node
            edge_gradient = edge.target.gradient
            # Weight it by this edge's weight
            total_gradient += edge.weight * edge_gradient
            # Armazenar cópia do gradiente
            if isinstance(total_gradient, torch.Tensor):
                self.grad = total_gradient.clone().detach()
            else:
                self.grad = total_gradient
            
        # Apply activation function derivative
        if self.value is not None and hasattr(self, 'pre_activation'):
            # Calcular a derivada da função de ativação
            if isinstance(self.activation_fn, torch.nn.ReLU):
                derivative = (self.pre_activation > 0).float()
            elif isinstance(self.activation_fn, torch.nn.LeakyReLU):
                derivative = torch.where(self.pre_activation > 0,
                                      torch.ones_like(self.pre_activation),
                                      torch.full_like(self.pre_activation, 0.01))
            elif isinstance(self.activation_fn, torch.nn.Sigmoid):
                sig = torch.sigmoid(self.pre_activation)
                derivative = sig * (1 - sig)
            elif isinstance(self.activation_fn, torch.nn.Tanh):
                tanh = torch.tanh(self.pre_activation)
                derivative = 1 - tanh * tanh
            elif isinstance(self.activation_fn, torch.nn.ELU):
                derivative = torch.where(self.pre_activation > 0,
                                     torch.ones_like(self.pre_activation),
                                     torch.exp(self.pre_activation))
            else:  # Identity ou qualquer outra
                derivative = torch.ones_like(self.pre_activation)
                
            self.gradient = total_gradient * derivative

class Edge:
    def __init__(self, source: Node, target: Node):
        self.source = source
        self.target = target
        self.weight = torch.randn(1, requires_grad=True)
        
        # Add this edge to the nodes' edge lists
        source.outgoing_edges.append(self)
        target.incoming_edges.append(self)

class NeuralGraph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.layers: Dict[int, List[Node]] = {}
        self.input_size = 0
        self.output_size = 0
        
    def add_node(self, layer: int, activation_fn=torch.nn.ReLU()) -> Node:
        node_id = len(self.nodes)
        node = Node(node_id, layer, activation_fn)
        self.nodes.append(node)
        
        if layer not in self.layers:
            self.layers[layer] = []
        self.layers[layer].append(node)
        
        return node
        
    def add_edge(self, source: Node, target: Node) -> Edge:
        edge = Edge(source, target)
        self.edges.append(edge)
        return edge
        
    def build_mlp(self, layer_sizes: List[int], activation_fn=torch.nn.ReLU()):
        """
        Constrói uma MLP com as dimensões especificadas
        layer_sizes: lista com o número de neurônios em cada camada
        activation_fn: função de ativação para as camadas ocultas
        """
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]
        
        # Criar nós para cada camada
        prev_layer = []
        for layer_idx, size in enumerate(layer_sizes):
            current_layer = []
            # Usar função de ativação fornecida para camadas ocultas, Identity para saída
            layer_activation = activation_fn if layer_idx < len(layer_sizes)-1 else torch.nn.Identity()
            
            for _ in range(size):
                node = self.add_node(layer_idx, layer_activation)
                if layer_idx > 0:  # Adicionar bias exceto para a camada de entrada
                    node.bias = torch.randn(1, requires_grad=True)
                current_layer.append(node)
            
            # Conectar com a camada anterior
            if prev_layer:
                for source in prev_layer:
                    for target in current_layer:
                        self.add_edge(source, target)
            
            prev_layer = current_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza a passagem forward da rede
        x: tensor de entrada com shape (batch_size, input_size)
        """
        batch_size = x.shape[0]
        
        # Definir valores para os nós de entrada
        for i, node in enumerate(self.layers[0]):
            node.value = x[:, i]
        
        # Propagar através da rede
        output = []
        for node in self.layers[len(self.layers) - 1]:
            output.append(node.forward())
        
        # Armazenar a última saída para uso no backward pass
        self.last_output = torch.stack(output, dim=1)
        return self.last_output
    
    def backward(self, gradient: torch.Tensor):
        """
        Realiza a passagem backward da rede
        gradient: gradiente da função de perda em relação à saída
        """
        # Definir gradientes para os nós de saída
        output_layer = len(self.layers) - 1
        for i, node in enumerate(self.layers[output_layer]):
            node.backward(gradient[:, i])
        
        # Propagar gradientes para trás
        for layer_idx in range(output_layer - 1, -1, -1):
            for node in self.layers[layer_idx]:
                node.backward()
    
    def get_parameters(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Retorna os pesos e biases da rede
        """
        weights = [edge.weight for edge in self.edges]
        biases = []
        for layer_idx in range(1, len(self.layers)):  # Começar da primeira camada oculta
            for node in self.layers[layer_idx]:
                if hasattr(node, 'bias'):
                    biases.append(node.bias)
        return weights, biases