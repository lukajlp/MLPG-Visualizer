import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                         QHBoxLayout, QPushButton, QLabel, QSpinBox,
                         QLineEdit, QTextEdit, QFileDialog, QComboBox,
                         QRadioButton, QCheckBox)
from PyQt5.QtCore import Qt
from .neural_graph import NeuralGraph
from .mlp import MLPGraph

class MLPVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MLP Graph Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Variáveis de estado
        self.mlp = None
        self.df = None
        
        # Layout principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Painel de controle (lado esquerdo)
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)
        
        # Área de visualização (lado direito)
        visualization_panel = self._create_visualization_panel()
        layout.addWidget(visualization_panel)
        
        self._update_graph()

    def _create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Seleção de arquivo CSV
        csv_group = QWidget()
        csv_layout = QVBoxLayout(csv_group)
        
        # Botão para selecionar arquivo CSV
        csv_btn = QPushButton("Selecionar arquivo CSV")
        csv_btn.clicked.connect(self._load_csv)
        csv_layout.addWidget(csv_btn)
        
        # Seleção de colunas
        columns_layout = QHBoxLayout()
        
        # Colunas de entrada
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("Colunas de entrada:"))
        self.input_cols_list = QTextEdit()
        self.input_cols_list.setEnabled(False)
        self.input_cols_list.setMaximumHeight(60)
        input_layout.addWidget(self.input_cols_list)
        
        # Botão para selecionar todas as colunas de entrada
        select_all_input = QPushButton("Selecionar Todas")
        select_all_input.clicked.connect(lambda: self._select_all_columns('input'))
        input_layout.addWidget(select_all_input)
        
        columns_layout.addLayout(input_layout)
        
        # Colunas de saída
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("Colunas de saída:"))
        self.output_cols_list = QTextEdit()
        self.output_cols_list.setEnabled(False)
        self.output_cols_list.setMaximumHeight(60)
        output_layout.addWidget(self.output_cols_list)
        
        # Botão para selecionar todas as colunas de saída
        select_all_output = QPushButton("Selecionar Todas")
        select_all_output.clicked.connect(lambda: self._select_all_columns('output'))
        output_layout.addWidget(select_all_output)
        
        columns_layout.addLayout(output_layout)
        
        csv_layout.addLayout(columns_layout)
        layout.addWidget(csv_group)
        
        # Configuração da arquitetura
        arch_group = QWidget()
        arch_layout = QVBoxLayout(arch_group)
        
        # Input size
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Size:"))
        self.input_size = QSpinBox()
        self.input_size.setRange(1, 100)
        self.input_size.setValue(2)
        input_layout.addWidget(self.input_size)
        arch_layout.addLayout(input_layout)
        
        # Hidden layers
        hidden_layout = QHBoxLayout()
        hidden_layout.addWidget(QLabel("Hidden Layers:"))
        self.hidden_layers = QLineEdit("4,3")
        self.hidden_layers.setPlaceholderText("Example: 4,3")
        hidden_layout.addWidget(self.hidden_layers)
        arch_layout.addLayout(hidden_layout)
        
        # Output size
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Size:"))
        self.output_size = QSpinBox()
        self.output_size.setRange(1, 100)
        self.output_size.setValue(1)
        output_layout.addWidget(self.output_size)
        arch_layout.addLayout(output_layout)
        
        # Função de ativação
        activation_layout = QHBoxLayout()
        activation_layout.addWidget(QLabel("Função de Ativação:"))
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU"])
        activation_layout.addWidget(self.activation_combo)
        arch_layout.addLayout(activation_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.learning_rate = QLineEdit("0.01")
        lr_layout.addWidget(self.learning_rate)
        arch_layout.addLayout(lr_layout)
        
        # Opções de salvamento
        save_layout = QHBoxLayout()
        self.save_model_cb = QCheckBox("Salvar modelo após treino")
        save_layout.addWidget(self.save_model_cb)
        arch_layout.addLayout(save_layout)
        
        layout.addWidget(arch_group)
        
        # Configurações de treinamento
        train_group = QWidget()
        train_layout = QVBoxLayout(train_group)
        train_layout.addWidget(QLabel("Configurações de Treinamento"))
        
        # Número de épocas
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Épocas:"))
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 10000)
        self.epochs_spinbox.setValue(1000)
        epochs_layout.addWidget(self.epochs_spinbox)
        train_layout.addLayout(epochs_layout)
        
        # Tamanho do batch
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Tamanho do Batch:"))
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1000)
        self.batch_size.setValue(32)
        batch_layout.addWidget(self.batch_size)
        train_layout.addLayout(batch_layout)
        
        # Split treino/teste
        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel("% Treino:"))
        self.train_split = QSpinBox()
        self.train_split.setRange(10, 90)
        self.train_split.setValue(80)
        split_layout.addWidget(self.train_split)
        train_layout.addLayout(split_layout)
        
        # Pré-processamento
        preprocess_layout = QHBoxLayout()
        preprocess_layout.addWidget(QLabel("Pré-processamento:"))
        self.preprocess_group = QWidget()
        preprocess_options = QHBoxLayout(self.preprocess_group)
        
        self.no_preprocess = QRadioButton("Nenhum")
        self.normalize = QRadioButton("Normalização")
        self.standardize = QRadioButton("Padronização")
        
        preprocess_options.addWidget(self.no_preprocess)
        preprocess_options.addWidget(self.normalize)
        preprocess_options.addWidget(self.standardize)
        
        # Selecionar padronização por padrão
        self.standardize.setChecked(True)
        
        preprocess_layout.addWidget(self.preprocess_group)
        train_layout.addLayout(preprocess_layout)
        
        layout.addWidget(train_group)
        
        # Botões
        buttons_layout = QHBoxLayout()
        
        # Botão para criar rede
        create_btn = QPushButton("Create Network")
        create_btn.clicked.connect(self._create_network)
        buttons_layout.addWidget(create_btn)
        
        # Botão para treinar
        train_btn = QPushButton("Train")
        train_btn.clicked.connect(self.train_network)
        buttons_layout.addWidget(train_btn)
        
        # Botão para carregar modelo
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self._load_model)
        buttons_layout.addWidget(load_btn)
        
        # Botão para testar modelo
        test_btn = QPushButton("Test Model")
        test_btn.clicked.connect(self._test_model)
        buttons_layout.addWidget(test_btn)
        
        layout.addLayout(buttons_layout)
        
        # Área de log
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)
        
        layout.addStretch()
        
        return panel

    def _load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar arquivo CSV", "", "CSV Files (*.csv)")
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.log_area.append(f"Arquivo CSV carregado: {file_path}")
                
                # Atualizar text edits com as colunas
                columns = self.df.columns.tolist()
                
                self.input_cols_list.setEnabled(True)
                self.output_cols_list.setEnabled(True)
                
                # Configurar tamanhos iniciais
                self.input_size.setValue(len(columns)-1)  # Todas exceto a última
                self.output_size.setValue(1)  # Uma coluna de saída
                
                # Adicionar todas as colunas por padrão
                self._select_all_columns('input')
                self.output_cols_list.setText(columns[-1])  # Última coluna como saída
                
            except Exception as e:
                self.log_area.append(f"Erro ao carregar CSV: {str(e)}")
    
    def _create_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Criar figura para o grafo da rede
        self.figure = plt.figure(figsize=(10, 8))
        self.ax_network = self.figure.add_subplot(111)
        
        # Criar canvas
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Adicionar área para mostrar a loss atual
        self.loss_label = QLabel("Loss: -")
        layout.addWidget(self.loss_label)
        
        return panel
        
    def _plot_final_loss(self, losses):
        # Criar uma nova janela para mostrar o gráfico de loss
        loss_window = QMainWindow()
        loss_window.setWindowTitle("Curva de Aprendizado")
        
        # Criar widget central
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        # Criar figura para o gráfico de loss
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plotar curva de loss
        ax.plot(losses, 'b-', label='Loss', linewidth=2)
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Curva de Aprendizado', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        
        # Criar canvas e adicionar ao layout
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Gráfico será salvo ao final do treinamento na pasta específica
        
        loss_window.setCentralWidget(central_widget)
        loss_window.resize(800, 500)
        loss_window.show()
    
    def _create_network(self):
        try:
            # Obter configuração da rede
            input_size = self.input_size.value()
            hidden_layers = [int(x.strip()) for x in self.hidden_layers.text().split(',')]
            output_size = self.output_size.value()
            learning_rate = float(self.learning_rate.text())
            
            # Obter função de ativação selecionada
            activation = self.activation_combo.currentText()
            
            # Criar nova rede neural
            self.mlp = MLPGraph(input_size, hidden_layers, output_size, learning_rate, activation)
            
            # Atualizar visualização
            self._update_graph()
            
            self.log_area.append(f"Created network with architecture: {[input_size] + hidden_layers + [output_size]}, activation: {activation}")
            
        except Exception as e:
            self.log_area.append(f"Error creating network: {str(e)}")
    
    def _select_all_columns(self, which='input'):
        if self.df is None:
            return
            
        columns = self.df.columns.tolist()
        if which == 'input':
            # Todas as colunas exceto a última como entrada
            self.input_cols_list.setText(','.join(columns[:-1]))
        else:
            # Última coluna como saída
            self.output_cols_list.setText(columns[-1])

    def _load_model(self):
        """Carrega um modelo salvo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Carregar Modelo", "", "Model Files (*.pkl)")
        
        if file_path:
            try:
                self.mlp = MLPGraph.load_model(file_path)
                self.log_area.append(f"Modelo carregado: {file_path}")
                
                # Atualizar interface com os parâmetros do modelo
                self.input_size.setValue(self.mlp.input_size)
                self.output_size.setValue(self.mlp.output_size)
                self.hidden_layers.setText(",".join(map(str, self.mlp.hidden_sizes)))
                self.learning_rate.setText(str(self.mlp.learning_rate))
                self.activation_combo.setCurrentText(self.mlp.activation_name)
                
                # Atualizar visualização
                self._update_graph(show_weights=True)
            except Exception as e:
                self.log_area.append(f"Erro ao carregar modelo: {str(e)}")
    
    def _test_model(self):
        """Testa o modelo com um novo dataset"""
        if self.mlp is None:
            self.log_area.append("Por favor, crie ou carregue uma rede primeiro!")
            return
        
        # Carregar arquivo de teste
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar arquivo CSV para teste", "", "CSV Files (*.csv)")
        
        if file_path:
            try:
                # Carregar dados
                test_df = pd.read_csv(file_path)
                self.log_area.append(f"Arquivo de teste carregado: {file_path}")
                
                # Obter colunas
                input_cols = [col.strip() for col in self.input_cols_list.toPlainText().split(',') if col.strip()]
                output_cols = [col.strip() for col in self.output_cols_list.toPlainText().split(',') if col.strip()]
                
                if not input_cols or not output_cols:
                    self.log_area.append("Por favor, selecione as colunas de entrada e saída!")
                    return
                
                # Preparar dados
                X = torch.tensor(test_df[input_cols].values, dtype=torch.float32)
                y = torch.tensor(test_df[output_cols].values, dtype=torch.float32)
                
                # Pré-processamento
                if self.standardize.isChecked():
                    X = (X - X.mean(dim=0)) / X.std(dim=0)
                    y = (y - y.mean(dim=0)) / y.std(dim=0)
                elif self.normalize.isChecked():
                    X = (X - X.min(dim=0)[0]) / (X.max(dim=0)[0] - X.min(dim=0)[0])
                    y = (y - y.min(dim=0)[0]) / (y.max(dim=0)[0] - y.min(dim=0)[0])
                
                # Avaliar modelo
                loss, accuracy = self.mlp.evaluate(X, y)
                self.log_area.append(f"\nResultados no conjunto de teste:")
                self.log_area.append(f"Loss: {loss:.4f}, Acurácia: {accuracy:.2%}")
                
                # Mostrar algumas predições
                with torch.no_grad():
                    sample_size = min(5, len(X))
                    indices = np.random.choice(len(X), sample_size, replace=False)
                    
                    self.log_area.append("\nExemplos de predições:")
                    for idx in indices:
                        inputs = X[idx]
                        pred = self.mlp.predict(inputs.unsqueeze(0))
                        
                        original_input = test_df[input_cols].iloc[idx]
                        original_output = test_df[output_cols].iloc[idx]
                        
                        self.log_area.append(
                            f"Entrada: {dict(zip(input_cols, original_input))} -> "
                            f"Predição: {dict(zip(output_cols, pred.numpy().flatten()))} "
                            f"(Real: {dict(zip(output_cols, original_output))})"
                        )
            
            except Exception as e:
                self.log_area.append(f"Erro durante o teste: {str(e)}")
    
    def _on_training_update(self, phase, outputs, loss, activations=None):
        """Callback chamado durante o treinamento para atualizar a visualização"""
        # Atualizar apenas após o backward pass (fim da época)
        if phase == 'backward':
            # Atualizar visualização com os pesos e ativações
            self._update_graph(show_weights=True, activations=activations)
            self.canvas.draw()
            QApplication.processEvents()
    
    def train_network(self):
        if self.mlp is None:
            self.log_area.append("Por favor, crie a rede primeiro!")
            return
            
        if self.df is None:
            self.log_area.append("Por favor, carregue um arquivo CSV primeiro!")
            return
            
        try:
            # Preparar dados do CSV
            input_cols = [col.strip() for col in self.input_cols_list.toPlainText().split(',') if col.strip()]
            output_cols = [col.strip() for col in self.output_cols_list.toPlainText().split(',') if col.strip()]
            
            if not input_cols or not output_cols:
                self.log_area.append("Por favor, selecione as colunas de entrada e saída!")
                return
            
            # Converter dados para tensor
            X = torch.tensor(self.df[input_cols].values, dtype=torch.float32)
            y = torch.tensor(self.df[output_cols].values, dtype=torch.float32)
            
            # Pré-processamento dos dados
            if self.standardize.isChecked():
                self.log_area.append("Aplicando padronização...")
                X = (X - X.mean(dim=0)) / X.std(dim=0)
                y = (y - y.mean(dim=0)) / y.std(dim=0)
            elif self.normalize.isChecked():
                self.log_area.append("Aplicando normalização...")
                X = (X - X.min(dim=0)[0]) / (X.max(dim=0)[0] - X.min(dim=0)[0])
                y = (y - y.min(dim=0)[0]) / (y.max(dim=0)[0] - y.min(dim=0)[0])
            else:
                self.log_area.append("Dados sem pré-processamento...")
            
            # Split treino/teste
            train_size = int(len(X) * self.train_split.value() / 100)
            indices = torch.randperm(len(X))
            
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            
            self.log_area.append(f"Split de dados: {train_size} amostras para treino, {len(X) - train_size} para teste")
            
            # Parâmetros de treinamento
            epochs = self.epochs_spinbox.value()
            batch_size = self.batch_size.value()
            
            # Treinar rede e coletar histórico de loss
            losses = []
            best_loss = float('inf')
            patience = 10  # Número de épocas para esperar por melhoria
            patience_counter = 0
            min_delta = 0.001  # Mínima mudança considerada como melhoria
            
            self.log_area.append(f"Iniciando treinamento com {epochs} épocas...")
            
            # Treinar a rede com visualização em tempo real
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # Atualizar título com a época atual
                self.ax_network.set_title(f"Época {epoch+1}/{epochs}")
                self.canvas.draw()
                
                for i in range(0, len(X_train), batch_size):
                    batch_x = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    # Forward pass com animação
                    loss = self.mlp.train_step(batch_x, batch_y, 
                                             callback=self._on_training_update,
                                             current_epoch=epoch+1,
                                             total_epochs=epochs)
                    epoch_loss += loss
                    num_batches += 1
                
                epoch_loss /= num_batches
                losses.append(epoch_loss)
                
                # Early stopping check
                if epoch_loss < best_loss - min_delta:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Atualizar status a cada época
                status_msg = f"Época {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}"
                print(f"\r{status_msg}", end="", flush=True)  # Mostrar no terminal
                self.log_area.setText(status_msg)  # Apenas época e loss atual
                # Atualizar título do gráfico com a época e loss atual
                self.ax_network.set_title(status_msg, fontsize=12, pad=10)
                # Atualizar visualização da rede com os pesos atuais
                self._update_graph(show_weights=True)
                self.canvas.draw()
                
                # Early stopping
                if patience_counter >= patience:
                    self.log_area.setText(f"Early stopping na época {epoch+1} - Loss não melhorou por {patience} épocas")
                    break
            
            # Avaliar no conjunto de treino e teste
            train_loss, train_accuracy = self.mlp.evaluate(X_train, y_train)
            test_loss, test_accuracy = self.mlp.evaluate(X_test, y_test)
            
            # Atualizar visualização final com os resultados
            final_status = f"Treinamento concluído - Loss final: {test_loss:.4f}"
            self.log_area.setText(final_status)
            
            # Criar pasta para esta execução
            runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs')
            if not os.path.exists(runs_dir):
                os.makedirs(runs_dir)
            
            # Encontrar próximo número de execução
            existing_runs = [int(d.replace('run_', '')) for d in os.listdir(runs_dir) 
                           if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith('run_')]
            next_run = max(existing_runs + [0]) + 1
            run_dir = os.path.join(runs_dir, f'run_{next_run}')
            os.makedirs(run_dir)
            
            # Salvar gráfico de loss
            loss_path = os.path.join(run_dir, 'loss_history.png')
            plt.savefig(loss_path, dpi=300, bbox_inches='tight')
            self.log_area.append(f"\nGráfico de loss salvo em: {loss_path}")
            
            # Salvar modelo se selecionado
            if self.save_model_cb.isChecked():
                try:
                    model_path = os.path.join(run_dir, 'model.pkl')
                    self.mlp.save_model(model_path)
                    self.log_area.append(f"Modelo salvo em: {model_path}")
                except Exception as e:
                    self.log_area.append(f"Erro ao salvar modelo: {str(e)}")
            
            # Mostrar gráfico de loss em uma nova janela
            self._plot_final_loss(losses)
            
            # Atualizar visualização final da rede
            self._update_graph(show_weights=True)
            self.canvas.draw()
            
            # Mostrar algumas predições de exemplo
            with torch.no_grad():
                sample_size = min(5, len(X))
                indices = np.random.choice(len(X), sample_size, replace=False)
                
                self.log_area.append("\nPredições de exemplo:")
                for idx in indices:
                    inputs = X[idx]
                    pred = self.mlp.predict(inputs.unsqueeze(0))
                    
                    original_input = self.df[input_cols].iloc[idx]
                    original_target = self.df[output_cols].iloc[idx]
                    
                    self.log_area.append(
                        f"Entrada: {dict(zip(input_cols, original_input))} -> "
                        f"Predição: {dict(zip(output_cols, pred.numpy().flatten()))} "
                        f"(Real: {dict(zip(output_cols, original_target))})"
                    )
            
        except Exception as e:
            self.log_area.append(f"Erro durante o treinamento: {str(e)}")
    
    def _update_graph(self, show_weights=False, activations=None, phase='forward'):
        self.ax_network.clear()
        
        if self.mlp is None:
            self.ax_network.text(0.5, 0.5, "No network created yet", 
                        horizontalalignment='center',
                        verticalalignment='center')
            self.canvas.draw()
            return
        
        # Criar grafo NetworkX
        G = nx.DiGraph()
        
        # Adicionar nós
        pos = {}
        labels = {}
        colors = []
        
        max_layer = max(self.mlp.graph.layers.keys())
        for layer_idx, nodes in self.mlp.graph.layers.items():
            layer_x = layer_idx / max_layer
            n_nodes = len(nodes)
            
            for i, node in enumerate(nodes):
                # Posicionar nós uniformemente em cada camada
                y_pos = (i - (n_nodes-1)/2) / max(n_nodes-1, 1)
                pos[node.id] = (layer_x, y_pos)
                
                G.add_node(node.id)
                
                # Se temos ativações, mostrar os valores nos nós
                if activations is not None and node.id in activations:
                    val = float(activations[node.id])
                    labels[node.id] = f"{val:.1f}"  # Reduzir casas decimais
                    self.ax_network.text(pos[node.id][0], pos[node.id][1], labels[node.id],
                                     fontsize=8, ha='center', va='center')
                    
                    # Cor baseada no valor da ativação
                    if phase == 'forward':
                        # No forward pass, vermelho para negativo, verde para positivo
                        intensity = min(abs(val), 1.0)
                        if val >= 0:
                            colors.append((0.5*(1-intensity), 0.5+0.5*intensity, 0.5*(1-intensity)))  # Verde
                        else:
                            colors.append((0.5+0.5*intensity, 0.5*(1-intensity), 0.5*(1-intensity)))  # Vermelho
                    else:
                        # No backward pass, azul para gradientes
                        intensity = min(abs(val), 1.0)
                        colors.append((0.5*(1-intensity), 0.5*(1-intensity), 0.5+0.5*intensity))  # Azul
                else:
                    labels[node.id] = f"N{node.id}"
                    # Cor padrão sem ativação
                    if layer_idx == 0:
                        colors.append('lightblue')  # Input
                    elif layer_idx == max_layer:
                        colors.append('lightgreen')  # Output
                    else:
                        colors.append('lightgray')  # Hidden
        
        # Adicionar arestas com pesos e cores baseadas nos valores
        edges = []
        edge_colors = []
        widths = []
        edge_labels = {}
        
        for edge in self.mlp.graph.edges:
            edges.append((edge.source.id, edge.target.id))
            if show_weights and hasattr(edge, 'weight'):
                weight = float(edge.weight)
                # Normalizar peso para cor (vermelho para negativo, azul para positivo)
                color = '#FF6B6B' if weight < 0 else '#4D96FF'  # Cores mais suaves
                edge_colors.append(color)
                widths.append(1 + 3 * abs(weight))  # Espessura mais visível
                edge_labels[(edge.source.id, edge.target.id)] = f"{weight:+.2f}"  # Mostrar sinal
            else:
                edge_colors.append('gray')
                widths.append(1.0)
        
        # Desenhar grafo
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1500, ax=self.ax_network)
        nx.draw_networkx_labels(G, pos, labels=labels, ax=self.ax_network, font_size=10, font_weight='bold')
        
        # Desenhar arestas com cores e espessuras diferentes
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,
                             width=widths, ax=self.ax_network, arrowsize=20,
                             alpha=0.8)  # Arestas mais visíveis
        
        # Mostrar pesos nas arestas com fundo branco
        if show_weights:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                       ax=self.ax_network, font_size=9,
                                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Adicionar título com a loss atual
        if hasattr(self, 'loss_label'):
            current_loss = self.loss_label.text().replace('Loss atual: ', '')
            self.ax_network.set_title(f"Neural Network Graph (Loss: {current_loss})")
        else:
            self.ax_network.set_title("Neural Network Graph")
        
        # Remover eixos
        self.ax_network.set_axis_off()
        
        self.canvas.draw()