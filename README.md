# MLP Graph Visualizer

Este é um projeto de uma interface gráfica para visualização e treinamento interativo de Redes Neurais Multicamadas (MLP). O projeto permite visualizar em tempo real o processo de forward e backward propagation, com animações das ativações e gradientes através da rede.

## Características

- Interface gráfica intuitiva usando PyQt5
- Visualização interativa da arquitetura da rede usando NetworkX
- Suporte para diferentes funções de ativação:
  - ReLU
  - Sigmoid
  - Tanh
  - LeakyReLU
  - ELU
- Carregamento e pré-processamento de dados CSV
- Visualização em tempo real do treinamento
- Animação do fluxo de ativações (forward) e gradientes (backward)
- Monitoramento de métricas de treinamento
- Salvamento e carregamento de modelos treinados

## Estrutura do Projeto

```
MLPG/
├── datasets/              # Conjuntos de dados de exemplo
│   ├── diabetic_data.csv
│   ├── heart.csv
│   └── IDS_mapping.csv
├── src/                  # Código fonte
│   ├── __init__.py
│   ├── gui.py           # Interface gráfica
│   ├── mlp.py          # Implementação da MLP
│   └── neural_graph.py  # Implementação do grafo neural
├── main.py              # Ponto de entrada do programa
├── environment.yml      # Configurações do ambiente
└── README.md
```

## Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd MLPG
```

2. Crie e ative o ambiente conda:
```bash
conda env create -f environment.yml
conda activate mlp-graph-new
```

3. Execute o programa:
```bash
python main.py
```

## Como Usar

1. **Carregar Dados**:
   - Clique em "Selecionar arquivo CSV"
   - Selecione as colunas de entrada e saída
   - Os dados serão automaticamente divididos em treino e teste

2. **Configurar Rede**:
   - Defina o número de neurônios em cada camada
   - Escolha a função de ativação
   - Ajuste os parâmetros de treinamento (taxa de aprendizado, épocas, etc.)

3. **Treinar**:
   - Clique em "Create Network" para criar a rede
   - Clique em "Train" para iniciar o treinamento
   - Observe a animação do forward/backward pass
   - Acompanhe a loss e acurácia em tempo real

4. **Visualização**:
   - Nós verdes/vermelhos: ativações positivas/negativas
   - Nós azuis: gradientes durante o backward pass
   - Espessura das conexões: magnitude dos pesos
   - Valores numéricos: ativações/gradientes em cada nó

5. **Salvar/Carregar**:
   - Use "Save Model" para salvar o modelo treinado
   - Use "Load Model" para carregar um modelo existente

## Dependências

- Python 3.10
- PyQt5
- PyTorch
- NetworkX
- Matplotlib
- NumPy
- Pandas

## Contribuição

Sinta-se à vontade para contribuir com o projeto através de pull requests. Para mudanças maiores, abra primeiro uma issue para discussão.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.