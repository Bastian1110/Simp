from SimpNeuralNetwork import SimpNeuralNetwork

import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def plot_neural_network(layers):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for layer in layers:
        G.add_node(layer['id'], label=layer['name'], layer_type=layer['layer_type'], size=layer['size'])
    
    # Add edges
    for i in range(len(layers) - 1):
        source_layer = layers[i]
        target_layer = layers[i + 1]
        for j in range(source_layer['size']):
            for k in range(target_layer['size']):
                G.add_edge(source_layer['id'], target_layer['id'])

    # Set node positions
    pos = {}
    y_pos = {}
    layer_count = 0
    max_nodes = max(layer['size'] for layer in layers)
    
    for layer in layers:
        layer_id = layer['id']
        layer_size = layer['size']
        
        x = layer_count
        y_step = max_nodes / (layer_size + 1)
        
        y_start = (max_nodes - layer_size) / 2 + 0.5
        y_pos[layer_id] = []
        
        for i in range(layer_size):
            pos[(layer_id, i)] = (x, y_start + i * y_step)
            y_pos[layer_id].append(y_start + i * y_step)
        
        layer_count += 1

    # Draw nodes
    plt.figure(figsize=(12, 6))
    node_colors = {'input': 'lightblue', 'hidden': 'lightgreen', 'output': 'lightcoral'}
    
    for layer in layers:
        for i in range(layer['size']):
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=[(layer['id'], i)],
                                   node_color=node_colors[layer['layer_type']],
                                   node_size=1000,
                                   alpha=0.8,
                                   label=layer['name'])

    # Draw edges
    for u, v in G.edges():
        source_layer_size = layers[u]['size']
        target_layer_size = layers[v]['size']
        
        for i in range(source_layer_size):
            for j in range(target_layer_size):
                source_pos = (u, i)
                target_pos = (v, j)
                nx.draw_networkx_edges(G, pos, edgelist=[(source_pos, target_pos)], alpha=0.5, arrows=False)
    
    # Add labels
    for layer in layers:
        for i in range(layer['size']):
            label_pos = (layer['id'], i)
            plt.text(pos[label_pos][0], pos[label_pos][1] + 0.1, f"{layer['name']}_{i}",
                     fontsize=12, ha='center', va='center')
    
    plt.title("Neural Network Structure")
    plt.axis('off')
    plt.show()





nn = SimpNeuralNetwork("XOR")
nn.add_layer(2, layer_type="input")
nn.summary()
nn.add_layer(3)
nn.summary()
nn.add_layer(3)
nn.summary()
nn.add_layer(1, activation_function="none", layer_type="output")
nn.summary()
plot_neural_network(nn.layers)

