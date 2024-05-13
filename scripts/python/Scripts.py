import os
import random
import networkx as nx
import matplotlib.pyplot as plt

def parse_ports(file_content):
    input_ports = []
    output_ports = []
    parts = file_content.split()  # Split the content by spaces
    index = 0
    while index < len(parts):
        if parts[index] == 'I':  # Check for input ports
            num_input_ports = int(parts[index + 1])  # Get the number of input ports
            input_ports.extend(parts[index + 2:index + 2 + num_input_ports])  # Add input ports to the list
            index += num_input_ports + 2  # Move the index to the next section
        elif parts[index] == 'O':  # Check for output ports
            num_output_ports = int(parts[index + 1])  # Get the number of output ports
            output_ports.extend(parts[index + 2:index + 2 + num_output_ports])  # Add output ports to the list
            index += num_output_ports + 2  # Move the index to the next section
        else:
            index += 1  # Move to the next part if not 'I' or 'O'
    return input_ports, output_ports

def create_graph(selected_files,folder_path):
    G = nx.DiGraph()
    for i in range(len(selected_files) - 1):
        with open(os.path.join(folder_path, selected_files[i]), 'r') as file:
            file_content = file.read()
            _, output_ports = parse_ports(file_content)
        with open(os.path.join(folder_path, selected_files[i+1]), 'r') as file:
            file_content = file.read()
            input_ports, _ = parse_ports(file_content)
        for output_port in output_ports:
            for input_port in input_ports:
                G.add_edge(output_port, input_port)
    return G


def generate_cypher_file(selected_files, folder_path, cypher_file_path):
    f = open(cypher_file_path, "w")
    for i in range(len(selected_files)):
        with open(os.path.join(folder_path, selected_files[i]), 'r') as file:
            file_content = file.read()
            _, output_ports = parse_ports(file_content)
            macro_name = f"X{i}"
            macro_x = random.randint(0, 50000)
            macro_y = random.randint(0, 50000)
            f.write(f"CREATE (macro{i}: Macro {{name:'{macro_name}' , x: {macro_x} , y: {macro_y}}})\n")
    
    for i in range(len(selected_files) - 1):
        with open(os.path.join(folder_path, selected_files[i]), 'r') as file:
            file_content = file.read()
            _, output_ports = parse_ports(file_content)
        with open(os.path.join(folder_path, selected_files[i+1]), 'r') as file:
            file_content = file.read()
            input_ports, _ = parse_ports(file_content)
        for output_port in output_ports:
            for input_port in input_ports:
                f.write(f"CREATE (macro{i})-[w{random.randint(0, 100)}:IS_CONNECTED]->(macro{i+1})\n")
    
    f.close()



def select_files(folder_path, num_files):
    files = os.listdir(folder_path)
    selected_files = random.sample(files, num_files)
    return selected_files
