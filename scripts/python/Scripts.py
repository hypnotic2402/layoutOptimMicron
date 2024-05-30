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
            WIDTH_MAX = 150
            WIDTH_MIN = 50
            HEIGHT_MAX = 150
            HEIGHT_MIN = 50
            macro_w = random.randint(WIDTH_MIN,WIDTH_MAX)
            macro_h = random.randint(HEIGHT_MIN,HEIGHT_MAX)
            #w indicates width and h indicates height
            f.write(f"CREATE (macro{i}: Macro {{name:'{macro_name}' , w: {macro_w} , h: {macro_h}}})\n")
    
    for i in range(len(selected_files) - 1):
        # with open(os.path.join(folder_path, selected_files[i]), 'r') as file:
        #     file_content = file.read()
        #     _, output_ports = parse_ports(file_content)
        # with open(os.path.join(folder_path, selected_files[i+1]), 'r') as file:
        #     file_content = file.read()
        #     input_ports, _ = parse_ports(file_content)
        # for output_port in output_ports:
        #     for input_port in input_ports:
        #         f.write(f"CREATE (macro{i})-[w{random.randint(0, 100)}:IS_CONNECTED]->(macro{i+1})\n")
        f.write(f"CREATE (macro{i})-[w{random.randint(0, 100)}:IS_CONNECTED]->(macro{i+1})\n")
    f.close()



def select_files(folder_path, num_files):
    files = os.listdir(folder_path)
    selected_files = random.sample(files, num_files)
    return selected_files





def main(folder_path="../../examples/netlist_text_files"):
    files = os.listdir(folder_path)
    print("Available files in the folder:")
    for file_name in files:
        print(file_name)
    try:
        x = int(input("Enter the number of files you want to select: "))
        if x <= 0:
            print("Please enter a positive integer.")
            return
        elif x > len(files):
            print(f"There are only {len(files)} files available. Please select a smaller number.")
            return
        selected_files = select_files(folder_path, x)
        print(f"Selected files: {selected_files}")

        for file_name in selected_files:
            with open(os.path.join(folder_path, file_name), 'r') as file:
                file_content = file.read()
                input_ports, output_ports = parse_ports(file_content)
                print(f"Input ports of {file_name}: {input_ports}")
                print(f"Output ports of {file_name}: {output_ports}")

        G = create_graph(selected_files,folder_path)
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

        cypher_file_path = "output.cypher"
        generate_cypher_file(selected_files, folder_path, cypher_file_path)
        print(f"Cypher file generated: {cypher_file_path}")


    except ValueError:
        print("Please enter a valid integer.")



# def main():
#     folder_path = "text_files"
#     files = os.listdir(folder_path)
#     print("Available files in the folder:")
#     for file_name in files:
#         print(file_name)
#     try:
#         x = int(input("Enter the number of files you want to select: "))
#         if x <= 0:
#             print("Please enter a positive integer.")
#             return
#         elif x > len(files):
#             print(f"There are only {len(files)} files available. Please select a smaller number.")
#             return
#         selected_files = select_files(folder_path, x)
#         print(f"Selected files: {selected_files}")
#     except ValueError:
#         print("Please enter a valid integer.")

if __name__ == "__main__":
    main()

