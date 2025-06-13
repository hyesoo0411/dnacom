import matplotlib.pyplot as plt
import numpy as np

# Function to read and parse weights from the file
def extract_weights_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    weights = []
    is_collecting = False
    current_tensor = []

    # Extract weights from the relevant section
    for line in lines:
        if 'layer_stack.1.weight' in line:
            is_collecting = True
        elif is_collecting:
            # Check if the line contains the start of tensor data or continuation
            if '[' in line:
                # Remove unwanted parts and parse the line into floats
                line = line.strip().replace('tensor(', '').replace('device=\'cuda:0\'', '')
                line = line.replace('[', '').replace(']', '').replace(' ', '')
                current_tensor.extend([float(x) for x in line.split(',') if x.strip()])
            elif ']' in line:
                # When the tensor ends, append and reset
                if current_tensor:
                    weights.append(np.array(current_tensor))
                    current_tensor = []  # Reset for next collection
                if len(weights) == 6:  # Stop after collecting 6 nodes
                    break
    
    return weights

# Load the weight data from the file (adjust the path as needed)
text_dir = "/home/hyesoo/DNAcomputing/State_dict/"
file_path = text_dir + "WSmodel_200units_lr0.001_epoch20_wQAT-101True_biasFalse_PruningTrue.txt"
weights = extract_weights_from_file(file_path)

# Check extracted weights and reshape to 7x7 if possible
for i in range(6):
    # Check if the extracted data has the correct size
    if len(weights[i]) == 49:  # Check for correct number of elements
        weight_matrix = weights[i].reshape(7, 7)
    else:
        print(f"Error: Hidden node {i+1} has {len(weights[i])} elements, expected 49. Data extraction issue.")
        continue
    
    # Plotting and saving the heatmap
    plt.figure(figsize=(5, 5))
    plt.imshow(weight_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Heatmap of Hidden Node {i+1}")
    
    # Save the heatmap as an image
    output_filename = f"heatmap_hidden_node_{i+1}.png"
    plt.savefig("/home/hyesoo/DNAcomputing/Heatmap/" + output_filename)
    plt.close()

    print(f"Saved heatmap of hidden node {i+1} as {output_filename}")
