import re
import matplotlib.pyplot as plt
import numpy as np

# Example data: file names and corresponding cross-entropy values (you will have actual data)
file_names = ['100_gen.txt', '250_gen.txt', '300_spam.txt', '50_spam.txt', '450_gen.txt']
cross_entropy_values = [1.5, 1.8, 1.7, 1.9, 2.0]  # Example cross-entropy for each file

# Function to extract the file length from filename
def get_file_length(file_name):
    match = re.match(r'(\d+)', file_name)  # Match the first number in the filename
    if match:
        return int(match.group(1))
    return None

# Extract file lengths
file_lengths = [get_file_length(name) for name in file_names]

# Bin the file lengths into intervals (for example, intervals of 100 words)
bin_edges = np.arange(0, 501, 100)  # Bins from 0 to 500 in intervals of 100
bin_indices = np.digitize(file_lengths, bin_edges)

# Calculate average cross-entropy in each bin
binned_cross_entropy = [np.mean([cross_entropy_values[i] for i in range(len(bin_indices)) if bin_indices[i] == j])
                        for j in range(1, len(bin_edges))]

# Calculate bin centers for plotting
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(file_lengths, cross_entropy_values, label='Raw Data', color='blue')
plt.plot(bin_centers, binned_cross_entropy, label='Binned Average', color='red', marker='o')

plt.xlabel('File Length (words)')
plt.ylabel('Cross-Entropy')
plt.title('Cross-Entropy vs File Length with Add-λ Smoothing')
plt.legend()
plt.grid(True)
plt.show()