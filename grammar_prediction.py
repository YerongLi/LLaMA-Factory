from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

# Selected error types to be plotted
selected_error_types = {
    'Sentence Structure Errors': 0,
    'Spelling Mistakes': 4,
    'Passive Voice Overuse': 21,
    'Redundancy/Repetition': 35,
    'No Error': 36
}

# Define index_to_error_type for selected error types
index_to_error_type = {value: key for key, value in selected_error_types.items()}

# Plot only selected error types
selected_index_to_error_type = {key: index_to_error_type[key] for key in sorted(selected_error_types.values())}

# Create new dictionaries to store error counts and percentages for selected error types
selected_output_error_counts = {}
selected_response_error_counts = {}
selected_gan_error_counts = {}
selected_output_error_percentages = {}
selected_response_error_percentages = {}
selected_gan_error_percentages = {}

# Update selected dictionaries with counts and percentages for selected error types
for error_type, index in selected_error_types.items():
    selected_output_error_counts[error_type] = output_error_counts[error_type]
    selected_response_error_counts[error_type] = response_error_counts[error_type]
    selected_gan_error_counts[error_type] = gan_error_counts[error_type]
    selected_output_error_percentages[error_type] = output_error_percentages[error_type]
    selected_response_error_percentages[error_type] = response_error_percentages[error_type]
    selected_gan_error_percentages[error_type] = gan_error_percentages[error_type]

# Determine the bar width
bar_width = 0.4

# Define the y-coordinates for the bars
y_pos = np.arange(len(selected_index_to_error_type))

plt.figure(figsize=(12, 8))  # Larger and wider figure

# Plot the blue bars (output)
plt.barh(y_pos - bar_width, list(selected_output_error_percentages.values()), color='blue', label='Human', alpha=0.5, height=bar_width)

# Plot the red bars (response)
plt.barh(y_pos, list(selected_response_error_percentages.values()), color='red', label='LLM', height=bar_width)

# Plot the green bars (GAN)
plt.barh(y_pos + bar_width, list(selected_gan_error_percentages.values()), color='green', label='GAN', alpha=0.5, height=bar_width)

plt.xlabel('Percentage')
plt.ylabel('Error Type')
plt.title('Error Type Frequencies')
plt.legend()

# Adjust font size
plt.yticks(y_pos, list(selected_index_to_error_type.values()), fontsize='small')


plt.savefig("Grammar.png")