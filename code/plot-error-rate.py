import os
import subprocess
import matplotlib.pyplot as plt

# Define the paths and parameters
train_data_base = "../data/english_spanish/train/"
dev_data_base = "../data/english_spanish/dev/"
vocab_base = "vocab-english_spanish"
lambda_val = 0.5
threshold = 0.7

# Define the different training set sizes
training_multipliers = [1, 2, 5, 10, 20, 50]

# Function to run a shell command and capture the output
def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}\n{e.output}")
        return None

# Function to extract error rate from textcat.py output
def parse_textcat_output(output):
    lines = output.splitlines()
    error_rate = None
    for line in lines:
        if "Error rate:" in line:
            # Extract the error rate from the line
            error_rate = float(line.split(":")[1].replace("%", "").strip())
            break
    return error_rate

# Store error rates for each training size
error_rates = []

# Iterate over training set sizes
for multiplier in training_multipliers:
    print(f"\nRunning training and evaluation for en.{multiplier}K and sp{multiplier}K...\n")
    
    # Step 1: Build vocabulary based on the multiplier
    vocab_file = f"{vocab_base}-{multiplier}K.txt"
    
    train_data_gen_spam = f"{{en.{multiplier}K,sp.{multiplier}K}}"  # Use the multiplied datasets
    
    vocab_command = f"./build_vocab.py {train_data_base}{train_data_gen_spam} --threshold 3 --output {vocab_file}"
    print(f"Running command: {vocab_command}")
    run_command(vocab_command)

    # Step 2: Train language model for gen and spam with dynamic model names
    gen_model_base = f"en-{multiplier}K.model"
    spam_model_base = f"sp-{multiplier}K.model"
    
    train_data_gen = f'{train_data_base}en.{multiplier}K'
    train_data_spam = f'{train_data_base}sp.{multiplier}K'
    
    train_command_gen = f"./train_lm.py {vocab_file} add_lambda --lambda {lambda_val} {train_data_gen} --output {gen_model_base}"
    print(f"Running command: {train_command_gen}")
    run_command(train_command_gen)

    train_command_spam = f"./train_lm.py {vocab_file} add_lambda --lambda {lambda_val} {train_data_spam} --output {spam_model_base}"
    print(f"Running command: {train_command_spam}")
    run_command(train_command_spam)

    # Step 3: Evaluate using textcat.py on the dev dataset with dynamic model names
    dev_data_path = f"{dev_data_base}{{english,spanish}}/*/*"
    eval_command = f"./textcat.py {gen_model_base} {spam_model_base} {threshold} {dev_data_path}"
    print(f"Running command: {eval_command}")
    eval_output = run_command(eval_command)
    
    if eval_output:
        # Parse the output and calculate the error rate
        error_rate = parse_textcat_output(eval_output)
        if error_rate is not None:
            error_rates.append(error_rate)
            print(f"Training size multiplier {multiplier}: Error rate = {error_rate:.2f}%")

# Step 4: Plot the learning curve
plt.figure(figsize=(8, 6))
plt.plot(training_multipliers, error_rates, marker='o')

# Add labels and title
plt.xlabel('Training Size Multiplier (times-x)')
plt.ylabel('Error Rate (%)')
plt.title('Learning Curve: Error Rate vs Training Data Size')

# Annotate the points with their error rates
for i, multiplier in enumerate(training_multipliers):
    plt.text(multiplier, error_rates[i], f'{error_rates[i]:.2f}%', ha='right')

# Display the graph
plt.grid(True)
plt.show()