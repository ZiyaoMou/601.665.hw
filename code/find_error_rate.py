import subprocess
import re

# Define the range of prior values to test (e.g., from 0.1 to 1.0 with a step of 0.1)
prior_values = sorted([round(i * 0.01, 2) for i in range(1, 101)], reverse=True)

# Model filenames
gen_model = "gen-log-linear-c=1-d=10.model"
spam_model = "spam-log-linear-c=1-d=10.model"
test_files = "../data/gen_spam/dev/{gen,spam}/*"

# Regular expression to extract error rate from the output
error_rate_pattern = re.compile(r"Error rate:\s*([0-9.]+)%")

# Function to run the textcat.py script and capture the error rate
def run_textcat(prior):
    command = f"./textcat.py {gen_model} {spam_model} {prior} {test_files}"
    try:
        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        output = result.stdout
        # Search for the error rate in the output
        match = error_rate_pattern.search(output)
        if match:
            return float(match.group(1))
        else:
            print(f"Error: Could not find error rate in output for prior={prior}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running textcat.py for prior={prior}: {e}")
        return None

# Loop through all the prior values and find the one with the lowest error rate
best_prior = None
lowest_error_rate = float('inf')

for prior in prior_values:
    print(f"Running textcat.py with prior={prior}...")
    error_rate = run_textcat(prior)
    print(f"Error rate: {error_rate}%")
    if error_rate is not None and error_rate < lowest_error_rate:
        lowest_error_rate = error_rate
        best_prior = prior

# Output the best result
if best_prior is not None:
    print(f"Best prior: {best_prior} with the lowest error rate: {lowest_error_rate:.2f}%")
else:
    print("No valid error rates found.")