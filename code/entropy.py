import subprocess
import re

# Define arrays for the different values of c and d
C_VALUES = [0, 0.1, 0.5, 1, 5]
D_VALUES = [10, 50, 200]

# Define model file paths
GEN_MODEL_BASE = "gen-log-linear-c={c}-d={d}.model"
SPAM_MODEL_BASE = "spam-log-linear-c={c}-d={d}.model"

# Define the prior and test files
PRIOR = 0.7
TEST_FILES = "../data/gen_spam/dev/{gen,spam}/*"  # Note: Adjust paths if needed

# Define the output file for cross-entropy results
OUTPUT_FILE = "cross_entropy_results.txt"

# Clear the output file if it already exists
with open(OUTPUT_FILE, "w") as f:
    f.write("")

# Regular expression to extract "Overall cross-entropy"
cross_entropy_pattern = re.compile(r"Overall cross-entropy:\s*([0-9.]+)")

# Function to run the textcat.py script and capture the output
def run_textcat(gen_model, spam_model):
    command = f"./textcat.py {gen_model} {spam_model} {PRIOR} {TEST_FILES}"
    print(f"Running command: {command}")
    # try:
    #     # Run the command and capture both stdout and stderr
    #     result = subprocess.run(command, capture_output=True, text=True, shell=True, check=True)
    #     return result.stdout
    # except subprocess.CalledProcessError as e:
    #     # Print standard error for debugging
    #     print(f"Error running textcat.py: {e}")
    #     print(f"Standard Output:\n{e.stdout}")
    #     print(f"Standard Error:\n{e.stderr}")
    #     return ""

# Iterate over the different values of c and d
for c in C_VALUES:
    for d in D_VALUES:
        # Replace placeholders in model filenames with the actual values of c and d
        gen_model = GEN_MODEL_BASE.format(c=c, d=d)
        spam_model = SPAM_MODEL_BASE.format(c=c, d=d)

        print(f"Running textcat with c={c}, d={d}...")

        # Run textcat.py and capture the output
        output = run_textcat(gen_model, spam_model)

        # # Search for the cross-entropy in the output
        # match = cross_entropy_pattern.search(output)
        # if match:
        #     cross_entropy = match.group(1)
        #     # Write the result to the output file
        #     with open(OUTPUT_FILE, "a") as f:
        #         f.write(f"c={c}, d={d}: Overall cross-entropy = {cross_entropy}\n")
        # else:
        #     # If no cross-entropy was found, log an error
        #     with open(OUTPUT_FILE, "a") as f:
        #         f.write(f"c={c}, d={d}: Error, cross-entropy not found\n")

print(f"All runs completed. Results saved in {OUTPUT_FILE}.")