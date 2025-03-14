import argparse

# Define command-line argument parser
parser = argparse.ArgumentParser(description="Run specific training model based on input command.")
parser.add_argument("--cmt", type=str, required=True, help="Specify which training model to run")

args = parser.parse_args()
cmt = args.cmt  # Get the argument value

h5_file_path = "../../../../ds/other/CrosSim/UniMocap/full_dataset.h5"
batch_size = 256

# Dictionary mapping commands to functions
train_functions = {
    "train_bitext": lambda: __import__("Pretraining_bimodal_textimu").train_bimodel(epochs=500, batch_size=batch_size, h5_file_path=h5_file_path),
    "train_bitext_downtext": lambda: __import__("Pretraining_bimodal_textimu_downtext").train_bimodel(epochs=500, batch_size=batch_size, h5_file_path=h5_file_path),
    "train_bitext_downpose": lambda: __import__("Pretraining_bimodal_textimu_downpose").train_bimodel(epochs=500, batch_size=batch_size, h5_file_path=h5_file_path),
    
    
}

# Run the selected function if it exists
if cmt in train_functions:
    print(f"Running {cmt}...")
    train_functions[cmt]()
else:
    print(f"Invalid cmt command: {cmt}")
