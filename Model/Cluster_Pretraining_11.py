import argparse

# Define command-line argument parser
parser = argparse.ArgumentParser(description="Run specific training model based on input command.")
parser.add_argument("--cmt", type=str, required=True, help="Specify which training model to run")

args = parser.parse_args()
cmt = args.cmt  # Get the argument value

h5_file_path = "../../../../ds/other/CrosSim/UniMocap/full_dataset.h5"
batch_size = 1024

# Dictionary mapping commands to functions
train_functions = {
    "train_model": lambda: __import__("Pretraining_JLR_11").train_model(epochs=300, batch_size=batch_size, h5_file_path=h5_file_path),
    "train_model_ft": lambda: __import__("Pretraining_JLR_ft_11").train_model_ft(epochs=300, batch_size=batch_size, h5_file_path=h5_file_path),
    "train_bimodel_ft": lambda: __import__("Pretraining_bimodal_textimu_ft_11").train_bimodel_ft(epochs=300, batch_size=batch_size, h5_file_path=h5_file_path),
    "train_bimodel": lambda: __import__("Pretraining_bimodal_textimu_11").train_bimodel(epochs=300, batch_size=batch_size, h5_file_path=h5_file_path),
    "train_bipose": lambda: __import__("Pretraining_bimodal_imupose_11").train_bipose(epochs=300, batch_size=batch_size, h5_file_path=h5_file_path),
    "train_model_imubi": lambda: __import__("Pretraining_JLR_imubi_11").train_model_imubi(epochs=300, batch_size=batch_size/2, h5_file_path=h5_file_path),
    "train_model_imubi_ft": lambda: __import__("Pretraining_JLR_imubi_ft_11").train_model_imubi_ft(epochs=300, batch_size=batch_size/2, h5_file_path=h5_file_path),
}

# Run the selected function if it exists
if cmt in train_functions:
    print(f"Running {cmt}...")
    train_functions[cmt]()
else:
    print(f"Invalid cmt command: {cmt}")
