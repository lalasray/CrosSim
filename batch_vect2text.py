import vec2text
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import os

# Function to get embeddings from text
def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer) -> torch.Tensor:

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length",).to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings

# Load models and tokenizer
encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

# Directory to process
root_directory = r"semantic_labels\motionx_seq_text_new\motionx_seq_text_v1.1\aist"  # Replace with the path to your root directory

# Process all .txt files in the directory and its subdirectories
for dirpath, _, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.endswith(".txt"):
            file_path = os.path.join(dirpath, filename)
            output_path = f"{os.path.splitext(file_path)[0]}_gtr.pt"

            # Read text from the file
            with open(file_path, "r") as file:
                text_list = [line.strip() for line in file if line.strip()]  # Remove blank lines

            # Skip empty files
            if not text_list:
                print(f"Skipping empty file: {file_path}")
                continue

            # Get embeddings for the text
            embeddings = get_gtr_embeddings(text_list, encoder, tokenizer)

            # Save embeddings to the same directory as the input file
            torch.save(embeddings, output_path)
            print(f"Processed and saved embeddings for: {file_path}")
