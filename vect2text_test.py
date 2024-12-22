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
corrector = vec2text.load_pretrained_corrector("gtr-base")

# File paths
file_path = "sample\Dance_Break_3_Step_clip_1.txt"  # Replace with your file path
output_path = f"{os.path.splitext(file_path)[0]}_gtr.pt"  # Save embeddings as .pt file

# Read text from file
with open(file_path, "r") as file:
    text_list = [line.strip() for line in file if line.strip()]  # Remove blank lines

# Get embeddings for the text
embeddings = get_gtr_embeddings(text_list, encoder, tokenizer)

# Save embeddings to the same directory as the input file
torch.save(embeddings, output_path)

# Invert embeddings back to text
inverted_texts = vec2text.invert_embeddings(
    embeddings=embeddings.cuda(),
    corrector=corrector,
    num_steps=20,
)

# Print original and inverted texts
for original, inverted in zip(text_list, inverted_texts):
    print(f"Original: {original}")
    print(f"Inverted: {inverted}")
    print("-" * 50)

print(f"Embeddings saved to: {output_path}")
