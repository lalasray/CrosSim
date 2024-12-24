from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Initialize the model and tokenizer
model_id = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def infer_activity(file_name: str, sentence: str) -> str:
    # Combine the file name and sentence into a refined prompt
    prompt = (
        f"File name: {file_name}\n"
        f"Description: {sentence}\n"
        "What is the physical activity described above in a short descriptive phrase (e.g., Break Dance Twist)?\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the output
    outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.7, top_p=0.9)
    activity = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the activity response and clean up
    activity = activity.replace(prompt, "").strip()
    return activity

def process_file(file_path: str):
    try:
        with open(file_path, 'r') as file:
            # Read the first line as the sentence
            sentence = file.readline().strip()
            file_name = os.path.basename(file_path)
            
            # Infer the activity
            predicted_activity = infer_activity(file_name, sentence)
            print(f"File: {file_name}")
            print(f"Sentence: {sentence}")
            print(f"Predicted Activity: {predicted_activity}\n")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_directory(directory_path: str):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):  # Process only .txt files
                file_path = os.path.join(root, file)
                process_file(file_path)

# Example usage
directory_path = r"/home/lala/Documents/GitHub/CrosSim/Data/test/text/"  # Replace with your directory path
process_directory(directory_path)

