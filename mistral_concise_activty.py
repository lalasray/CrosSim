from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import csv

# Initialize the model and tokenizer
model_id = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def infer_activity(file_name: str, sentence: str) -> str:
    # Combine the file name and sentence into a refined prompt
    prompt = (
        f"File name: {file_name}\n"
        f"Description: {sentence}\n"
        "What is the physical activity described above in a short descriptive phrase (e.g., Break Dance Twist)? stop repetation with massximun 3 words  \nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the output
    outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.7, top_p=0.9)
    activity = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the activity response and clean up
    activity = activity.replace(prompt, "").strip()
    return activity

def process_file(file_path: str, csv_writer):
    try:
        with open(file_path, 'r') as file:
            # Read the first line as the sentence
            sentence = file.readline().strip()
            file_name = os.path.basename(file_path)
            
            # Infer the activity
            predicted_activity = infer_activity(file_name, sentence)
            
            # Print the results
            print(f"File: {file_name}")
            print(f"Sentence: {sentence}")
            print(f"Predicted Activity: {predicted_activity}\n")
            
            # Write the results to the CSV file
            csv_writer.writerow([file_name, predicted_activity])
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_directory(directory_path: str, csv_file_path: str):
    # Open the CSV file for writing
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(["File Name", "Predicted Activity"])
        
        # Iterate through all files in the directory and subdirectories
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".txt"):  # Process only .txt files
                    file_path = os.path.join(root, file)
                    process_file(file_path, csv_writer)

def process_multiple_directories(directory_list: list, output_base_path: str):
    for directory in directory_list:
        # Generate a CSV file name based on the directory name
        directory_name = os.path.basename(os.path.normpath(directory))
        csv_file_path = os.path.join(output_base_path, f"{directory_name}.csv")
        
        print(f"Processing directory: {directory}")
        print(f"Output CSV: {csv_file_path}\n")
        
        # Process the directory and save results to the corresponding CSV file
        process_directory(directory, csv_file_path)

# Example usage
directories = [
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/aist",
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/animation",
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/dance",
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/EgoBody",
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/fitness",
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/game_motion",
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/HAA500",
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/humman",
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/idea400",
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/kungfu",
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/music",
    "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/perform",
    "/media/lala/Crucial X62/CrosSim/Data/UniMocap/texts",
]
output_base_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/hard_labels"

process_multiple_directories(directories, output_base_path)

