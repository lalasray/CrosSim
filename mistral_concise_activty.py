import os
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the model and tokenizer
model_id = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ACTIVITY_JSON_PATH = "activities.json"

def load_or_initialize_activity_json(file_path: str):
    """
    Load the activity JSON file, or initialize it if it doesn't exist.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {}

def save_activity_json(activity_dict: dict, file_path: str):
    """
    Save the activity dictionary to a JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(activity_dict, file, indent=4)

def get_activity_id(activity: str, activity_dict: dict):
    """
    Check if an activity exists in the dictionary. If not, add it.
    """
    if activity not in activity_dict:
        activity_dict[activity] = len(activity_dict) + 1
        save_activity_json(activity_dict, ACTIVITY_JSON_PATH)
    return activity_dict[activity]

def infer_activity(file_name: str, sentence: str, activity_dict: dict) -> (str, int):
    """
    Infer the physical activity and assign it an ID from the activity JSON file.
    """
    prompt = (
        f"File name: {file_name}\n"
        f"Description: {sentence}\n"
        "What is the physical activity described above in a short descriptive phrase? Limit repetition upto 3 words.\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.7)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    activity = output_text.split("Answer:")[-1].strip()
    activity_id = get_activity_id(activity, activity_dict)
    return activity, activity_id

def process_file(file_path: str, csv_writer, activity_dict: dict):
    """
    Process a single file to infer activity and write results to the CSV.
    """
    try:
        with open(file_path, 'r') as file:
            sentence = file.readline().strip()
            file_name = os.path.basename(file_path)
            predicted_activity, activity_id = infer_activity(file_name, sentence, activity_dict)
            logging.info(f"Processed file: {file_name} -> Predicted Activity: {predicted_activity} (ID: {activity_id})")
            csv_writer.writerow([file_name, predicted_activity, activity_id])
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

def process_directory(directory_path: str, csv_file_path: str, activity_dict: dict):
    """
    Process all .txt files in a directory and save results to a CSV file.
    """
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["File Name", "Predicted Activity", "Activity ID"])  # Header row
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    process_file(file_path, csv_writer, activity_dict)

def process_multiple_directories_parallel(directory_list: list, output_base_path: str):
    """
    Process multiple directories in parallel and save results to separate CSV files.
    """
    activity_dict = load_or_initialize_activity_json(ACTIVITY_JSON_PATH)
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_directory,
                directory,
                os.path.join(output_base_path, f"{os.path.basename(os.path.normpath(directory))}.csv"),
                activity_dict
            )
            for directory in directory_list
        ]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing directory: {e}")

# Example usage
if __name__ == "__main__":
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
    
    process_multiple_directories_parallel(directories, output_base_path)

