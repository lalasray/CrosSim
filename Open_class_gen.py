import os
import csv
import json
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Predefined classes
classes = {"running", "walking", "acting", "jumping", "eating", "sports", "housework", "tooling", "social interactions"}
default_class_list = list(classes)

def load_class_list(file_path="class_list.json"):
    """Load the class list from an external JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            class_list = json.load(file)
        logging.info(f"Class list loaded from {file_path}")
        return class_list
    else:
        logging.warning(f"No class list file found at {file_path}. Using default classes.")
        return default_class_list  # Fallback to initial classes

def create_openai_prompt(question, content):
    """Create a prompt for the OpenAI API."""
    return f"Given the following text description: \"{question}\". {content}"

def get_openai_response(client, model_version, prompt):
    """Get a response from the OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=model_version,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in getting response from OpenAI: {e}")
        return None

def llm_givelabel(text_sentence, class_list):
    """Classify an activity using OpenAI based on the predefined class list."""
    prompt = (
        f"Can any of the following Classes closely describe this activity? {class_list}. "
        f"Answer with only one of the Classes with the exact letters and nothing else, without special characters. "
    )
    full_prompt = create_openai_prompt(text_sentence, prompt)
    logging.info(f"Prompt sent: {full_prompt}")
    
    class_label = get_openai_response(client, "gpt-3.5-turbo", full_prompt)
    
    if not class_label:
        return None, None  # No response from the model

    class_label = class_label.replace("'", "").strip()

    if class_label.upper() == "NO" or class_label not in class_list:
        logging.warning(f"No matching class found for: {text_sentence}")
        return None, None

    class_index = class_list.index(class_label)
    return class_label, class_index

def process_file(file_path, class_list, csv_writer):
    """Process a file to classify its content."""
    try:
        with open(file_path, 'r') as file:
            sentence = file.readline().strip()
            file_name = os.path.basename(file_path)
            
            predicted_activity, activity_index = llm_givelabel(sentence, class_list)
            
            if predicted_activity:
                logging.info(f"File: {file_name}")
                logging.info(f"Sentence: {sentence}")
                logging.info(f"Predicted Activity: {predicted_activity} (Index: {activity_index})\n")
                # Write to CSV
                csv_writer.writerow([file_name, sentence, predicted_activity, activity_index])
            else:
                logging.warning(f"Failed to classify activity for file: {file_name}")

            return predicted_activity, activity_index
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        return None, None
    except Exception as e:
        logging.error(f"An error occurred while processing the file: {e}")
        return None, None

def process_directory(directory_path, class_list, csv_file):
    """Process all .txt files in a directory and its subdirectories."""
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write CSV headers
        csv_writer.writerow(["File Name", "Sentence", "Predicted Activity", "Activity Index"])

        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    logging.info(f"Processing file: {file_path}")
                    process_file(file_path, class_list, csv_writer)

if __name__ == "__main__":
    # Load the class list
    class_list = load_class_list()

    # Specify the directory to process
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/aist/"
    output_csv_file = "aist_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
    
    #directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/EgoBody/"
    #output_csv_file = "EgoBody_results.csv"
    #process_directory(directory_path, class_list, output_csv_file)
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/animation/"
    output_csv_file = "animation_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/dance/"
    output_csv_file = "dance_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/fitness/"
    output_csv_file = "fitness_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/game_motion/"
    output_csv_file = "game_motion_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/HAA500/"
    output_csv_file = "HAA500_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/humman/"
    output_csv_file = "humman_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/idea400/"
    output_csv_file = "idea400_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/kungfu/"
    output_csv_file = "kungfu_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/music/"
    output_csv_file = "music_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/perform/"
    output_csv_file = "perform_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/UniMocap/texts/"
    output_csv_file = "UniMocap_results.csv"
    process_directory(directory_path, class_list, output_csv_file)
