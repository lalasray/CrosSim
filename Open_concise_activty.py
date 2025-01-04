import os
import csv
import json
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def save_class_list(class_list, file_path="class_list.json"):
    """Save the class list to an external JSON file."""
    with open(file_path, 'w') as file:
        json.dump(class_list, file)
    logging.info(f"Class list saved to {file_path}")


def load_class_list(file_path="class_list.json"):
    """Load the class list from an external JSON file. If it doesn't exist, initialize with default classes."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            class_list = json.load(file)
        logging.info(f"Class list loaded from {file_path}")
    else:
        # Initialize with default classes if the file doesn't exist
        initial_classes = ["running", "walking", "acting", "jumping", 
                           "eating", "sports", "housework", 
                           "tooling", "social interactions"]
        with open(file_path, 'w') as file:
            json.dump(initial_classes, file)
        logging.info(f"Class list initialized with default values and saved to {file_path}")
        class_list = initial_classes
    return class_list


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
    """Classify an activity using OpenAI and update the class list if necessary."""
    prompt1 = f"Can any of the following Classes closely describe this activity? {class_list}. Answer with only one of the Classes with the exact letters and with nothing else, without special characters. Define a new Class if none of the Classes are close enough to resemble the activity. The new Class should be on the same abstraction level as the given list of Classes."
    prompt1 = create_openai_prompt(text_sentence, prompt1)
    logging.info(f"Prompt sent: {prompt1}")
    class_label = get_openai_response(client, "gpt-3.5-turbo", prompt1)
    
    if not class_label:
        return None, None  # No response from the model

    class_label = class_label.replace("'", "").strip()
    try:
        class_ind = class_list.index(class_label)
    except ValueError:
        logging.info("New label identified. Validating...")
        prompt2 = f"Can any of the following Classes closely describe this activity? {class_list}. Answer with only one of the Classes with the exact letters and with nothing else, without special characters. Say NO if none of the Classes are close enough to resemble the activity."
        prompt2 = create_openai_prompt(class_label, prompt2)
        validation_response = get_openai_response(client, "gpt-3.5-turbo", prompt2)
        
        if validation_response and validation_response.upper() == "NO":
            logging.warning("New label rejected.")
            return None, None
        else:
            logging.info(f"New label '{class_label}' added.")
            class_list.append(class_label)
            class_ind = len(class_list) - 1

    return class_label, class_ind


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
                    # Save the updated class list after each file
                    save_class_list(class_list)


if __name__ == "__main__":
    # Load the class list (initialize if missing)
    class_list = load_class_list()

    # Specify the directory to process
    #directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/EgoBody/"
    #output_csv_file = "EgoBody_results.csv"
    
    #directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/aist/"
    #output_csv_file = "aist_results.csv"
    
    #directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/animation/"
    #output_csv_file = "animation_results.csv"
    
    #directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/dance/"
    #output_csv_file = "dance_results.csv"
    
    #directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/fitness/"
    #output_csv_file = "fitness_results.csv"
    
    directory_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/game_motion/"
    output_csv_file = "game_motion_results.csv"
    
    # Process all .txt files in the directory and subdirectories
    process_directory(directory_path, class_list, output_csv_file)
