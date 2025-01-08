import os
import json
import logging
import datetime
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
#openai.api_key =

def get_openai_response(prompt):
    """Get a response from the OpenAI API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Ensure this is the correct model identifier
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in getting response from OpenAI: {e}")
        return None

def save_class_list(class_list, directory="class_lists"):
    """Save the class list to a new JSON file with a unique name in the specified directory."""
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate a unique filename using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"class_list_{timestamp}.json"
    file_path = os.path.join(directory, file_name)
    
    # Save the class list to the new file
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
        initial_classes = ["running", "walking", "jumping", "eating", "dancing", "filming", "interacting", "standing", "sleeping", "biking", "skiing", "squatting", "stretching", "observing", "skateboarding", "transforming", "rowing", "boxing", "climbing", "driving", "riding", "sitting", "weightlifting"]
        with open(file_path, 'w') as file:
            json.dump(initial_classes, file)
        logging.info(f"Class list initialized with default values and saved to {file_path}")
        class_list = initial_classes
    return class_list

def create_openai_prompt(question, content):
    """Create a prompt for the OpenAI API."""
    return f"Given the following text description: \"{question}\". {content}"

def llm_givelabel(text_sentence, class_list):
    """Classify an activity using OpenAI and update the class list if necessary."""
    prompt1 = f"Can any of the following Classes closely describe this activity? {class_list}. Answer with only one of the Classes with the exact letters and with nothing else, without special characters. If none of the classes are appropriate, propose a new class at the same level of abstraction as the existing ones."
    prompt1 = create_openai_prompt(text_sentence, prompt1)
    logging.info(f"Prompt sent: {prompt1}")
    class_label = get_openai_response(prompt1)

    if not class_label:
        return None, None  # No response from the model

    class_label = class_label.replace("'", "").strip()

    # Handle cases where OpenAI suggests a new class
    if "None of the classes are appropriate" in class_label:
        # Extract the suggested class name
        suggested_class = class_label.split('could be')[-1].strip().strip('"')
        logging.info(f"New class identified: {suggested_class}. Validating...")
        
        # Validate the new class
        prompt2 = f"Can the following class: '{suggested_class}' be classified under any of the existing classes {class_list}? Answer with YES if it matches, or NO if it doesn't."
        prompt2 = create_openai_prompt(suggested_class, prompt2)
        validation_response = get_openai_response(prompt2)

        if validation_response.upper() == "NO":
            class_list.append(suggested_class)
            # Save the updated class list with a new timestamped filename
            save_class_list(class_list)
            class_ind = len(class_list) - 1
            return suggested_class, class_ind

    # Check if the class exists in the list
    if class_label in class_list:
        class_ind = class_list.index(class_label)
    else:
        logging.warning(f"Class '{class_label}' not found.")
        return None, None

    return class_label, class_ind

def process_file(file_path, class_list):
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
            else:
                logging.warning(f"Failed to classify activity for file: {file_name}")

            # Save the class list after processing each file
            save_class_list(class_list)

            return predicted_activity, activity_index
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        return None, None
    except Exception as e:
        logging.error(f"An error occurred while processing the file: {e}")
        return None, None

def process_directory(directory_path, class_list):
    """Process all .txt files in a directory and its subdirectories."""
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                logging.info(f"Processing file: {file_path}")
                process_file(file_path, class_list)

if __name__ == "__main__":
    class_list = load_class_list()

    directory_path = "/media/lala/Crucial X62/CrosSim/Data/UniMocap/texts/"
    
    process_directory(directory_path, class_list)

