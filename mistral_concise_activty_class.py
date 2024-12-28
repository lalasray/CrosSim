import os
import csv
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define default classes
default_classes = [
    "energetic dance", "slow dance", "running", "walking", "acting", "jumping",
    "eating", "sports", "housework", "tooling", "social interactions",
    "fitness", "sitting down", "getting up"
]

class ActivityClassifier:
    def __init__(self, model_id, classes_file="/home/lala/Documents/GitHub/CrosSim/classes.json"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.classes_file = classes_file

        # Load classes from file or initialize with defaults
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                self.class_list = json.load(f)
                logger.info(f"Loaded existing classes from {classes_file}")
        else:
            self.class_list = default_classes
            self.save_classes()
            logger.info(f"Initialized classes and saved to {classes_file}")

    def save_classes(self):
        """Save the current list of classes to a JSON file."""
        with open(self.classes_file, 'w') as f:
            json.dump(self.class_list, f, indent=4)
        logger.info(f"Saved updated classes to {self.classes_file}")

    def classify_activity(self, text_sentence):
        """
        Classify an activity using Mistral and handle new classes dynamically.
        """
        prompt1 = (
            f"Can any of the following Classes closely describe this activity? {self.class_list}. "
            "Answer with only one of the Classes with the exact letters and nothing else, without special characters. "
            "Animals can be classified as acting. Define a new Class if none of the Classes are close enough to resemble "
            "the activity. The new Class should be at a similar abstraction level as the given list of Classes.\n"
        )

        try:
            inputs = self.tokenizer(prompt1 + text_sentence, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=50)
            response1 = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            logger.info(f"Response1: {response1}")

            try:
                class_index = self.class_list.index(response1)
            except ValueError:
                # Secondary prompt to confirm new class
                prompt2 = (
                    f"Can any of the following Classes closely describe this activity? {self.class_list}. "
                    "Answer with only one of the Classes with the exact letters and nothing else, without special characters. "
                    "Say NO if none of the Classes are close enough to resemble the activity.\n"
                )
                inputs = self.tokenizer(prompt2 + response1, return_tensors="pt")
                outputs = self.model.generate(**inputs, max_new_tokens=50)
                response2 = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                logger.info(f"Response2: {response2}")

                if response2 == "NO":
                    self.class_list.append(response1)
                    self.save_classes()  # Save updated classes
                    logger.info(f"New class added: {response1}")
                    class_index = len(self.class_list) - 1
                else:
                    response1 = response2
                    class_index = self.class_list.index(response1)

            return response1, class_index
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return "Error", -1


def process_file(file_path, csv_writer, classifier):
    """
    Read a file, classify its sentence using Mistral, and write the result into a CSV file.
    """
    try:
        with open(file_path, 'r') as file:
            sentence = file.readline().strip()
            file_name = os.path.basename(file_path)

            # Classify the activity
            class_label, class_index = classifier.classify_activity(sentence)

            # Write results to CSV
            csv_writer.writerow([file_name, sentence, class_label, class_index])
            logger.info(f"Processed file: {file_name}, Classified as: {class_label} (Index: {class_index})")
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")


def main():
    input_folder = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/perform/"
    output_csv = "/home/lala/Documents/GitHub/CrosSim/classified_activities.csv"
    model_id = "mistralai/Mistral-7B-v0.3"

    # Initialize the classifier
    classifier = ActivityClassifier(model_id)

    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["File Name", "Sentence", "Class Label", "Class Index"])

 	# Process each file in the input folder and its subfolders
        for root, _, files in os.walk(input_folder):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    process_file(file_path, csv_writer, classifier)

    logger.info(f"Classification completed. Results saved to {output_csv}.")


if __name__ == "__main__":
    main()
