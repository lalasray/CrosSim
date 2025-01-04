# Chat with an intelligent assistant in your terminal
from openai import OpenAI
import logging
import json
import os
import csv

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
# or point to openai
# client = OpenAI(api_key=api_key)  # API key can be set here or through environment variable


history = [
    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
    {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
]

completion = client.chat.completions.create(
    model="MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF",
    messages=history,
    temperature=0.7,
    stream=True,
)

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

# %%

classes = {"running", "walking", "acting", "jumping", "eating", "sports", "housework", "tooling", "social interactions"}
#classes = {"basic locomotion", "complex sports", "housework", "social", "none of the above"}
class_list = list(classes)


def llm_givelabel(text_sentence):
    prompt1 = f" Can any of the following Classes closely describe this activity? {classes}. Answer with only one of the Classes with the exact letters and with nothing else, without special characters. Animals can be classified as acting. Define a new Class if none of the Classes are close enough to resemble the activity. The new Class should be on the samilar abstraction level as the given list of Classes."
    prompt1 = create_openai_prompt(text_sentence, prompt1)
    print(prompt1)
    class_label = get_openai_response(client, "gpt-3.5-turbo", prompt1)
    print(class_label, ": ", text_sentence)
    class_label.replace("'", "") # replace hallucinated '
    
    try:
        class_ind=class_list.index(class_label)
    except:
        print("exception finding index, potentially new label")
        class_ind = len(class_list)+1
    if class_ind > len(class_list):
        # make sure it's new not hallucination
        prompt2 = f" Can any of the following Classes closely describe this activity? {classes}. Answer with only one of the Classes with the exact letters and with nothing else, without special characters. Say NO if none of the Classes are close enough to resemble the activity."
        prompt2 = create_openai_prompt(class_label, prompt2)
        # append new label
        new_label = class_label
        class_list.append(new_label)
        print(new_label)

    return  class_label,  class_ind

file_path = "/media/lala/Crucial X62/CrosSim/Data/MotionX/semantic_labels/text/EgoBody/recording_20210907_S04_S03_01/body_idx_1/001.txt"
try:
        with open(file_path, 'r') as file:
            # Read the first line as the sentence
            sentence = file.readline().strip()
            file_name = os.path.basename(file_path)
            
            # Infer the activity
            predicted_activity = llm_givelabel(sentence)
            
            # Print the results
            print(f"File: {file_name}")
            print(f"Sentence: {sentence}")
            print(f"Predicted Activity: {predicted_activity}\n")
            
            # Write the results to the CSV file
            #csv_writer.writerow([file_name, predicted_activity])
except FileNotFoundError:
    print(f"File {file_path} not found.")
except Exception as e:
    print(f"An error occurred: {e}")
