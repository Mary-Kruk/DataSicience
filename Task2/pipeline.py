
import os
import random
from IPython.display import display, Image as IPImage



import argparse
from inference_ner import extract_animal
from inference_classification import classify_image

# Dictionary: English -> Italian (folder names)
EN_TO_IT_MAP = {
    "cat": "gatto",
    "dog": "cane",
    "horse": "cavallo",
    "cow": "mucca",
    "elephant": "elefante",
    "butterfly": "farfalla",
    "chicken": "gallina",
    "sheep": "pecora",
    "spider": "ragno",
    "squirrel": "scoiattolo"
}

def verify_animal_in_image(text, image_path):
    print(f"Input text: '{text}'")
    print(f"Input image: '{image_path}'")

    #Extract English animal name from text
    animal_in_text_en = extract_animal(text)
    if not animal_in_text_en:
        print("Search result: Animal not found in text.")
        return False

    # Translate EN -> IT
    animal_in_text_it = EN_TO_IT_MAP.get(animal_in_text_en)
    if not animal_in_text_it:
        print(f"Translation error: Italian class name not found for '{animal_in_text_en}'.")
        return False
    print(f"Search and translation result: '{animal_in_text_en}' -> '{animal_in_text_it}'")

    #  Image classification
    animal_in_image = classify_image(image_path, "animal_classifier.pth", "class_names.json")
    print(f"Classification result: Image contains '{animal_in_image}'")

    return animal_in_text_it == animal_in_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    
    result = verify_animal_in_image(args.text, args.image)
    
    print("\n" + "="*20)
    print(f"FINAL RESULT: {result}")
    print("="*20)
