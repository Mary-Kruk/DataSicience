import spacy
from spacy.training.example import Example
import random
import argparse
import os

# Training data in English
# The character indices have been updated to match the English sentences.
TRAIN_DATA = [
    ("There is a cow in the picture", {"entities": [(11, 14, "ANIMAL")]}),
    ("This is a picture of an elephant", {"entities": [(25, 33, "ANIMAL")]}),
    ("I see a dog in the picture", {"entities": [(8, 11, "ANIMAL")]}),
    ("A beautiful horse is in the frame", {"entities": [(10, 15, "ANIMAL")]}),
    ("This is definitely a butterfly", {"entities": [(20, 29, "ANIMAL")]}),
    ("A chicken lives on the farm", {"entities": [(2, 9, "ANIMAL")]}),
    ("A sheep is grazing in the meadow", {"entities": [(2, 7, "ANIMAL")]}),
    ("A small squirrel is sitting on a tree", {"entities": [(8, 16, "ANIMAL")]}),
    ("A dangerous spider has spun a web", {"entities": [(12, 18, "ANIMAL")]}),
    ("Look, it's a cat!", {"entities": [(14, 17, "ANIMAL")]}),
]

def train_ner_model(output_dir, iterations):
    # Create a blank English model
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    
    # Add the "ANIMAL" label to the pipeline
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Start the training process
    optimizer = nlp.begin_training()
    for itn in range(iterations):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        
        # Print the losses for each iteration
        print(f"Iteration {itn+1}/{iterations}, Losses: {losses}")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save the trained model to the output directory
    nlp.to_disk(output_dir)
    print(f"\nModel saved to '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NER model.")
    parser.add_argument("--output_dir", type=str, default="ner_model", help="Directory to save the trained model.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of training iterations.")
    args = parser.parse_args()
    
    train_ner_model(args.output_dir, args.iterations)