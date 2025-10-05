import spacy
import argparse

# A set with the main animal names in English
ENGLISH_ANIMALS = {
    "cat", "dog", "horse", "cow", "elephant", "butterfly",
    "chicken", "sheep", "spider", "squirrel"
}

def extract_animal(text):
    """Finds an animal name (in English) in the text."""
    # Remove punctuation and split the text into words
    words = text.lower().replace("!", "").replace(".", "").replace(",", "").split()
    for word in words:
        if word in ENGLISH_ANIMALS:
            return word # Return the found English word
    return None

