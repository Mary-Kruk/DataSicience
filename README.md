# DataSicience
Project: MNIST Classification and Image Verification Pipeline (Tasks 1 & 2)
This repository contains solutions for two separate tasks (Task 1 and Task 2).

Project Launch (Recommended Method)
The fastest and easiest way to run and verify both tasks is by using the Jupyter Notebook provided, which contains all necessary steps for dependency installation, data loading, and functional demonstration.

  1. Cloning the Repository
    Open your terminal or command prompt and clone the repository:

    git clone <YOUR_REPOSITORY_URL>
    cd <YOUR_REPOSITORY_FOLDER_NAME>

  2. Running the Jupyter Notebook
    You can open the file Task2.ipynb (or other relevant notebook) in any environment that supports Jupyter Notebooks (e.g., VS Code, Google Colab, or Jupyter Lab) and execute all code blocks sequentially.

Task 1: MNIST Image Classification + OOP
Task 1 focuses on creating three different classification models for the MNIST dataset (handwritten digits) and encapsulating them within a flexible Object-Oriented Programming (OOP) structure.

Implementation Requirements (OOP Structure)
The entire functionality of Task 1 must be implemented using OOP principles, including the following classes and interfaces:

  1. Interface (Abstract Base Class)
    An abstract class, MnistClassifierInterface (using Python's abc module), must be created to define the contract for all models. It must include two abstract methods:

    from abc import ABC, abstractmethod

  class MnistClassifierInterface(ABC):
      @abstractmethod
      def train(self, X_train, y_train):
          """Trains the model on the provided data."""
          pass
    
      @abstractmethod
      def predict(self, X_test):
          """Returns predictions for the test data."""
          pass

  2. Model Classes
    Each of the three models must be a separate class that inherits from MnistClassifierInterface and implements the train and predict methods:

    <img width="1485" height="421" alt="image" src="https://github.com/user-attachments/assets/373571c1-2634-4dc4-a44a-ebd586b3a05f" />

  3. Factory/Wrapper Class
    All three models must be hidden under a single wrapper class, MnistClassifier.

      - The MnistClassifier class accepts the algorithm name as an input parameter (algorithm).

      - Possible values for algorithm are: cnn, rf, and nn.

      - The constructor (__init__) of MnistClassifier must instantiate the corresponding model class (CNNModel, RandomForestClassifierModel, or FeedForwardNNModel).

      - This class serves as the single entry point, delegating the train and predict calls to the chosen internal model.

Implementation and Execution of Task 1
The entire implementation of Task 1 is contained within the code blocks inside the Jupyter Notebook.

The notebook will:

  1. Define all necessary classes (MnistClassifierInterface, CNNModel, MnistClassifier, etc.).

  2. Load the MNIST data.

  3. Demonstrate the training and prediction functionality of the MnistClassifier class for each algorithm (cnn, rf, nn).
 Task 2: "Text-to-Image" Verification Pipeline
Task 2 implements a pipeline to verify the correspondence between a text input (animal name) and an image input (the animal class in the picture).

Pipeline Functionality
The pipeline, primarily implemented in pipeline.py, performs the following steps:

  1. Text Extraction: Extracts the English animal name (e.g., cat, dog) from the input text. (Script inference_ner.py).

  2. Translation: Translates the English animal name to Italian (based on the map in pipeline.py).

  3. Image Classification: Classifies the input image to determine the animal class (using the Italian folder names). (Scripts train_classification.py and inference_classification.py).

  4. Verification: Compares whether the translated text name matches the classified image class.

Task 2 File Structure
The code for Task 2 consists of separate Python scripts that are integrated and executed within the Jupyter Notebook:

<img width="1399" height="478" alt="image" src="https://github.com/user-attachments/assets/9d07e6c1-264a-49db-a8f6-388410bc199a" />
