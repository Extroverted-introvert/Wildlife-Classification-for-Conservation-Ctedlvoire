## Wildlife Image Classification for Conservation in Côte d'Ivoire
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

This project, developed as part of the WorldQuant University AI Lab on Deep Learning for Computer Vision, focuses on applying deep learning techniques to classify wildlife from camera trap images. The goal is to build a robust model that can accurately identify different animal species (or their absence) to aid conservation efforts in Côte d'Ivoire.

The project progresses from foundational data exploration and building a simple binary classifier to developing a sophisticated Convolutional Neural Network (CNN) for a multiclass classification challenge, culminating in a submission for the DrivenData competition.

### 📊 Key Results
The final Convolutional Neural Network (CNN) model achieved a validation accuracy of approximately 84% on the multiclass classification task. The model demonstrates a strong ability to distinguish between eight different classes, including seven animal species and empty "blank" images.

(Example Confusion Matrix from the final model evaluation)

### ✨ Key Features & Learning Highlights
Data Exploration and Preprocessing: Analyzed class distributions, handled images with different sizes and color modes (grayscale vs. RGB), and implemented data normalization to improve model performance.
Binary Classification Baseline: Built and trained a shallow, fully-connected neural network to establish a baseline for a simplified binary task (identifying a "hog" vs. a "blank" image).
Multiclass Classification with CNNs: Designed and trained a Convolutional Neural Network (CNN) from scratch, leveraging Conv2d, MaxPool2d, and Dropout layers to effectively handle the complexity of image data.
End-to-End PyTorch Workflow: Mastered the complete PyTorch pipeline, including creating custom Transforms, using ImageFolder and DataLoader for efficient data handling, defining a training loop, and evaluating model performance.
Competition Submission: Processed a test dataset, generated class predictions (confidences), and formatted the output into a CSV file according to competition specifications.

### 📂 File Structure
The project is organized into a series of Jupyter notebooks that document the step-by-step process.

- #### 01_Data_Exploration_and_Tensors.ipynb:

Introduces fundamental PyTorch tensor operations (creation, attributes, slicing, math).
Explores the project's file structure and visualizes the class distribution.
Loads images using Pillow, examines their properties (size, mode), and converts them into tensors.

- #### 02_Debugging_Techniques.ipynb:
A practical guide to reading and interpreting Python and PyTorch tracebacks.
Covers common errors like NameError, TypeError, and PyTorch-specific RuntimeError (e.g., device mismatch).

- #### 03_Binary_Classification_Shallow_NN.ipynb:
Focuses on a simplified binary classification problem ("hog" vs. "blank").
Implements a complete data preparation pipeline: custom transforms, train/validation split, and DataLoaders.
Builds, trains, and evaluates a shallow, fully-connected neural network using nn.Sequential.
Introduces the core training loop, loss function (CrossEntropyLoss), optimizer (Adam), and evaluation metrics (accuracy, confusion matrix).

- #### 04_Multiclass_Classification_CNN.ipynb:
Tackles the full multiclass classification problem with eight classes.
Implements data normalization by calculating the mean and standard deviation of the training set.
Builds a Convolutional Neural Network (CNN) architecture from scratch.
Trains the CNN on the full dataset and evaluates its performance.
Generates a submission.csv file with model predictions for the competition's test set.

### 🛠️ Technologies Used
- Core Libraries:
    - `Python 3`
    - `PyTorch & Torchvision`
    - `NumPy`
    - `Pandas`
- Data Visualization & Image Processing:
    - `Matplotlib`
    - `Pillow (PIL)`
    - `scikit-learn (for metrics like confusion_matrix)`
- Utilities:
    - `Jupyter Notebook`
    - `tqdm (for progress bars)`

### 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  It is recommended to create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required packages:
    ```bash
    pip install torch torchvision opencv-python matplotlib pandas pyyaml torchinfo
    ```
4. Run the notebooks:
    - Launch Jupyter and run the notebooks in sequential order (01 to 04) to reproduce the entire workflow.

jupyter notebook
🙏 Acknowledgements & License
This project was made possible by the excellent curriculum and resources provided by WorldQuant University (WQU).

This file © 2024 by WorldQuant University is licensed under CC BY-NC-ND 4.0.