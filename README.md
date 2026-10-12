# Pothole Detection and Severity Classification System

This is a B.Tech Machine Learning project that classifies pothole severity into three categories: **Minor**, **Moderate**, and **Severe**. The application uses a trained image-classification model (MobileNetV2) and provides a user-friendly web interface using Streamlit.

## Technology Stack

- **Python**: Core programming language.
- **TensorFlow/Keras (MobileNetV2)**: For training the image-classification model using transfer learning.
- **Streamlit**: For the web application interface.
- **Pillow**: For image processing.
- **NumPy & pandas**: For data manipulation.
- **scikit-learn**: For dataset splitting and metrics.
- **matplotlib**: For generating plots and evaluation results.

## Project Structure

```text
pothole_detection/
├── data/
│   └── raw/              # Place the original downloaded Kaggle dataset files here
├── models/               # Will contain the trained model weights
├── outputs/              # Will contain evaluation results and generated plots
├── app.py                # Streamlit web interface for predictions
├── train.py              # Script to train the MobileNetV2 model
├── prepare_data.py       # Script to process annotations and split dataset
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation (this file)
└── .gitignore            # Git ignore configuration
```

## Setup Instructions (Windows)

Follow these steps to set up the project on your Windows machine.

### 1. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies. Open your terminal (Command Prompt or PowerShell), navigate to the root of the workspace, and run:

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

Activate the virtual environment with the following command:

```bash
# In Command Prompt:
venv\Scripts\activate.bat

# In PowerShell:
venv\Scripts\Activate.ps1
```
*(When activated, you will see `(venv)` at the beginning of your command prompt).*

### 3. Install Dependencies

Navigate to the `pothole_detection` directory and install the required Python packages:

```bash
cd pothole_detection
pip install -r requirements.txt
```

### 4. Add the Dataset

1. Download the dataset from Kaggle: [Annotated Potholes Dataset](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset/data).
2. Place the downloaded files directly into the `pothole_detection/data/raw/` directory. (Do not modify or rearrange the dataset manually—our scripts will handle that).

## How to Run

*Note: The model training and dataset preparation scripts are pending implementation in the next parts of the project.*

Once implemented, you will be able to run the application using:
```bash
streamlit run app.py
```
