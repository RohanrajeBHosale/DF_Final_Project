
# Steganography Detection Project

This project aims to detect the presence of steganographic content in images using a deep learning model. The solution is built using TensorFlow and includes a Flask-based web application for user interaction.

## Features
- Detects steganographic content in uploaded images.
- Utilizes a MobileNetV2-based model trained on synthetic datasets.
- Provides confidence scores for predictions.
- Includes a user-friendly web interface for predictions.

## Installation
### Prerequisites
- Python 3.8 or later
- pip package manager

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Directly run the predict_from_folder.py
(this takes the files from a folder and then can use as input for detecting.
   
## Usage
### Training the Model

PLEASE REFER TO THE LINK TO DOWNLOAD THE DATA FOLDER:   https://umassd-my.sharepoint.com/:u:/g/personal/rbhosale_umassd_edu/EWL52tiiVJVDph_syU_CONUB7tOjTiKkpOTili1CA3UD-g?e=oSLmJb

To train the model, run:
```bash
python model_training/train_model.py
```


### Testing
Use `predict_from_folder.py` to test images in a specific folder:
```bash
python predict_from_folder.py --folder <folder-path>
```

## Project Structure
- `data/`: Contains datasets for training and testing.
- `model_training/`: Scripts for training the steganography detection model.
- `utilities/`: Helper functions and pretrained models.

## License
