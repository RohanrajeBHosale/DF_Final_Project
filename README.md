
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

3. Set up Git LFS for large files:
   ```bash
   git lfs install
   git lfs pull
   ```

4. Ensure the dataset is correctly placed in the `data/` directory.

## Usage
### Training the Model
To train the model, run:
```bash
python model_training/train_model.py
```

### Running the Web Application
Start the Flask application:
```bash
python web_app/app.py
```

Access the web interface at `http://127.0.0.1:5000`.

### Testing
Use `predict_from_folder.py` to test images in a specific folder:
```bash
python predict_from_folder.py --folder <folder-path>
```

## Project Structure
- `data/`: Contains datasets for training and testing.
- `model_training/`: Scripts for training the steganography detection model.
- `web_app/`: Flask application for real-time predictions.
- `utilities/`: Helper functions and pretrained models.

## License
This project is licensed under the MIT License.
