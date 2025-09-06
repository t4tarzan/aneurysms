# Aneurysm Detection App

A Streamlit-based web application for detecting potential aneurysms in medical images.

## Features

- Supports DICOM, PNG, and JPG image formats
- Adjustable confidence threshold for detection
- Multiple model support (CPM-Net, 3D-CNN-TR, Ensemble)
- Interactive visualization of detection results
- Download results in PNG format

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd aneurysm_app
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload DICOM or standard image files using the file uploader

4. Adjust the confidence threshold if needed

5. Click "Process Images" to run the detection

## Deployment

### Option 1: Streamlit Sharing
1. Push your code to a GitHub repository
2. Sign up at [Streamlit Sharing](https://share.streamlit.io/)
3. Connect your repository and deploy

### Option 2: Local Docker
1. Build the Docker image:
   ```bash
   docker build -t aneurysm-app .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 aneurysm-app
   ```
3. Access at `http://localhost:8501`

## Model Integration

To integrate your trained model:

1. Add your model file (`.pth` or similar) to the project directory
2. Update the `detect_aneurysms()` function in `app.py` to load and use your model
3. Process the model outputs and format the detection results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This application is for research and educational purposes only. It is not intended for clinical use. Always consult with a qualified healthcare professional for medical diagnosis and treatment.
