import os
import streamlit as st
import torch
from PIL import Image
import numpy as np
import pydicom
from io import BytesIO
import cv2

# Set page config
st.set_page_config(
    page_title="Aneurysm Detection",
    page_icon="🩺",
    layout="wide"
)

# Title and description
st.title("🩺 Aneurysm Detection System")
st.markdown("""
Upload DICOM or standard image files (PNG/JPG) to detect potential aneurysms.
The system will analyze the images and highlight regions of interest.
""")

# Initialize session state for model loading
@st.cache_resource
def load_model():
    """Load the trained model."""
    # TODO: Replace with actual model loading code
    # model = YourModel.load_from_checkpoint('path/to/checkpoint.ckpt')
    # model.eval()
    return None  # Return None for now as a placeholder

# Load model (cached)
model = load_model()

def process_dicom(dicom_file):
    """Process DICOM file and return image array."""
    try:
        dicom_data = pydicom.dcmread(BytesIO(dicom_file.getvalue()))
        img = dicom_data.pixel_array
        
        # Normalize to 0-255
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
        # If volume, take middle slice
        if len(img.shape) == 3:
            img = img[img.shape[0] // 2]
            
        return img
    except Exception as e:
        st.error(f"Error processing DICOM: {str(e)}")
        return None

def detect_aneurysms(image, confidence=0.5):
    """
    Process image through the model to detect aneurysms.
    Currently uses mock detections - replace with actual model inference.
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create a copy to draw on
    result = image.copy()
    
    # TODO: Replace with actual model inference
    # detections = model.predict(image)
    
    # Mock detection results (x, y, w, h, confidence)
    height, width = image.shape[:2]
    mock_detections = [
        [width//4, height//4, width//2, height//2, 0.87],
    ]
    
    detections = []
    for x, y, w, h, conf in mock_detections:
        if conf >= confidence:
            # Draw rectangle
            color = (0, 255, 0)  # Green
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            
            # Add confidence text
            label = f"{conf:.2f}"
            cv2.putText(result, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            detections.append({
                'bbox': [x, y, w, h],
                'confidence': conf,
                'class': 'aneurysm'
            })
    
    return result, detections

# Sidebar for model options
with st.sidebar:
    st.header("Settings")
    model_type = st.selectbox(
        "Select Model",
        ["CPM-Net (Default)", "3D-CNN-TR", "Ensemble"],
        help="Choose the model for detection"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Adjust the sensitivity of detection"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses deep learning to detect potential aneurysms in medical images.
    Upload DICOM files or standard images to get started.
    """)

# File uploader
uploaded_files = st.file_uploader(
    "### Upload Medical Images",
    type=["dcm", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Upload DICOM or standard image files"
)

# Process button
process_btn = st.button("Process Images", type="primary")

# Display uploaded files
if uploaded_files:
    st.subheader("Results")
    
    # Create tabs for each uploaded file
    tabs = st.tabs([file.name for file in uploaded_files])
    
    for idx, (tab, file) in enumerate(zip(tabs, uploaded_files)):
        with tab:
            col1, col2 = st.columns(2)
            
            try:
                # Load image
                if file.type == "application/dicom":
                    image = process_dicom(file)
                    if image is None:
                        continue
                else:
                    image = np.array(Image.open(file))
                
                # Display original image
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                
                # Process and display results when button is clicked
                if process_btn:
                    with st.spinner(f"Processing {file.name}..."):
                        # Process image through model
                        result, detections = detect_aneurysms(image, confidence_threshold)
                        
                        # Display result
                        with col2:
                            st.image(result, caption="Detection Result", use_container_width=True)
                            
                            if detections:
                                st.success(f"✅ Detected {len(detections)} potential aneurysm(s)")
                                
                                # Show detection details
                                with st.expander("Detection Details", expanded=False):
                                    for i, det in enumerate(detections, 1):
                                        st.write(f"**Detection {i}**")
                                        st.json(det)
                                
                                # Download button for results
                                result_img = Image.fromarray(result)
                                buf = BytesIO()
                                result_img.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                
                                st.download_button(
                                    label="Download Result",
                                    data=byte_im,
                                    file_name=f"result_{file.name}",
                                    mime="image/png"
                                )
                            else:
                                st.warning("No aneurysms detected above confidence threshold")
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                
    # Show processing complete message
    if process_btn:
        st.success("✅ All images processed!")

# Add some sample images if no files uploaded
else:
    st.markdown("### Sample Aneurysm Types")
    st.markdown("Upload your own images or try with these sample aneurysm types:")
    
    # Local sample images (relative to app directory)
    sample_images = [
        {"path": "images/Saccular.png", "name": "Saccular Aneurysm"},
        {"path": "images/Fusiform.png", "name": "Fusiform Aneurysm"},
        {"path": "images/Mycotic.png", "name": "Mycotic Aneurysm"},
        {"path": "images/Traumatic.png", "name": "Traumatic Aneurysm"}
    ]
    
    # Display images in a grid
    cols = st.columns(min(4, len(sample_images)))  # Max 4 columns
    
    for i, img_info in enumerate(sample_images):
        with cols[i % len(cols)]:
            try:
                img = Image.open(img_info["path"])
                st.image(img, use_container_width=True)
                st.caption(img_info["name"])
            except Exception as e:
                st.error(f"Error loading {img_info['name']}: {str(e)}")
    
    st.info("ℹ️ Upload your own DICOM or standard image files to analyze them for potential aneurysms.")

# Add footer
st.markdown("---")
st.markdown("""
**Note**: This is a demonstration application. For clinical use, please consult with medical professionals.
""")
