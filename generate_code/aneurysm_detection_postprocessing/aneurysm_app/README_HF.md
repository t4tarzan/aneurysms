# Aneurysm Detection App - Hugging Face Deployment

This guide explains how to deploy the Aneurysm Detection app to Hugging Face Spaces.

## Prerequisites

1. A Hugging Face account
2. Git installed on your local machine
3. Docker installed (for local testing)

## Deployment Steps

### 1. Create a new Space on Hugging Face

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Configure your Space:
   - Name: `aneurysm-detection`
   - License: MIT
   - Space SDK: Docker
   - Visibility: Public or Private
4. Click "Create Space"

### 2. Set up the Space

1. Clone the Space repository:
   ```bash
   git clone https://huggingface.co/spaces/your-username/aneurysm-detection
   cd aneurysm-detection
   ```

2. Copy the app files to the repository:
   ```bash
   cp /path/to/aneurysm_app/* .
   ```

3. Push the changes:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push
   ```

### 3. Configure the Space

1. In your Space settings, ensure the following environment variables are set if needed:
   - `HF_TOKEN`: Your Hugging Face authentication token (for private models)
   - `MODEL_PATH`: Path to your model (if using Hugging Face Hub)

2. Set the hardware requirements:
   - Go to Space settings
   - Under "Hardware", select a GPU instance if needed

## Local Testing (Optional)

To test the Docker container locally:

```bash
docker build -t aneurysm-detection .
docker run -p 7860:7860 --gpus all aneurysm-detection
```

Then open `http://localhost:7860` in your browser.

## Updating the App

To update the app:

1. Make your changes to the files
2. Commit and push the changes:
   ```bash
   git add .
   git commit -m "Update app"
   git push
   ```

## Troubleshooting

- If the app fails to start, check the logs in the Hugging Face Space
- Ensure all file paths are correct and all required files are included
- For large models, make sure you've selected an appropriate hardware tier

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
