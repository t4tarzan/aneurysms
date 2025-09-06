# Docker Deployment Guide for Hugging Face Spaces

This guide provides step-by-step instructions for deploying the Aneurysm Detection app to Hugging Face Spaces using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Git](https://git-scm.com/downloads) installed
- A [Hugging Face](https://huggingface.co/) account
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/quick-start) installed (optional)

## Table of Contents
1. [Local Docker Testing](#1-local-docker-testing)
2. [Hugging Face Space Setup](#2-hugging-face-space-setup)
3. [Deployment](#3-deployment)
4. [Troubleshooting](#4-troubleshooting)
5. [Updating the App](#5-updating-the-app)

## 1. Local Docker Testing

### 1.1 Build the Docker Image

```bash
# Navigate to the app directory
cd /path/to/aneurysm_app

# Build the Docker image
docker build -t aneurysm-detection .
```

### 1.2 Run the Container

For CPU-only mode:
```bash
docker run -p 7860:7860 --name aneurysm-app aneurysm-detection
```

For GPU support (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):
```bash
docker run --gpus all -p 7860:7860 --name aneurysm-app aneurysm-detection
```

### 1.3 Access the App

Open your browser and navigate to:
```
http://localhost:7860
```

## 2. Hugging Face Space Setup

### 2.1 Create a New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Configure your Space:
   - Name: `aneurysm-detection` (or your preferred name)
   - License: MIT
   - Space SDK: Docker
   - Visibility: Public or Private
4. Click "Create Space"

### 2.2 Set Up Git Repository

```bash
# Clone your Space's repository
git clone https://huggingface.co/spaces/your-username/aneurysm-detection
cd aneurysm-detection

# Copy the app files
cp -r /path/to/aneurysm_app/* .

# Commit and push the changes
git add .
git commit -m "Initial commit"
git push
```

## 3. Deployment

### 3.1 Configure Hardware

1. Go to your Space's settings
2. Under "Hardware", select:
   - CPU (basic) for testing
   - GPU (T4 small) for better performance
   - Higher GPU tiers for production use

### 3.2 Environment Variables

If your app requires any environment variables:

1. Go to your Space's settings
2. Under "Repository secrets and variables", add:
   - `HF_TOKEN`: Your Hugging Face authentication token (for private models)
   - `MODEL_PATH`: Path to your model (if using Hugging Face Hub)

### 3.3 Build and Deploy

Hugging Face will automatically build and deploy your app when you push changes. Monitor the build logs in the "Logs" tab of your Space.

## 4. Troubleshooting

### Common Issues

1. **Build Failures**
   - Check the logs in the "Logs" tab
   - Ensure all files are correctly copied
   - Verify Dockerfile syntax

2. **App Crashes**
   - Check the runtime logs
   - Ensure all dependencies are in `hf_requirements.txt`
   - Verify file paths in your code

3. **GPU Issues**
   - Make sure you've selected a GPU instance
   - Check CUDA compatibility
   - Update your Dockerfile if using custom CUDA versions

## 5. Updating the App

To update your deployed app:

```bash
# Make your changes to the files

# Commit and push the changes
git add .
git commit -m "Update: Add model integration"
git push
```

## 6. Advanced Configuration

### 6.1 Custom Domain

1. Go to your Space's settings
2. Under "Domain", enter your custom domain
3. Update your DNS settings as instructed

### 6.2 CI/CD Pipeline

For automated deployments, set up GitHub Actions:

1. Create `.github/workflows/deploy.yml` in your repository
2. Add your workflow configuration
3. Set up repository secrets for authentication

## 7. Monitoring and Maintenance

1. **Monitor Usage**
   - Check the "Metrics" tab for usage statistics
   - Set up alerts for errors

2. **Scale Resources**
   - Upgrade hardware if needed
   - Optimize your Docker image for smaller size

3. **Backup**
   - Regularly back up your model weights and configurations
   - Use Git tags for versioning important releases

## Support

For additional help, refer to:
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
