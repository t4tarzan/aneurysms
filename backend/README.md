# Aneurysm Detection API

A FastAPI-based backend service for detecting brain aneurysms in medical imaging data using a 3D-CNN-TR deep learning model.

## Features

- **DICOM Image Processing**: Handles DICOM file uploads and preprocessing
- **3D-CNN-TR Model**: Implements the 3D-CNN-TR architecture for aneurysm detection
- **Artery Segmentation**: Integrates with artery segmentation for improved detection
- **RESTful API**: Provides endpoints for uploading, processing, and retrieving results
- **Background Processing**: Long-running tasks are processed asynchronously
- **Interactive Documentation**: Auto-generated API documentation with Swagger UI and ReDoc

## Prerequisites

- Python 3.8+
- pip
- CUDA-capable GPU (recommended) for faster inference

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install PyTorch with CUDA support (recommended):
   ```bash
   # For CUDA 11.7
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   
   # For CPU-only (not recommended for production)
   # pip3 install torch torchvision torchaudio
   ```

5. Install SimpleITK for DICOM processing:
   ```bash
   pip install SimpleITK
   ```

## Configuration

Copy the example environment file and update the settings as needed:

```bash
cp .env.example .env
```

## Running the API

### Development Mode

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Production Mode

For production, use a production ASGI server like Uvicorn with Gunicorn:

```bash
pip install gunicorn

gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```

## API Documentation

- **Swagger UI**: `http://localhost:8000/api/docs`
- **ReDoc**: `http://localhost:8000/api/redoc`
- **OpenAPI Schema**: `http://localhost:8000/api/openapi.json`

## API Endpoints

### Upload DICOM Files

```http
POST /api/v1/aneurysm/upload
```

Upload DICOM files for processing. Returns an upload ID that can be used to track processing status and retrieve results.

### Start Processing

```http
POST /api/v1/aneurysm/process/{upload_id}
```

Start processing the uploaded DICOM files for aneurysm detection.

### Check Processing Status

```http
GET /api/v1/aneurysm/status/{upload_id}
```

Get the current status of a processing job.

### Get Results

```http
GET /api/v1/aneurysm/results/{upload_id}
```

Retrieve the results of a completed processing job.

## Development

### Code Style

This project uses `black` for code formatting and `isort` for import sorting. To format the code:

```bash
pip install black isort
black .
isort .
```

### Testing

Run tests with pytest:

```bash
pip install pytest pytest-cov
pytest --cov=app tests/
```

## Deployment

### Docker

A `Dockerfile` is provided for containerized deployment:

```bash
docker build -t aneurysm-detection-api .
docker run -p 8000:8000 aneurysm-detection-api
```

### Kubernetes

Example Kubernetes deployment files are provided in the `k8s/` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
