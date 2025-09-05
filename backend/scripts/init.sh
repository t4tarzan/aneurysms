#!/bin/bash

# Create necessary directories
mkdir -p app/{api/endpoints,core,models,services,utils,tests}

# Create empty __init__.py files
touch app/__init__.py
touch app/api/__init__.py
touch app/api/endpoints/__init__.py
touch app/core/__init__.py
touch app/models/__init__.py
touch app/services/__init__.py
touch app/utils/__init__.py
touch app/tests/__init__.py

# Create uploads directory
mkdir -p uploads

# Set permissions
chmod +x scripts/*.sh

echo "Backend directory structure initialized successfully!"
