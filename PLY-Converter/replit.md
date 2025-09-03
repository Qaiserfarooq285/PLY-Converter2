# PLY File Converter

## Overview
A web-based PLY (Polygon File Format) converter that transforms PLY files into multiple 3D formats including STL, OBJ, GLB, 3MF, and DXF. The application uses robust parsing strategies and enforces surface reconstruction for watertight surfaces.

## Project Architecture
- **Backend**: Flask web server on Python 3.11
- **Frontend**: Bootstrap-based responsive UI with drag-and-drop file upload
- **Core Processing**: Open3D and Trimesh for 3D mesh processing
- **Surface Reconstruction**: Poisson reconstruction algorithm
- **Real-time Progress**: WebSocket-like progress tracking via AJAX polling

## Features
- **Multiple Format Support**: Convert PLY to STL, OBJ, GLB, 3MF, and DXF
- **Robust Parsing**: Multiple parsing strategies for PLY file compatibility
- **Color Preservation**: Maintains vertex colors and normals when possible
- **Surface Reconstruction**: Automatic Poisson reconstruction for solid surfaces
- **Progress Tracking**: Real-time conversion progress with detailed messages
- **File Management**: Automatic cleanup of temporary files

## Technology Stack
- Flask 3.1.2 (Web framework)
- Open3D 0.19.0 (3D data processing)
- Trimesh 4.7.4 (Mesh processing and export)
- NumPy 2.3.2 (Numerical computations)
- PLYFile 1.1.2 (PLY file parsing)
- PyGLTFLib 1.16.5 (GLB export)
- EZDXF 1.4.2 (DXF export)
- Scikit-image 0.25.2 (Image processing)

## File Structure
```
/
├── app.py                 # Flask web application
├── ply_converter.py       # Core PLY conversion logic
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── css/style.css     # Styling
│   └── js/main.js        # Frontend JavaScript
├── uploads/              # Temporary upload directory
├── outputs/              # Conversion output directory
└── pyproject.toml        # Python dependencies
```

## API Endpoints
- `GET /` - Main web interface
- `POST /upload` - Upload PLY file and start conversion
- `GET /progress/<id>` - Get conversion progress
- `GET /download/<id>/<format>` - Download converted file
- `POST /cleanup/<id>` - Clean up conversion files

## Configuration
- **Host**: 0.0.0.0 (Replit compatible)
- **Port**: 5000 (Replit frontend port)
- **Max File Size**: 500MB
- **Supported Formats**: PLY input, STL/OBJ/GLB/3MF/DXF output
- **Deployment**: Autoscale (stateless web application)

## Recent Changes
- September 3, 2025: Initial Replit setup and configuration
- Fixed syntax errors in PLY converter module
- Added PLYConverter wrapper class for Flask integration
- Configured deployment for autoscale hosting
- Set up development workflow on port 5000

## User Preferences
- Web-based interface preferred for accessibility
- Drag-and-drop file upload for user convenience
- Real-time progress feedback during conversions
- Multiple output format selection
- Automatic file cleanup after download

## Development Notes
- Uses Poisson reconstruction for all inputs (ensures watertight surfaces)
- Vertex colors embedded in GLB exports when available
- Fallback algorithms for increased PLY file compatibility
- Thread-based asynchronous processing for large files
- Bootstrap UI framework for responsive design