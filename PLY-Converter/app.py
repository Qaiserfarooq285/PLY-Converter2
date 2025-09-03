import os
import uuid
import logging
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from ply_converter import PLYConverter
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store conversion progress
conversion_progress = {}
conversion_results = {}

ALLOWED_EXTENSIONS = {'ply'}
OUTPUT_FORMATS = ['stl', 'obj', 'glb', '3mf', 'dxf']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PLY files are allowed'}), 400
        
        # Generate unique ID for this conversion
        conversion_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{conversion_id}_{filename}")
        file.save(input_path)
        
        # Get output formats from request
        output_formats = request.form.getlist('formats')
        if not output_formats:
            output_formats = ['stl']  # Default to STL
        
        # Validate output formats
        valid_formats = [fmt for fmt in output_formats if fmt in OUTPUT_FORMATS]
        if not valid_formats:
            return jsonify({'error': 'Invalid output formats specified'}), 400
        
        # Initialize progress tracking
        conversion_progress[conversion_id] = {
            'status': 'starting',
            'progress': 0,
            'message': 'Initializing conversion...',
            'input_file': filename,
            'output_formats': valid_formats
        }
        
        # Start conversion in background thread
        thread = threading.Thread(
            target=convert_file_async,
            args=(conversion_id, input_path, valid_formats)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'conversion_id': conversion_id,
            'message': 'Conversion started',
            'input_file': filename,
            'output_formats': valid_formats
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

def convert_file_async(conversion_id, input_path, output_formats):
    """Convert PLY file asynchronously"""
    try:
        converter = PLYConverter()
        
        def progress_callback(message, progress=None):
            if conversion_id in conversion_progress:
                conversion_progress[conversion_id]['message'] = message
                if progress is not None:
                    conversion_progress[conversion_id]['progress'] = progress
                logger.info(f"Conversion {conversion_id}: {message}")
        
        # Perform conversion
        conversion_progress[conversion_id]['status'] = 'converting'
        results = converter.convert_ply(
            input_path, 
            app.config['OUTPUT_FOLDER'], 
            output_formats,
            conversion_id,
            progress_callback
        )
        
        # Store results
        conversion_results[conversion_id] = results
        conversion_progress[conversion_id]['status'] = 'completed'
        conversion_progress[conversion_id]['progress'] = 100
        conversion_progress[conversion_id]['message'] = 'Conversion completed! Point clouds automatically converted to solid surfaces.'
        
        # Clean up input file
        try:
            os.remove(input_path)
        except:
            pass
        
    except Exception as e:
        logger.error(f"Conversion error for {conversion_id}: {str(e)}")
        conversion_progress[conversion_id]['status'] = 'error'
        conversion_progress[conversion_id]['message'] = f'Conversion failed: {str(e)}'
        
        # Clean up input file
        try:
            os.remove(input_path)
        except:
            pass

@app.route('/progress/<conversion_id>')
def get_progress(conversion_id):
    """Get conversion progress"""
    if conversion_id not in conversion_progress:
        return jsonify({'error': 'Conversion not found'}), 404
    
    progress_data = conversion_progress[conversion_id].copy()
    
    # Add download links if conversion is completed
    if progress_data['status'] == 'completed' and conversion_id in conversion_results:
        results = conversion_results[conversion_id]
        progress_data['download_links'] = {}
        
        for format_name, file_path in results.items():
            if file_path and os.path.exists(file_path):
                progress_data['download_links'][format_name] = url_for(
                    'download_file', 
                    conversion_id=conversion_id, 
                    format_name=format_name
                )
    
    return jsonify(progress_data)

@app.route('/download/<conversion_id>/<format_name>')
def download_file(conversion_id, format_name):
    """Download converted file"""
    if conversion_id not in conversion_results:
        return jsonify({'error': 'Conversion not found'}), 404
    
    results = conversion_results[conversion_id]
    if format_name not in results:
        return jsonify({'error': 'Format not found'}), 404
    
    file_path = results[format_name]
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=f"converted_{conversion_id}.{format_name}"
    )

@app.route('/cleanup/<conversion_id>', methods=['POST'])
def cleanup_conversion(conversion_id):
    """Clean up conversion files and data"""
    try:
        # Remove from progress tracking
        if conversion_id in conversion_progress:
            del conversion_progress[conversion_id]
        
        # Remove output files and cleanup results
        if conversion_id in conversion_results:
            results = conversion_results[conversion_id]
            for file_path in results.values():
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
            del conversion_results[conversion_id]
        
        return jsonify({'message': 'Cleanup completed'})
        
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
