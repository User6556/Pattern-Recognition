from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
from hopfield import HopfieldNetwork, preprocess_image, create_training_patterns, visualize_patterns, save_preprocessed_debug
import json

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'hopfield_neural_network_secret_key'

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Global variables for the trained network
network = None
training_patterns = None

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_network():
    """Initialize and train the Hopfield network with predefined patterns."""
    global network, training_patterns
    
    # Create training patterns (letters P and Q)
    training_patterns = create_training_patterns()
    
    # Initialize network with 49 neurons (7x7 grid)
    network = HopfieldNetwork(size=49)
    
    # Train the network with pattern names for better debugging
    pattern_names = list(training_patterns.keys())
    network.train(list(training_patterns.values()), pattern_names)
    
    print("Hopfield network initialized and trained with patterns:", list(training_patterns.keys()))

@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and pattern recognition."""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image for 7x7 grid
            input_pattern = preprocess_image(filepath, target_size=(7, 7))
            
            if input_pattern is None:
                flash('Error processing image. Please try a different image.')
                return redirect(url_for('index'))
            
            # Save preprocessed pattern for debugging
            save_preprocessed_debug(input_pattern, grid_size=(7, 7), save_path='static/preprocessed_debug.png')
            
            # Recall pattern using Hopfield network with debugging
            recalled_pattern, converged, iterations, debug_info = network.recall_with_debug(input_pattern)
            
            # Create visualization for 7x7 grid
            best_match, similarity = visualize_patterns(
                input_pattern, recalled_pattern, training_patterns,
                save_path='static/result.png', grid_size=(7, 7)
            )
            
            # Get detailed debugging information
            initial_distances = debug_info['initial_distances']
            final_distances = debug_info['final_distances']
            similarities = debug_info['final_similarities']
            
            # Prepare enhanced results with debugging information
            results = {
                'filename': filename,
                'converged': converged,
                'iterations': iterations,
                'best_match': best_match,
                'similarity': similarity,
                'similarities': similarities,
                'input_pattern': input_pattern.tolist(),
                'recalled_pattern': recalled_pattern.tolist(),
                'initial_distances': initial_distances,
                'final_distances': final_distances,
                'debug_info': debug_info
            }
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return render_template('results.html', results=results)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF)')
        return redirect(url_for('index'))

@app.route('/api/patterns')
def get_patterns():
    """API endpoint to get training patterns."""
    if training_patterns is None:
        return jsonify({'error': 'Network not initialized'}), 500
    
    patterns_data = {}
    for name, pattern in training_patterns.items():
        patterns_data[name] = {
            'pattern': pattern.tolist(),
            'grid': pattern.reshape(5, 5).tolist()
        }
    
    return jsonify(patterns_data)

@app.route('/api/recall', methods=['POST'])
def api_recall():
    """API endpoint for pattern recall."""
    try:
        data = request.get_json()
        
        if 'pattern' not in data:
            return jsonify({'error': 'No pattern provided'}), 400
        
        input_pattern = np.array(data['pattern'])
        
        if len(input_pattern) != 49:
            return jsonify({'error': 'Pattern must have 49 elements (7x7 grid)'}), 400
        
        # Recall pattern with debugging
        recalled_pattern, converged, iterations, debug_info = network.recall_with_debug(input_pattern)
        
        return jsonify({
            'recalled_pattern': recalled_pattern.tolist(),
            'recalled_grid': recalled_pattern.reshape(7, 7).tolist(),
            'converged': converged,
            'iterations': iterations,
            'best_match': debug_info['best_match'],
            'similarities': debug_info['final_similarities'],
            'initial_distances': debug_info['initial_distances'],
            'final_distances': debug_info['final_distances']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo')
def demo():
    """Demo page with interactive pattern input."""
    return render_template('demo.html', patterns=training_patterns)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Initialize the network
    initialize_network()
    
    # Run the Flask app
    print("Starting Hopfield Network Web Application...")
    print("Access the application at: http://localhost:5000")
    print("Training patterns available:", list(training_patterns.keys()))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
