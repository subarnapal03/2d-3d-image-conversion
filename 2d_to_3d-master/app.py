from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import numpy as np
import pyvista as pv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Directory to store uploaded images
app.config['MODEL_FOLDER'] = 'static/models/'   # Directory to store generated 3D models

# line for adding the trained model
trained_model = 'pointnet_model.pth'

def create_3d_model(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Create a mesh grid based on the image dimensions
    x = np.linspace(0, 1, image.shape[1])
    y = np.linspace(0, 1, image.shape[0])
    x, y = np.meshgrid(x, y)
    z = image / 255.0  # Normalize pixel values
    
    # Create the surface plot
    grid = pv.StructuredGrid(x, y, z)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, show_edges=True)
    
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'model.obj')
    plotter.export_obj(model_path)
    
    save_model = 'Saved trained model to {}'.format(trained_model)
    print(save_model)
    
    return model_path

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/app', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the uploaded file to the upload folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Create the 3D model from the uploaded image
            model_path = create_3d_model(filepath)
            if model_path:
                # Serve the 3D model file for download
                return send_from_directory(app.config['MODEL_FOLDER'], 'model.obj')
            else:
                return 'Error processing image'
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the upload and model directories exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['MODEL_FOLDER']):
        os.makedirs(app.config['MODEL_FOLDER'])
    
    # Run the Flask app
    app.run(debug=True)
