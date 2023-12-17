from flask import Flask, jsonify, render_template, request
from roboflow import Roboflow
# from PIL import Image, ImageDraw
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


rf = Roboflow(api_key="6aYB3RfkHPUv4AOQbma7")
project = rf.workspace().project("e-scooter-h83gg")
model = project.version(1).model


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        # Check if the 'file' key is in the request files
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Save the uploaded file
        file_path = 'uploaded_image.jpg'
        file.save(file_path)

        # Infer on the uploaded image
        response = model.predict(file_path, confidence=50, overlap=50)

        # Initialize object count
        object_count = 0

        # List to store detected objects
        detected_objects = []

        # Display the detected object numbers
        for i, prediction in enumerate(response):
            try:
                object_count += 1
                class_name = prediction["class"]
                confidence = prediction["confidence"]

                # Add object information to the list
                detected_objects.append({
                    'object_number': object_count,
                    'class_name': class_name,
                    'confidence': confidence
                })

            except KeyError as e:
                print(f"Error processing prediction {i + 1}: {e}")

        # Return the detected objects as JSON response
        return jsonify({'detected_objects': detected_objects, 'total_objects': object_count})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'})
    
if __name__ == '__main__':
    app.run(debug=True)
