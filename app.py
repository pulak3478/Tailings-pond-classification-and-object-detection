from flask import Flask
import os
from flask import Flask, render_template, request, redirect, url_for,jsonify, Response 
from ultralytics import YOLO
import cv2
import base64
from io import BytesIO
import numpy as np
from pyngrok import ngrok
import sqlite3
import uuid
import time
import threading
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image

app=Flask(__name__)
model = keras.models.load_model('vgg16t.h5')
models=YOLO('last.pt')
class_name = ['no_tailings','tailings']

def generate_unique_filename(filename):
    _, extension = os.path.splitext(filename)
    unique_filename = str(uuid.uuid4()) + extension
    return unique_filename

@app.route('/')
def home():
    image_exists = os.path.exists('static/temp.JPG')
    if image_exists:
        image_url = f'/static/temp.JPG'
    else:
        image_url = None

    return render_template('index.html', image_exists=image_exists, image_url=image_url, background_image_url="/static/pf.jpg")
@app.route('/prediction', methods=["POST"])
def prediction():
    img = request.files['img']
    
    # Generate a unique filename
    unique_filename = generate_unique_filename(img.filename)
    image_urls = os.path.join('static', 'images', unique_filename)
    
    # Save the image
    img.save(image_urls)
  
    # Load and preprocess the image for prediction
    img = image.load_img(image_urls, target_size=(224, 224)) #299,299 for inceptionV3 model
    x = image.img_to_array(img) / 255
    resized_img_np = np.expand_dims(x, axis=0)
    
    # Make predictions using the model
    prediction = model.predict(resized_img_np)
    pred_class_index = np.argmax(prediction)
    pred_class_name = class_name[pred_class_index]

    # Render the template with the prediction results
    return render_template("index.html", data=prediction, class_name=pred_class_name, image_url=image_urls, background_image_url="/static/pf.jpg")

@app.route('/pure',methods=['GET','POST'])
def pure():
    image_path = os.path.exists('static/temp.jpg')
    if image_path:
        image_url = f'/static/temp.jpg'
    else:
        image_url = None 

    return render_template('pure.html', image_path=image_path,image_url=image_url,background_image_url="/static/pf.jpg")

from PIL import Image

@app.route('/predictions', methods=["GET","POST"])
def predictions():

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)

        if file:
            unique_filename = generate_unique_filename(file.filename)
            image_path = os.path.join('static', 'images', unique_filename)
            file.save(image_path)
            
            # Open the uploaded image
            img = Image.open(image_path)
            
            # Resize the image to 608x608
            img = img.resize((608, 608))
            
            # Save the resized image
            resized_image_path = os.path.join('static', 'images', 'resized_' + unique_filename)
            img.save(resized_image_path)
            
            # Run inference on the resized image
            results = models(resized_image_path)  # results list

            # Visualize the results
            for i, r in enumerate(results):
                # Plot results image
                im_bgr = r.plot()  # BGR-order numpy array
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

                # Save the result image
                result_image_path = os.path.join('static', 'images', 'result_' + unique_filename)
                im_rgb.save(result_image_path)

            # Render the HTML template with the result image path
            return render_template('pure.html', image_url=result_image_path, image_path=image_path, background_image_url="/static/pf.jpg")

@app.route('/live_feed_page')
def live_feed_page():
    return render_template('live_feed.html')

@app.route('/live_feed')
def live_feed():
    return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_live_frames():
    cap = cv2.VideoCapture(0)  # 0 represents the default webcam

    while True:
        success, frame = cap.read()

        if success:
            # Perform prediction on the frame using your YOLO model
            results = models(frame)
            annotated_frame = results[0].plot()

            # Convert the annotated frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame bytes as part of the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

    cap.release()

@app.route('/vidpred', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            unique_filename = generate_unique_filename(file.filename)
            video_path = os.path.join('static', 'images',unique_filename)
            file.save(video_path)
            
            return redirect(url_for('video_feed', video_path=video_path))
    
    return render_template('pure.html')

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = models(frame)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

    cap.release()
    os.remove(video_path)

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path', None)
    
    if video_path:
        return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return 'Error: No video file provided.'

def delete_images_after_delay():
    while True:
        time.sleep(86400)  # Wait 1 day
        image_folder = 'static/images'
        for filename in os.listdir(image_folder):
            file_path = os.path.join(image_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# Flask route to delete images after 2 minutes
@app.route('/delete', methods=['GET'])
def delete():
    threading.Thread(target=delete_images_after_delay).start()
    return jsonify({"message": "Images will be deleted continuously after 2 minutes."})


#print(f"To acces the Gloable link please click\n{public_url}")
if __name__ == '__main__':
    app.run(port=5000)
