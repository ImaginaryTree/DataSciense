from flask import Flask, render_template, Response
from animation import cartoonize
import cv2

app = Flask(__name__, template_folder='template')



# Open the video capture
#image = cv2.VideoCapture("http://192.168.43.1:8080/shot.jpg")
image = cv2.VideoCapture(0)

def gen_frame():
    while True:
        ret, frame = image.read()
        # Cartoonize the frame
        cartoon_frame = cartoonize(frame)
        # Convert the frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', cartoon_frame)
        # Yield the JPEG frame
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
              jpeg.tobytes() +
              b'\r\n'
             )

@app.route('/')
def index():
    #return 'Hello world!'
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frame(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(debug=True)
