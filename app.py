from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import time

from utils.image_processing import process_image
from model.crowd_model import predict_crowd



app = Flask(__name__)



# Upload folder

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER




# Allowed image formats

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}




def allowed_file(filename):

    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS






@app.route("/")
def home():

    return render_template("index.html")








@app.route("/upload", methods=["POST"])
def upload():


    # Get uploaded image

    image = request.files["image"]



    # Check file type

    if not allowed_file(image.filename):

        return "Only JPG, JPEG and PNG images are allowed"




    # Secure filename

    filename = secure_filename(image.filename)




    # Save image

    path = app.config["UPLOAD_FOLDER"] + "/" + filename

    image.save(path)



    print("Image saved:", path)




    # Image information

    width, height, channels = process_image(path)


    print("Image Width:", width)
    print("Image Height:", height)
    print("Channels:", channels)




    # AI prediction with timer

    start_time = time.time()


    result = predict_crowd(path)


    end_time = time.time()


    processing_time = round(end_time - start_time, 2)



    people = result["count"]

    level = result["level"]

    message = result["message"]

    confidence = result["confidence"]





    # Heatmap

    heatmap = "heatmap.png"






    return render_template(

        "result.html",

        image_name=filename,

        width=width,

        height=height,

        channels=channels,

        people=people,

        level=level,

        message=message,
        
        confidence=confidence,

        heatmap=heatmap,


        # AI Model Information

        model_name="CSRNet",

        framework="PyTorch",

        device="CPU",

        processing_time=processing_time

    )









if __name__ == "__main__":

    app.run(debug=True)