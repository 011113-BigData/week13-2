# load the libraries
import datetime
from skimage import io
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import os
import hashlib
from waitress import serve
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template
from pymongo import MongoClient
from bson.objectid import ObjectId

# suppress all warnings (ignore unnecessary warnings msgs)
import warnings
warnings.filterwarnings("ignore")

# define the flask and template directory 
app = Flask(__name__,template_folder='templates', static_url_path='/static')
app.secret_key = os.urandom(24)

# change with your location
UPLOAD_FOLDER = os.getcwd()+'/static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# only allow image file
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# only image can be allowed 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 127.0.0.1 is the local mongodb address installed
client = MongoClient('mongodb://127.0.0.1:27017/')

# YOU SHOULD change '<<yourUSERNAME>>' with userSTUDENTID (for example: user22222)
db = client['<<yourUSERNAME>>'] #<<yourUSERNAME>>

# serve the index 
@app.route("/")
def index():
    # retrieve last 10 data
    last_data = retrieve_lastdata(10)
    
    return render_template('form.html', last_data=last_data)

# handle the form action
@app.route("/result", methods=["POST"])
def result():
    # size/number of clusters
    kvalue = int(request.form.get('kvalue'))
    print("kvalue =", kvalue)
    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            #rename the file name
            filename_withoutext = hashlib.sha256(str(secure_filename(file.filename)).encode('utf-8')).hexdigest()
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            filename = filename_withoutext+"."+file_extension
            
            #save the image file into images directory
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # reading the image
            ori_image = io.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # call the preprocess_img function
            preprocessed_img, rows, cols  = preprocess_img(ori_image)
            
            # run the image compression using k-means clustering algorithm
            compressed_img = image_compression(kvalue, preprocessed_img)
            
            # reshape the array to it's original shape (rows, cols, 3)
            compressed_img = compressed_img.reshape(rows, cols, 3)
            
            # Save and display output image:
            filename_compressed = filename_withoutext+"_compressed_"+str(kvalue)+"."+file_extension
            io.imsave(os.path.join(app.config['UPLOAD_FOLDER'], filename_compressed), compressed_img)
            
            original_img_size = os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            compressed_img_size = os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename_compressed))

            print("Original image size is:", original_img_size/1000, "KB")
            print("Compressed image size is:", compressed_img_size/1000, "KB")
            reduced_by = round(original_img_size/compressed_img_size,2)
            print("The size is reduced by", reduced_by,"times")
            
            # store the data into mongodb
            # new record data, formatted in json
            new_record = { 
                'original_file': filename,
                'compressed_file': filename_compressed,
                'original_size': original_img_size/1000, 
                'compressed_size': compressed_img_size/1000,
                'kvalue': kvalue,
                'reduced_by': reduced_by,
                'created_at': datetime.datetime.now()
            }
            save_to_mongodb(new_record)
    
            return render_template('result.html', data=new_record)

# handle the show action
@app.route("/show/<id>", methods=["GET"])
def show_result(id):
    print(id)
    # find the data based on _id
    data = retrieve_data_byid(id)
    
    return render_template('result.html', data=data)

# define pre-processing image function
def preprocess_img(image):
    # get the height and width of the image
    rows, cols = image.shape[0], image.shape[1]
    
    # Gives a new shape to an array without changing its data.
    # Docs: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    # reshape the image into a rows*cols with 3 columns array so that each row represents a pixel and the three columns represent the Red, Green, and Blue values.
    image = image.reshape(rows * cols, 3)
    
    return image, rows, cols

# define image_compression function
def image_compression(size, image):
    # size is the number of clusters
    kmeans = KMeans(n_clusters = size)
    # fit the data
    kmeans.fit(image)
    # Replace each pixel value with its nearby centroid:
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    # Clip (limit) the values in an array
    # Docs: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)
    
    return compressed_image


# function to save the new sentence into mongodb
def save_to_mongodb(new_record):
    #image_compression is the collection (table) name in our mongodb database
    image_compression = db['image_compression']

    # Insert one record
    image_compression.insert_one(new_record)
    
# function to retrieve last data from mongodb
def retrieve_lastdata(limit):
    #image_compression is the collection (table) name in our mongodb database
    image_compression = db['image_compression']
    # retrieve last 5 data (limit = 5)
    # sort by _id, -1 -> desc; 1 -> asc
    data = image_compression.find().sort("_id", -1).limit(int(limit))
        
    return data

# function to retrieve data by _id from mongodb
def retrieve_data_byid(id):
    #image_compression is the collection (table) name in our mongodb database
    image_compression = db['image_compression']
    data = image_compression.find_one({'_id':ObjectId(id)})
        
    return data

if __name__ == "__main__":
    '''
     # change the port number, available from 5200-5221 
     (there are 21 port slots, please choose one and post in the chat 
     so that other student can choose the available one)
    '''
    portNumber = 5201 # change this portnumber based on above slots
    hostAddress = '0.0.0.0' # public ip or change to '127.0.0.1' for localhost in your local computer
    
    print('The webapp can be accessed at', hostAddress+':'+str(portNumber))
    serve(app, host=hostAddress, port=portNumber)
    