from flask import *
from PIL import Image
from camera import VideoCamera
import os,sys
import matplotlib.pyplot as plt
import cv2
import imutils
from imutils.contours import sort_contours
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import argparse
import numpy as np

TEMPLATE_DIR = os.path.abspath('templates')
STATIC_DIR = os.path.abspath('static')
UPLOAD_FOLDER = 'upload'
html="Your code will be shown here!!!"
image=None
path=None

class tagDetectionModel(object):

    EMOTIONS_LIST = ["button","image","input","paragraph","title"]


    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        return tagDetectionModel.EMOTIONS_LIST[np.argmax(self.preds)]

print(STATIC_DIR)
# app = Flask(__name__) # to make the app run without any
app = Flask(__name__, static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
############################################################
@app.route('/', methods=['GET', 'POST'])
def index():
    global image,path
    if request.method == 'POST':
        
        test()
        code_output()
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def yieldImage(frame):
    global path
    while True:
        with open(path, 'rb') as f:
            img = f.read()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
    

###############################################################
@app.route('/video_feed')
def video_feed():
    global image
    if html=="Your code will be shown here!!!":
        return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    elif type(image)==type(None):
        return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        print(image.shape)
        return Response(yieldImage(image),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generateMetrics():
    global html
    return html

#################################################################
@app.route('/code_output')
def code_output():
    response = make_response(generateMetrics(), 200)
    response.mimetype = "text/plain"
    return response

####################################################################

        

######################################################################

@app.route('/test' , methods=['GET', 'POST'])
def test():
    global html, image, path
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'there is no image in form!'
        file1 = request.files['image']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        
        print(path)
        
        file1.save(path)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # perform edge detection, find contours in the edge map, and sort the
        # resulting contours from left-to-right

        edged = cv2.Canny(blurred, 30, 150)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="left-to-right")[0]
        chars = []

        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # filter out bounding boxes, ensuring they are neither too small
            # nor too large
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape
            dX = int(max(0, 70 - tW) / 2.0)
            dY = int(max(0, 70 - tH) / 2.0)
            
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0))
            padded = cv2.resize(padded, (75, 75))
            
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            
            chars.append((padded, (x, y, w, h)))
            
            if tW > tH:
                thresh = imutils.resize(thresh, width=70)
                # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=70)

        boxes = [b[1] for b in chars]
        chars = np.array([c[0] for c in chars], dtype="float32")

        labelNames = ["button","image","input","paragraph","title"]

        Model = tagDetectionModel("../ML/model_3.json", "../ML/model_weights_4.h5")

        style={"circle":[],"button":[],"title":[],"input":[],"paragraph":[],"image":[]}
        html="&lthtml&gt\n\t&lthead&gt\n\t\t&ltstyle&gt_&lt&gt_\n\t\t&lt/style&gt\n\t&lt/head&gt\n\t&ltbody&gt"
        styles=""
        for (x, y, w, h) in boxes:
            # find the index of the label with the largest corresponding
            # probability, then extract the probability and label
            fc = gray[y:y+h, x:x+w]
            try:
                roi = cv2.resize(fc, (75, 75))
            except:
                continue
            pred = Model.predict(roi[np.newaxis, :, :, np.newaxis])
            if pred=="circle":
                html+="\n\t\t&ltdiv class='{}'&gt&lt/div&gt".format("circle"+str(len(style["circle"])))
                styles+="\n\t\t."+"circle"+str(len(style["circle"]))+"{"+"\n\t\t\tposition: absolute;left:{};top:{};width:{};height:{};border-radius:100%;border:1px solid;".format(x,y,w,h)+"}"
                print(pred,end=", ")
                style["circle"].append((x,y,w,h))
            
            elif pred=="button":
                html+="\n\t\t&ltbutton class='{}'&gt{}&lt/button&gt".format("button"+str(len(style["button"])),"button"+str(len(style["button"])))
                styles+="\n\t\t."+"button"+str(len(style["button"]))+"{"+"\n\t\t\tposition: absolute;left:{};top:{};padding:5px;border:1px solid;".format(x,y)+"}"
                print(pred,end=", ")
                style["button"].append((x,y,w,h))
            
            elif pred=="input":
                html+="\n\t\t&ltinput class='{}'/&gt".format("input"+str(len(style["input"])))
                styles+="\n\t\t.input"+str(len(style["input"]))+"{\n\t\t\tposition: "+"absolute;left:{};top:{};padding:5px 10px;border:1px solid;".format(x,y)+"}"
                print(pred,end=", ")
                style["input"].append((x,y,w,h))
                
            elif pred=="paragraph":
                html+="\n\t\t&ltp class='{}'&gt{}&lt/p&gt".format("paragraph"+str(len(style["paragraph"])),"paragraph"+str(len(style["paragraph"])))
                styles+="\n\t\t.paragraph"+str(len(style["paragraph"]))+"{\n\t\t\t"+"position: absolute;left:{};top:{};width: {};height: {};padding:2px;border:1px solid;".format(x,y,w,h)+"}"
                print(pred,end=", ")
                style["paragraph"].append((x,y,w,h))
            
            elif pred=="image":
                html+="\n\t\t&ltimg class='{}' alt='{}'/&gt".format("image"+str(len(style["image"])),"image"+str(len(style["image"])))
                styles+="\n\t\t.image"+str(len(style["image"]))+"{\n\t\t\tposition: "+"absolute;left:{};top:{};width: {}; height: {};padding:2px;border:1px solid;".format(x,y,w,h)+"}"
                print(pred,end=", ")
                style["image"].append((x,y,w,h))
            elif pred=="title":
                html+="\n\t\t&lth2 class='{}'&gt{}&lt/h2&gt".format("title"+str(len(style["title"])),"title"+str(len(style["title"])))
                styles+="\n\t\t.title"+str(len(style["title"]))+"{"+"\n\t\t\tposition: "+"absolute;left:{};top:{};padding:2px;border:1px solid;".format(x,y)+"}"
                print(pred,end=", ")
                style["image"].append((x,y,w,h))
            # draw the prediction on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, pred, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            # show the image
        html+="\n\t&lt/body&gt\n&lt/html&gt"
        html=list(html.split("_"))
        styles+"\n\t\t&lt/styles&gt/n\t&lt/head&gt"
        styles=styles.replace(";",";\n\t\t\t")
        html[1]=styles
        html="\n".join(html)
        html.replace("<","&lt")
        html.replace(">","&gt")
        html="<pre>"+html+"</pre>"
        #print("\n\n\nYOUR CODE:\n\n",html)
        file=open("generated.html","w")
        file.write(html)
        file.close()
        path=path
        cv2.imwrite(path,image)
        
    else:
        print("Not Working")
    return '''
    <form method="post" enctype="multipart/form-data">
              <input id="imageinput" type="file" name="image" onchange="readUrl(this)">
              <button name="send" id = "sendbutton" >Send</button>
            </form>
            <hr>
    '''
######################################################################
    
@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

########################################################################
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
