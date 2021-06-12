# importing the required libraries
import cv2
import numpy as np


# accessing the secondary camera
cap = cv2.VideoCapture(1)


class_names_file = 'coco.names'
class_names = [] # names of the trained targets
with open(class_names_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
print(class_names)
print(len(class_names))                                              
    

model_config = 'yolov3-320.cfg'
model_weights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


confidence_threshold = 0.5
NMS_thershold = 0.3 # lower the value, lesser will be the number of overlapping boxes


def findObjects(outputs, img):
    
    height, width, channels = img.shape
    
    bounding_boxes = []
    class_ids = []
    list_confidence = []
    
    for output in outputs:
        for detetction in output:
            scores = detetction[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                w, h= int(detetction[2]*width), int(detetction[3]*height)
                x, y = int((detetction[0]*width)-w/2), int((detetction[1]*height)-h/2) 
                
                bounding_boxes.append([x, y, w, h])
                list_confidence.append(float(confidence))
                class_ids.append(class_id)
                
    # to filter out the overlapping boxes
    indices = cv2.dnn.NMSBoxes(bounding_boxes, list_confidence, confidence_threshold, NMS_thershold)
    for i in indices:
        i = i[0]
        [x, y, w, h] = bounding_boxes[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{class_names[class_ids[i]].upper()} | ACC - {int(list_confidence[i]*100)}%', 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
     
while True:
    
    _, img = cap.read()
    
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    
    layer_names = net.getLayerNames()
    
    # names of the 3 output layers of the YoloV3 model
    output_layer_names = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layer_names)
    
    '''
    print(outputs)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    '''
    
    findObjects(outputs, img)
    cv2.imshow('Image', img)
   
    # delay by 1ms
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break
    
     
     
    
    
    
 