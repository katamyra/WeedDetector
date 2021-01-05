from imageai.Detection import ObjectDetection

detector = ObjectDetection()

model_path = "C:\\Users\\krish\\Downloads\\mask_rcnn_trained_weed_model.h5"
input_path = "C:\\Code\\Python\\Mask_RCNN\\imgs\\00000.jpg"
output_path = "C:\\Code\\Python\\Mask_RCNN\\imgout\\Rio.jpg"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])