# A Gender and Age Detection program by Mahesh Sawant (Modified)
import cv2
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(h / 150)), 8)
    return frameOpencvDnn, faceBoxes

# --- Argument parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

# --- Model files ---
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# --- Labels ---
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
           '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# --- Load models ---
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# --- Read image ---
image_path = args.image
if image_path is None:
    print("❌ No image provided. Use --image filename")
    exit()

frame = cv2.imread(image_path)
if frame is None:
    print(f"❌ Failed to load image: {image_path}")
    exit()

# --- Detect face and predict age/gender ---
padding = 20
resultImg, faceBoxes = highlightFace(faceNet, frame)

if not faceBoxes:
    print("⚠️ No face detected")
else:
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding):
                     min(faceBox[3]+padding, frame.shape[0]-1),
                     max(0, faceBox[0]-padding):
                     min(faceBox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                     MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        gender = genderList[genderNet.forward()[0].argmax()]
        print(f"🧑 Gender: {gender}")

        ageNet.setInput(blob)
        age = ageList[ageNet.forward()[0].argmax()]
        print(f"🎂 Age: {age[1:-1]} years")

        label = f"{gender}, {age}"
        cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the result
    cv2.imshow("Age and Gender Detection", resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
