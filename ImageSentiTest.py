from keras.models import load_model
import cv2
from contants import *
import os
import numpy as np


mymodel = load_model('sentiment_model.h5')
# image_file = r'my_images/'
# img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (48, 48))
# img = img.astype('float32')
# img = img.reshape(1, 48, 48, 1)
# pred = mymodel.predict(img)
# # img = img.reshape(28,28,1)
# cv2.imwrite("show.jpg",img)
# print(EMOTIONS[pred.argmax()])
# print(pred)


def format_image(image, vid=False):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is no face found in image
    if not len(faces) > 0:
        return None, None
    images = []
    if vid:
        max_are_face = faces[0]
        for face in faces:
            if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
                max_are_face = face
            # face to image
        face_coor = max_are_face
        image = image[face_coor[1]:(
            face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
        # Resize image to network size
        try:
            image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
        except Exception:
            print("[+} Problem during resize")
            return None, None
        return image, face_coor
    else:
        for face_coor in faces:
            new_image = image[face_coor[1]:(face_coor[1] + face_coor[2]),
                              face_coor[0]:(face_coor[0] + face_coor[3])]
            try:
                new_image = cv2.resize(
                    new_image, (48, 48), interpolation=cv2.INTER_AREA)
                images.append(new_image)
            except Exception:
                print("[+} Problem during resize")
                return None, None
        return images, faces


def image_to_tensor(image):
    tensor = np.asarray(image).reshape(1, 48, 48, 1)
    return tensor


def valid_model():
    files = os.listdir(valid_data)
    for file in files:
        if file.endswith('.jpg'):
            image_file = os.path.join(valid_data, file)
            image1 = cv2.imread(image_file, cv2.IMREAD_ANYCOLOR)
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            height, width = image.shape
            if height*width == 2304:
                cv2.imwrite(f"detected/{file}", image)
                tensor = image_to_tensor(image)
                result = mymodel.predict(tensor)
                print(file, EMOTIONS[result.argmax()])
            else:
                detected_faces, face_coors = format_image(image)
                if detected_faces is not None:
                    emotions = []
                    for i, face in enumerate(detected_faces):
                        cv2.imwrite(f"detected/{i}{file}", face)
                        image_file = os.path.join("detected", str(i)+file)
                        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                        tensor = image_to_tensor(image)
                        result = mymodel.predict(tensor)
                        text = EMOTIONS[result.argmax()]
                        emotions.append(text)
                        print(image_file, text)
                    for i, face_coor in enumerate(face_coors):
                        [x1, y1, w1, h1] = face_coor
                        cv2.rectangle(image1, (x1, y1),
                                      (x1 + w1, y1 + h1), (255, 0, 0), 2)
                        cv2.putText(
                            image1, emotions[i], (x1, y1), cv2.FONT_HERSHEY_PLAIN, h1//40, (0, 255, 0), w1//130)
                        cv2.imwrite(f"detected/orignal{file}", image1)
                        try:
                            os.remove(f"detected/{i}{file}")
                        except:
                            pass
                else:
                    print("No Faces Detected")


def video(showBox=True):
    feelings_faces = []
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread(
            './data/emojis/' + emotion + '.png', -1))
    cap = cv2.VideoCapture(1)

    emoji_face = []
    result = None

    while True:
        ret, frame = cap.read()
        detected_face, face_coor = format_image(frame, True)
        if showBox:
            if face_coor is not None:
                [x, y, w, h] = face_coor
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # if cv2.waitKey(1) & 0xFF == ord(' '):

        if detected_face is not None:
            cv2.imwrite('a.jpg', detected_face)
            tensor = image_to_tensor(detected_face)
            result = mymodel.predict(tensor)
        if result is not None:
            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(frame, emotion, (10, index * 20 + 20),
                            cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                              (255, 0, 0), -1)
                emoji_face = feelings_faces[np.argmax(result[0])]

            for c in range(0, 3):
                frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + \
                    frame[200:320, 10:130, c] * \
                    (1.0 - emoji_face[:, :, 3] / 255.0)
        cv2.imshow('face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


video()
