# /usr/bin/python3
import numpy as np
from contants import *
import tensorflow as tf


tfc = tf.compat.v1
tfc.disable_v2_behavior()


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


# def video(modelPath, showBox=False):
#     face_x = tfc.placeholder(tf.float32, [None, 2304])
#     y_conv = deepnn(face_x)
#     probs = tfc.nn.softmax(y_conv)

#     saver = tfc.train.Saver()
#     ckpt = tfc.train.get_checkpoint_state(modelPath)
#     sess = tfc.Session()
#     if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         print('Restore model sucsses!!'
#               '\nNOTE: Press SPACE on keyboard to capture face.')

#     feelings_faces = []
#     for index, emotion in enumerate(EMOTIONS):
#         feelings_faces.append(cv2.imread(
#             './data/emojis/' + emotion + '.png', -1))
#     video_captor = cv2.VideoCapture(0)

#     emoji_face = []
#     result = None

#     while True:
#         ret, frame = video_captor.read()
#         detected_face, face_coor = format_image(frame, True)
#         if showBox:
#             if face_coor is not None:
#                 [x, y, w, h] = face_coor
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         if cv2.waitKey(1) & 0xFF == ord(' '):

#             if detected_face is not None:
#                 cv2.imwrite('a.jpg', detected_face)
#                 tensor = image_to_tensor(detected_face)
#                 result = sess.run(probs, feed_dict={face_x: tensor})
#         if result is not None:
#             for index, emotion in enumerate(EMOTIONS):
#                 cv2.putText(frame, emotion, (10, index * 20 + 20),
#                             cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
#                 cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
#                               (255, 0, 0), -1)
#                 emoji_face = feelings_faces[np.argmax(result[0])]

#             for c in range(0, 3):
#                 frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + \
#                     frame[200:320, 10:130, c] * \
#                     (1.0 - emoji_face[:, :, 3] / 255.0)
#         cv2.imshow('face', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
