import os
import demo
import tensorflow as tf
# from utils import *
from contants import *

tfc = tf.compat.v1
tfc.disable_v2_behavior()


def deepnn(x):
    x_image = tf.reshape(x, [-1, 48, 48, 1])
    # conv1
    W_conv1 = weight_variables([5, 5, 1, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # pool1
    h_pool1 = maxpool(h_conv1)
    # norm1
    # norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # conv2
    W_conv2 = weight_variables([3, 3, 64, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    h_pool2 = maxpool(norm2)

    # Fully connected layer
    W_fc1 = weight_variables([12 * 12 * 64, 384])
    b_fc1 = bias_variable([384])
    h_conv3_flat = tfc.reshape(h_pool2, [-1, 12 * 12 * 64])
    h_fc1 = tfc.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # Fully connected layer
    W_fc2 = weight_variables([384, 192])
    b_fc2 = bias_variable([192])
    h_fc2 = tfc.matmul(h_fc1, W_fc2) + b_fc2

    # linear
    W_fc3 = weight_variables([192, 7])
    b_fc3 = bias_variable([7])
    y_conv = tfc.add(tf.matmul(h_fc2, W_fc3), b_fc3)

    return y_conv


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variables(shape):
    initial = tfc.truncated_normal(shape, stddev=0.1)
    return tfc.Variable(initial)


def bias_variable(shape):
    initial = tfc.constant(0.1, shape=shape)
    return tfc.Variable(initial)


# def train_model(train_data):
#     fer2013 = input_data(train_data)
#     max_train_steps = 30001

#     x = tfc.placeholder(tf.float32, [None, 2304])
#     y_ = tfc.placeholder(tf.float32, [None, 7])

#     y_conv = deepnn(x)

#     cross_entropy = tfc.reduce_mean(
#         tfc.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#     train_step = tfc.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#     correct_prediction = tfc.equal(tfc.argmax(y_conv, 1), tf.argmax(y_, 1))
#     accuracy = tfc.reduce_mean(tfc.cast(correct_prediction, tf.float32))
#     with tfc.Session() as sess:
#         saver = tfc.train.Saver()
#         sess.run(tfc.global_variables_initializer())
#         for step in range(max_train_steps):
#             batch = fer2013.train.next_batch(8)
#             if step % 100 == 0:
#                 train_accuracy = accuracy.eval(feed_dict={
#                     x: batch[0], y_: batch[1]})
#                 print('step %d, training accuracy %g' % (step, train_accuracy))
#             train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#             if step + 1 == max_train_steps:
#                 saver.save(sess, './models/emotion_model',
#                            global_step=step + 1)
#             if step % 1000 == 0:
#                 print('*Test accuracy %g' % accuracy.eval(feed_dict={
#                     x: fer2013.validation.images, y_: fer2013.validation.labels}))


def image_to_tensor(image):
    tensor = np.asarray(image).reshape(-1, 2304) * 1 / 255.0
    return tensor


def test_one_file():
    x = tfc.placeholder(tf.float32, [None, 2304])
    y_conv = deepnn(x)
    probs = tfc.nn.softmax(y_conv)

    saver = tfc.train.Saver()
    ckpt = tf.train.get_checkpoint_state("")
    with tfc.Session() as sess:
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restore model sucsses!!')

        file = "orignal.jpg"

        if file.endswith('.jpg'):
            # image_file = os.path.join(validFile, file)
            image1 = cv2.imread(file, cv2.IMREAD_ANYCOLOR)
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            height, width = image.shape
            if height*width == 2304:
                cv2.imwrite(f"{file}", image)
                tensor = image_to_tensor(image)
                result = sess.run(probs, feed_dict={x: tensor})
                print(file, EMOTIONS[result.argmax()])
            else:
                detected_faces, face_coors = demo.format_image(image)
                if detected_faces is not None:
                    emotions = []
                    for i, face in enumerate(detected_faces):
                        cv2.imwrite(f"{i}{file}", face)
                        image_file = os.path.join(str(i)+file)
                        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                        tensor = image_to_tensor(image)
                        result = sess.run(probs, feed_dict={x: tensor})
                        text = EMOTIONS[result.argmax()]
                        emotions.append(text)
                        print(image_file, text)
                    for i, face_coor in enumerate(face_coors):
                        [x1, y1, w1, h1] = face_coor
                        cv2.rectangle(image1, (x1, y1),
                                      (x1 + w1, y1 + h1), (255, 0, 0), 2)
                        cv2.putText(
                            image1, emotions[i], (x1, y1), cv2.FONT_HERSHEY_PLAIN, w1//110, (0, 255, 0), w1//130)
                        try:
                            os.remove(f"{i}orignal.jpg")
                        except:
                            pass
                    cv2.imwrite("static/images/detected1.jpg", image1)

                    return image1

                else:
                    print("No Faces Detected")


# def valid_model(modelPath, validFile):
#     x = tfc.placeholder(tf.float32, [None, 2304])
#     y_conv = deepnn(x)
#     probs = tfc.nn.softmax(y_conv)

#     saver = tfc.train.Saver()
#     ckpt = tf.train.get_checkpoint_state(modelPath)
#     with tfc.Session() as sess:
#         print(ckpt.model_checkpoint_path)
#         if ckpt and ckpt.model_checkpoint_path:
#             saver.restore(sess, ckpt.model_checkpoint_path)
#             print('Restore model sucsses!!')

#         files = os.listdir(validFile)

#         for file in files:
#             if file.endswith('.jpg'):
#                 image_file = os.path.join(validFile, file)
#                 image1 = cv2.imread(image_file, cv2.IMREAD_ANYCOLOR)
#                 image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
#                 height, width = image.shape
#                 if height*width == 2304:
#                     cv2.imwrite(f"detected/{file}", image)
#                     tensor = image_to_tensor(image)
#                     result = sess.run(probs, feed_dict={x: tensor})
#                     print(file, EMOTIONS[result.argmax()])
#                 else:
#                     detected_faces, face_coors = demo.format_image(image)
#                     if detected_faces is not None:
#                         emotions = []
#                         for i, face in enumerate(detected_faces):
#                             cv2.imwrite(f"detected/{i}{file}", face)
#                             image_file = os.path.join("detected", str(i)+file)
#                             image = cv2.imread(
#                                 image_file, cv2.IMREAD_GRAYSCALE)
#                             tensor = image_to_tensor(image)
#                             result = sess.run(probs, feed_dict={x: tensor})
#                             text = EMOTIONS[result.argmax()]
#                             emotions.append(text)
#                             print(image_file, text)
#                         for i, face_coor in enumerate(face_coors):
#                             [x1, y1, w1, h1] = face_coor
#                             cv2.rectangle(image1, (x1, y1),
#                                           (x1 + w1, y1 + h1), (255, 0, 0), 2)
#                             cv2.putText(
#                                 image1, emotions[i], (x1, y1), cv2.FONT_HERSHEY_PLAIN, w1//110, (0, 255, 0), w1//130)
#                             cv2.imwrite(f"detected/orignal{file}", image1)
#                             try:
#                                 os.remove(f"detected/{i}{file}")
#                             except:
#                                 pass
#                     else:
#                         print("No Faces Detected")
