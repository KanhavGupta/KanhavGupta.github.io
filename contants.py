import cv2

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful',
            'happy', 'sad', 'surprised', 'neutral']
# SET_MODE = 'test'
# ckpt_dir = './ckpt'
# train_data = './data/fer2013/fer2013.csv'
# valid_data = './my_images/'
show_box = True
