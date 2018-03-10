import numpy as np
import cv2
import glob
from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def crop_img(img):
    w = img.shape[0]
    h = img.shape[1]
    cropped_imgs = []
    cropped_imgs.append(img[0:int(0.9*w),0:int(0.9*h),:])
    cropped_imgs.append(img[int(0.1*w):w,0:int(0.9*h),:])
    cropped_imgs.append(img[0:int(0.9*w),int(0.1*h):h,:])
    cropped_imgs.append(img[int(0.1*w):w,int(0.1*h):h,:])
    return cropped_imgs

def get_hog_features(img, orient, pix_per_cell, cell_per_block):
    channel_hog_features = lambda ch: hog(ch, orientations=orient,
                                          pixels_per_cell=(pix_per_cell, pix_per_cell),
                                          cells_per_block=(cell_per_block, cell_per_block),
                                          transform_sqrt=True,
                                          visualise=False, feature_vector=True)

    ch1_hog = channel_hog_features(img[:, :, 0])
    ch2_hog = channel_hog_features(img[:, :, 1])
    ch3_hog = channel_hog_features(img[:, :, 2])

    features = np.hstack((ch1_hog, ch2_hog, ch3_hog))
    return features


def get_features(img, orient, pix_per_cell, cell_per_block):
    hog_features = get_hog_features(img, orient, pix_per_cell, cell_per_block)
    return hog_features


def get_training_data(imgs, orient, pix_per_cell, cell_per_block):
    training_data = []
    for img in imgs:
        image = cv2.imread(img)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
        training_data.append(get_features(image, orient, pix_per_cell, cell_per_block))
    return training_data


def train_classifier(vehicle_images, non_vehicle_images, orient, pix_per_cell, cell_per_block):
    vehicle_X = get_training_data(vehicle_images, orient, pix_per_cell, cell_per_block)
    non_vehicle_X = get_training_data(non_vehicle_images, orient, pix_per_cell, cell_per_block)
    X = np.vstack((vehicle_X, non_vehicle_X))
    y = np.hstack((np.ones(len(vehicle_images)), np.zeros(len(non_vehicle_images))))
    X, y = shuffle(X, y, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LinearSVC()
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)
    print("svm accuracy is {}".format(accuracy_score(y_test, y_test_pred)))
    return clf

if __name__=='__main__':
    orient = 8
    pix_per_cell = 16
    cell_per_block = 2
    vehicle_images = glob.glob('/Users/wuyang/Downloads/self_driving_car/CarND-Vehicle-Detection/vehicles/**/*.png')
    non_vehicle_images = glob.glob('/Users/wuyang/Downloads/self_driving_car/CarND-Vehicle-Detection/non-vehicles/**/*.png')
    svc = train_classifier(vehicle_images, non_vehicle_images, orient=orient, pix_per_cell=pix_per_cell,
                           cell_per_block=cell_per_block)

    filename = 'svc_model.sav'
    pickle.dump(svc, open(filename, 'wb'))