# coding=utf-8
import cv2

# Loading cat-face detector
cat_path = 'haarcascade_frontalcatface.xml'
face_cascade = cv2.CascadeClassifier(cat_path)


def read_img(img_name):
    '''Read the pictures, grayscale'''
    img = cv2.imread(img_name)
    return img


def recogn_face(img):
    # Cat face recognition
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.02,
        minNeighbors=3,
        minSize=(150, 150),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces


def face_rect():
    '''Face rectangle'''
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'Cat', (x, y - 7), 3, 1.2,
                    (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Cat?', img)
    c = cv2.waitKey(0)

def cut_img():
    '''Cut cat face'''
    img_name = 2
    while True:
        if img_name >= 220:
            break
        img = read_img('../images/yebi/' + str(img_name) + '.jpg')
        if img is not None:
            faces = recogn_face(img)
        else:
            img_name += 1
            continue
        cut_img = None
        # print(faces)
        if faces == ():
            img_name += 1
            continue
        for (x, y, w, h) in faces:
            cut_img = img[y: y + h, x: x + w]
        cv2.imwrite('traindata/yebi/face' + str(img_name) + '.jpg', cut_img)
        img_name += 1

if __name__ == '__main__':
    cut_face()
