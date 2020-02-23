"""
获得人脸数据

根据输入的名字在./image/trainfaces目录下面创建子目录，将本次采集的头像保存在该目录之下
使用opencv打开摄像头，获取头像
检测出人脸的区域，调整一下亮暗度，将图片保存
保存200张之后，采集结束，程序自动退出

"""

import os
import random
import numpy as np
import cv2

def createdir(*args):
    ''' create dir'''
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

IMGSIZE = 64

def getpaddingSize(shape):
    ''' get size to make image to be a square rect '''
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest]*4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()

def dealwithimage(img, h=64, w=64):
    ''' dealwithimage '''
    #img = cv2.imread(imgpath)
    top, bottom, left, right = getpaddingSize(img.shape[0:2])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img

def relight(imgsrc, alpha=1, bias=0):
    '''relight'''
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc

def getfacefromcamera(outdir):
    """
    根据输入的名字在./image/trainfaces目录下面创建子目录，将本次采集的头像保存在该目录之下
    使用opencv打开摄像头，获取头像
    检测出人脸的区域，调整一下亮暗度，将图片保存
    保存200张之后，采集结束
    :param outdir: 输出文件的路径
    :return:
    """
    createdir(outdir)
    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')
    n = 1
    img_num = 200   # 采集img_num张人脸图片
    while 1:
        if (n <= img_num):
            print('It`s processing %s image.' % n)
            # 读帧
            success, img = camera.read()

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
            for f_x, f_y, f_w, f_h in faces:
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                #could deal with face to train
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)

                cv2.putText(img, 'Collecting', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # 显示采集状态
                img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                n += 1
            cv2.imshow('img', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    name = input('please input yourename: ')
    getfacefromcamera(os.path.join('./image/trainfaces', name))


