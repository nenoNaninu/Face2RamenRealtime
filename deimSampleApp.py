import cv2
from model import cyclegan
import numpy as np
import tensorflow as tf
import time
tf.set_random_seed(19)
import argparse

cascade_path = './haarcascade_frontalface_default.xml'
#cascade_path = './haarcascade_frontalcatface.xml'
#cascade_path = 'haarcascade_frontalface_alt.xml'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='noodle', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000,
                    help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100,
                    help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False,
                    help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True,
                    help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50,
                    help='max size of image pool, 0 means do not use image pool')

args = parser.parse_args()

fpsTime = 0
count = 0
previosTime = time.time()

def printFps():
    global count
    count += 1
    if previosTime + 1 < time.time():
        print 'fps::'+str(count)
        global previosTime
        previosTime = time.time()
        count = 0
        return True

def main():
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(cascade_path)
    cv2.namedWindow('Cap', cv2.WINDOW_AUTOSIZE)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:

        model = cyclegan(sess, args)
        model.init(args)

        while True:
            flag = printFps()
            ret, img = cap.read()
            if ret == False:
                continue
            grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # timestamp= time.time()
            faceRects = cascade.detectMultiScale(grayImg)
            # print 'sess' + str(time.time() - timestamp)

            if len(faceRects) > 0:
                for rect in faceRects:
                    if rect[1] - 20 > 0 and rect[1] + rect[3] + 20 < img.shape[0] and rect[0] - 20 > 0 and rect[0] + \
                            rect[2] + 20 < img.shape[1]:
                        roiImg = img[rect[1] - 20:rect[1] + rect[3] + 20, rect[0] - 20:rect[0] + rect[2] + 20]
                        if roiImg.shape[0] == 0 or roiImg.shape[1] == 0:
                            continue
                        cv2.imshow('roiImg', roiImg)
                        cv2.waitKey(1)
                        convImg = model.conv(roiImg)
                        cv2.imshow('conv', convImg)
                        img[rect[1] - 20:rect[1] + rect[3] + 20, rect[0] - 20:rect[0] + rect[2] + 20] = convImg
                        cv2.rectangle(img,(rect[0]-20,rect[1]-20), (rect[0] + rect[2]+20,rect[1]+rect[3]+20),
                                      (255, 0, 0), 3)

            cv2.imshow('Cap', img)
            # if flag:
            #     global previosTime
            #     cv2.imwrite(str(previosTime)+'.png',img)

            if cv2.waitKey(1) >= 0:
                break


if __name__ == '__main__':
    main()
