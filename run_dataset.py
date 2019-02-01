import tensorflow as tf
import scipy.misc
import model
import cv2
import csv
import numpy as np
from subprocess import call

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save_focal150/model.ckpt")
#saver.restore(sess, "save_square/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0


#modified by Yuanwei 20171224
xs_ = []
ys_ = []
def read_csv(filename):
    with open(filename, 'r') as f:
        lines_all = [ln.strip().split(",")[:] for ln in f.readlines()]
        del(lines_all[0]) # remove the head of the csv
        lines_all = map(lambda x: (x[5], np.float128(x[6])), lines_all)
        return lines_all

def getTestingData(filename):
    lines_all = read_csv(filename)
    for ln in lines_all:
        if ln[0].find('center')  != -1:
            xs_.append("/home/weiy/dataset/udacity-output/"+ln[0])
            ys_.append(ln[1])

getTestingData("/home/weiy/dataset/udacity-output/interpolated_test_shuffle.csv")
#getTestingData("testingdata/testing-torcs.csv")

#print("xs_:",xs_)
# print("ys_[0]:",ys_[0])
# print("ys_[1]:",ys_[1])
# print("ys_[2]:",ys_[2])
#end by Yuaniwei 20171224


i = 0
while(cv2.waitKey(10) != ord('q')):
    #modified by Yuanwei 20171224
    full_image = scipy.misc.imread(xs_[i], mode="RGB")
    image = scipy.misc.imresize(full_image, [66, 200]) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0]

    # full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB")
    # image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    # degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi

    # new_path_original = "NO"
    # index = xs_[i].find('centercropmirror')
            
    # if index== -1:
    #     new_path = xs_[i].replace('centercrop','center')
    #     new_path_original = new_path

    # if  new_path_original == "NO":
    #     i += 1
    #     continue

    with open("result/resultTestingUdacityDemoFocal150.csv", 'a' , newline='') as csvFile:
                writer = csv.writer(csvFile)
                #writer.writerow([degrees, ys_[i], new_path_original])
                writer.writerow([degrees, ys_[i]])
                #print("pre_steer:%f,  pre_speed:%f",pre_steer,  pre_speed)
                csvFile.close()
    #end by Yuanwei 20171224
    call("clear")
    print("Predicted steering angle: " + str(degrees) + " degrees")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
