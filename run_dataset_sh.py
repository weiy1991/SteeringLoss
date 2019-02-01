import tensorflow as tf
import scipy.misc
import model
import cv2
import csv
import numpy as np
from subprocess import call
import os
from math import sqrt

# add by YuanWei 20181112

def getSaveFile(dir, tag):
    save_dir = []
    save_files = os.listdir(dir)
    for file in save_files:
        if os.path.isdir(file) and not file.find(tag):
            save_dir.append(file)
    return save_dir

save_dir = getSaveFile('.', 'save')
print(save_dir)
input("waiting")

# end by YuanWei 20181112

sess = tf.InteractiveSession()
saver = tf.train.Saver()
#saver.restore(sess, "save_udacity/model.ckpt")
#saver.restore(sess, "save_focal01/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0


#modified by Yuanwei 20171224
xs_ = []
ys_ = []

# def read_csv(filename):
#     with open(filename, 'r') as f:
#         lines_all = [ln.strip().split(",")[:] for ln in f.readlines()]  # get all the lines of the training data
#         lines_all = map(lambda x: (x[0], x[1], np.float128(x[2:])), lines_all) # imagefile, outputs   #np.float128:precise:0.000000001
#         return lines_all

# def getTestingData(filename):
#     count = 0
#     lines_all = read_csv(filename)
#     for ln in lines_all:
#         count +=1
#         if count<15:
#             continue
#         count = 0
#         #print(ln)
#         xs_.append(ln[1])
#         ys_.append(ln[2][0])

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

print("xs_:",len(xs_))
#input("waiting")
# print("ys_[0]:",ys_[0])
# print("ys_[1]:",ys_[1])
# print("ys_[2]:",ys_[2])
#end by Yuaniwei 20171224


def getPerformance(target, prediction):
    #target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
    #prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]

    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])

    #print("Errors: ", error)
    #print(error)

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)#target-prediction之差平方 
        absError.append(abs(val))#误差绝对值

    #print("Square Error: ", squaredError)
    #print("Absolute Value of Error: ", absError)

    MSE = sum(squaredError) / len(squaredError)
    #print("MSE = ", sum(squaredError) / len(squaredError))#均方误差MSE

    RMSE = sqrt(sum(squaredError) / len(squaredError))
    #print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))#均方根误差RMSE

    MAE = sum(absError) / len(absError)
    #print("MAE = ", sum(absError) / len(absError))#平均绝对误差MAE

    predictionDeviation = []
    predictionMean = sum(prediction) / len(prediction)#target平均值
    for val in prediction:
        predictionDeviation.append((val - predictionMean) * (val - predictionMean))

    VAR = sum(predictionDeviation) / len(predictionDeviation)
    #print("Target Variance = ", sum(targetDeviation) / len(targetDeviation))#方差

    SD = sqrt(sum(predictionDeviation) / len(predictionDeviation))
    #print("Target Standard Deviation = ", sqrt(sum(targetDeviation) / len(targetDeviation)))#标准差

    return MSE, RMSE, MAE, VAR, SD

def writeToResult(name, MSE, RMSE, MAE, VAR, SD):
    with open("result/"+"UdacityTest"+".csv", 'a' , newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([name,MSE, RMSE, MAE, VAR, SD])
        csvFile.close()


for save_file in save_dir:
    print("save_file:", save_dir)
    saver.restore(sess, save_file+"/model.ckpt")

    target = []
    prediction = []
    i = 0
    while(cv2.waitKey(10) != ord('q') and i<len(xs_)):
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

        with open("result/"+save_file+".csv", 'a' , newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow([degrees, ys_[i]])
                    #print("pre_steer:%f,  pre_speed:%f",pre_steer,  pre_speed)
                    csvFile.close()
        #end by Yuanwei 20171224
        #call("clear")
        print("Predicted steering angle: " + str(degrees) + " degrees")
        cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
        #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
        #and the predicted angle
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        cv2.imshow("steering wheel", dst)

        target.append(ys_[i])
        prediction.append(degrees)

        i += 1

        

    MSE, RMSE, MAE, VAR, SD = getPerformance(target, prediction)
    writeToResult(save_file, MSE, RMSE, MAE, VAR, SD)

    print(MSE, RMSE, MAE, VAR, SD)
    #input("waiting")

    cv2.destroyAllWindows()




# i = 0
# while(cv2.waitKey(10) != ord('q') and i<len(xs_)):
#     #modified by Yuanwei 20171224
#     full_image = scipy.misc.imread(xs_[i], mode="RGB")
#     image = scipy.misc.imresize(full_image, [66, 200]) / 255.0
#     degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0]

#     # full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB")
#     # image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
#     # degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi

#     # new_path_original = "NO"
#     # index = xs_[i].find('centercropmirror')
            
#     # if index== -1:
#     #     new_path = xs_[i].replace('centercrop','center')
#     #     new_path_original = new_path

#     # if  new_path_original == "NO":
#     #     i += 1
#     #     continue

#     with open("result/resultTestingfocal01.csv", 'a' , newline='') as csvFile:
#                 writer = csv.writer(csvFile)
#                 writer.writerow([degrees, ys_[i]])
#                 #print("pre_steer:%f,  pre_speed:%f",pre_steer,  pre_speed)
#                 csvFile.close()
#     #end by Yuanwei 20171224
#     call("clear")
#     print("Predicted steering angle: " + str(degrees) + " degrees")
#     cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
#     #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
#     #and the predicted angle
#     smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
#     M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
#     dst = cv2.warpAffine(img,M,(cols,rows))
#     cv2.imshow("steering wheel", dst)
#     i += 1

# cv2.destroyAllWindows()
