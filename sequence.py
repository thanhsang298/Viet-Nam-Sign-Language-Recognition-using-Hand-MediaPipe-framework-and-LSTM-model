from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
import argparse

def get_idx(numbers, split_idx):
    while (numbers[split_idx] !=0 or numbers[split_idx+1] !=0 or numbers[split_idx+2] !=0 or numbers[split_idx+3] !=0
    or numbers[split_idx+4] !=0 or numbers[split_idx+5] !=0 or numbers[split_idx+6] != 0 or numbers[split_idx+7] !=0
    or numbers[split_idx+8] !=0 or numbers[split_idx+9] !=0 or numbers[split_idx+10] != 0 or numbers[split_idx+11] !=0
    or numbers[split_idx+12] !=0 or numbers[split_idx+13] !=0 or numbers[split_idx+14] != 0 or numbers[split_idx+15] !=0
    or numbers[split_idx+16] !=0 or numbers[split_idx+17] !=0 or numbers[split_idx+18] != 0 or numbers[split_idx+19] !=0
    or numbers[split_idx+20] !=0 or numbers[split_idx+21] !=0 or numbers[split_idx+22] != 0 or numbers[split_idx+23] !=0
    or numbers[split_idx+24] !=0 or numbers[split_idx+25] !=0 or numbers[split_idx+26] != 0 or numbers[split_idx+27] !=0
    or numbers[split_idx+28] !=0 or numbers[split_idx+29] !=0 or numbers[split_idx+30] != 0 or numbers[split_idx+31] !=0
    or numbers[split_idx+32] !=0 or numbers[split_idx+33] !=0 or numbers[split_idx+34] != 0 or numbers[split_idx+35] !=0
    or numbers[split_idx+36] !=0 or numbers[split_idx+37] !=0 or numbers[split_idx+38] != 0 or numbers[split_idx+39] !=0
    or numbers[split_idx+40] !=0 or numbers[split_idx+41] !=0 or numbers[split_idx+42] != 0 or numbers[split_idx+43] !=0
    or numbers[split_idx+44] !=0 or numbers[split_idx+45] !=0 or numbers[split_idx+46] != 0 or numbers[split_idx+47] !=0
    or numbers[split_idx+48] !=0 or numbers[split_idx+49] !=0 or numbers[split_idx+50] != 0 or numbers[split_idx+51] !=0
    or numbers[split_idx+52] !=0 or numbers[split_idx+53] !=0 or numbers[split_idx+54] != 0 or numbers[split_idx+55] !=0
    or numbers[split_idx+56] !=0 or numbers[split_idx+57] !=0 or numbers[split_idx+58] != 0 or numbers[split_idx+59] !=0
    or numbers[split_idx+60] !=0 or numbers[split_idx+61] !=0 or numbers[split_idx+62] != 0 or numbers[split_idx+63] !=0
    or numbers[split_idx+64] !=0 or numbers[split_idx+65] !=0 or numbers[split_idx+66] != 0 or numbers[split_idx+67] !=0
    or numbers[split_idx+68] !=0 or numbers[split_idx+69] !=0 or numbers[split_idx+70] != 0 or numbers[split_idx+71] !=0
    or numbers[split_idx+72] !=0 or numbers[split_idx+73] !=0 or numbers[split_idx+74] != 0 or numbers[split_idx+75] !=0
    or numbers[split_idx+76] !=0 or numbers[split_idx+77] !=0 or numbers[split_idx+78] != 0 or numbers[split_idx+79] !=0
    or numbers[split_idx+80] !=0 or numbers[split_idx+81] !=0 or numbers[split_idx+82] != 0 or numbers[split_idx+83]!=0):
        split_idx += 1
    return split_idx
########################### Chuỗi 2 hành động ##############################
def split_list2(numbers):
    while numbers[0]==0:
        numbers = numbers[1:]
    split_idx = get_idx(numbers,0)
    number2 = numbers[split_idx:]
    number1 = numbers[:split_idx]
    while number2[0]==0:
        number2 = number2[1:]
    return number1, number2

def load_data2(dirname):
    listfile = os.listdir(dirname)
    X1 = []
    X2 = []
    Y = []
    for file in listfile:
        wordname = file
        textlist = os.listdir(dirname + wordname)
        ###################### Xu ly txt file #######################
        for text in textlist:
            if "DS_" in text:
                continue
            textname = dirname + wordname + "/" + text
            numbers = []
            with open(textname, mode='r') as t:
                numbers = [float(num) for num in t.read().split()]
                number1, number2 = split_list2(numbers)
                print("Do dai file txt tu thu nhat: " + str(len(number1)))
                print("Do dai file txt tu thu hai: " + str(len(number2)))
                print("===================================")
                for i in range(len(number1), 4200):
                    number1.extend([0.0])
                for i in range(len(number2), 4200):
                    number2.extend([0.0])

            landmark_frame1 = []
            row1 = 0
            for i in range(0, 35):
                landmark_frame1.extend(number1[row1:row1 + 84])
                row1 += 84
            landmark_frame1 = np.array(landmark_frame1)
            landmark_frame1 = landmark_frame1.reshape(-1, 84)

            landmark_frame2 = []
            row2 = 0
            for i in range(0, 35):
                landmark_frame2.extend(number2[row2:row2 + 84])
                row2 += 84
            landmark_frame2 = np.array(landmark_frame2)
            landmark_frame2 = landmark_frame2.reshape(-1, 84)

            X1.append(np.array(landmark_frame1))
            X2.append(np.array(landmark_frame2))
            Y.append(wordname)
    ##################################################################
    x1_train = np.array(X1)
    x2_train = np.array(X2)
    Y = np.array(Y)
    print(Y)
    return x1_train, x2_train, Y

######################### Chuỗi 3 hành động ##############################
def split_list3(numbers):
    while numbers[0] == 0:
        numbers = numbers[1:]
    split_idx = get_idx(numbers,0)
    number2 = numbers[split_idx:]
    number1 = numbers[:split_idx]
    while number2[0] == 0:
        number2 = number2[1:]
    split_idx1 = get_idx(number2,0)
    number3 = number2[split_idx1:]
    number2 = number2[:split_idx1]
    while number3[0] ==0:
        number3 = number3[1:]
    return number1, number2, number3

def load_data3(dirname):
    listfile = os.listdir(dirname)
    X1 = []
    X2 = []
    X3 = []
    Y = []
    for file in listfile:
        wordname = file
        textlist = os.listdir(dirname + wordname)
        ###################### Xu ly txt file #######################
        for text in textlist:
            if "DS_" in text:
                continue
            textname = dirname + wordname + "/" + text
            numbers = []
            with open(textname, mode='r') as t:
                numbers = [float(num) for num in t.read().split()]
                number1, number2, number3 = split_list3(numbers)
                print("Do dai file txt tu thu nhat: " + str(len(number1)))
                print("Do dai file txt tu thu hai: " + str(len(number2)))
                print("Do dai file txt tu thu ba: " + str(len(number3)))
                print("===================================")

                for i in range(len(number1), 4200):
                    number1.extend([0.000])
                for i in range(len(number2), 4200):
                    number2.extend([0.000])
                for i in range(len(number3), 4200):
                    number3.extend([0.000])

            landmark_frame1 = []
            row1 = 0
            for i in range(0, 35):
                landmark_frame1.extend(number1[row1:row1 + 84])
                row1 += 84
            landmark_frame1 = np.array(landmark_frame1)
            landmark_frame1 = landmark_frame1.reshape(-1, 84)

            landmark_frame2 = []
            row2 = 0
            for i in range(0, 35):
                landmark_frame2.extend(number2[row2:row2 + 84])
                row2 += 84
            landmark_frame2 = np.array(landmark_frame2)
            landmark_frame2 = landmark_frame2.reshape(-1, 84)

            landmark_frame3 = []
            row3 = 0
            for i in range(0, 35):
                landmark_frame3.extend(number3[row3:row3 + 84])
                row3 += 84
            landmark_frame3 = np.array(landmark_frame3)
            landmark_frame3 = landmark_frame3.reshape(-1, 84)

            X1.append(np.array(landmark_frame1))
            X2.append(np.array(landmark_frame2))
            X3.append(np.array(landmark_frame3))
            Y.append(wordname)
    ##################################################################
    x1_train = np.array(X1)
    x2_train = np.array(X2)
    x3_train = np.array(X3)
    Y = np.array(Y)
    print(Y)
    return x1_train, x2_train, x3_train, Y

#prediction: lấy từng label trong file label.txt
def load_label():
    listfile=[]
    with open("label.txt",mode='r') as l:
        listfile=[i for i in l.read().split()]
    listfile = ['Cách ly', 'Cảm ơn', 'CoronaCovid19', 'Ho', 'Khẩu trang', 'Lây lan', 'Mọi người', 'Rửa tay', 'Sốt',
                'Xà phòng']
    label = {}  #khởi tạo 1 dict
    count = 1
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label
    

def main(dirname):
    listfile=os.listdir(dirname)
    for file in listfile:
        if "_" in file:
            continue
        wordname=file
        textlist=os.listdir(dirname+wordname)
        for text in textlist:
            if "DS_" in text:
                continue
            textname=dirname+wordname+"/"+text
            numbers=[]
            with open(textname, mode = 'r') as t:
                numbers = [float(num) for num in t.read().split()]
                print("Do dai file txt ban dau: " + str(len(numbers)))
                while numbers[0] == 0:
                    numbers = numbers[1:]
                print("Do dai file txt luc sau: " + str(len(numbers)))
                check = len(numbers)

    if check <= 10080:
        x1_test, x2_test, Y=load_data2(dirname)
    elif check > 10080:
        x1_test, x2_test, x3_test, Y=load_data3(dirname)

    new_model = tf.keras.models.load_model('model.h5')
    labels=load_label()
    print(labels)

    y1hat = new_model.predict(x1_test)
    y2hat = new_model.predict(x2_test)
    if check > 10080:
        y3hat = new_model.predict(x3_test)
    print("y1hat va y2hat: ")
    print(y1hat)
    print(y2hat)
    if check >10080:
        print(y3hat)

    predictions1 = np.array([np.argmax(pred) for pred in y1hat])
    predictions2 = np.array([np.argmax(pred) for pred in y2hat])
    if check > 10080:
        predictions3 = np.array([np.argmax(pred) for pred in y3hat])
    print("pre1 va pre2")
    print(predictions1)
    print(predictions2)
    if check>10080:
        print(predictions3)
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    print("rev_labels:")
    print(rev_labels)

    txtpath=dirname+"sequence.txt" 
    s=0
    with open(txtpath, "w") as f:
        f.write("true_label: ")
        f.write(Y[s])
        f.write("      ===      ")
        f.write("Predict sequence: ")
        for i in predictions1:
            f.write(rev_labels[i])
            f.write(" ")
        for i in predictions2:
            f.write(rev_labels[i])
            f.write(" ")
        if check>10080:
            for i in predictions3:
                f.write(rev_labels[i])
                f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Sign language with Mediapipe')
    parser.add_argument("--dirname",help=" ")
    args=parser.parse_args()
    dirname=args.dirname
    main(dirname)
