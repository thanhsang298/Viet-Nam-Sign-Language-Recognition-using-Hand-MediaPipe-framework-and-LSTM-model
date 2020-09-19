from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from numpy import argmax
import argparse

def load_data(dirname):
    listfile=os.listdir(dirname)
    X = []
    Y = []
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
                while numbers[0] == 0:
                    numbers = numbers[1:]
                print("Chieu dai txt file: " + str(len(numbers)))
                for i in range(len(numbers),4200):
                    numbers.extend([0.000])

            landmark_frame=[]
            row=0
            for i in range(0,35):
                landmark_frame.extend(numbers[row:row+84])
                row += 84
            landmark_frame=np.array(landmark_frame)
            landmark_frame=landmark_frame.reshape(-1,84)
            X.append(np.array(landmark_frame))
            Y.append(wordname)
    X=np.array(X)
    Y=np.array(Y)
    print(Y)
    x_train = X
    x_train=np.array(x_train)
    return x_train,Y

#prediction: lấy từng label trong file label.txt
def load_label():
    #listfile=[]
    #with open("label.txt",mode='r') as l:
        #listfile=[i for i in l.read().split()]
    listfile = ['Cách ly', 'Cảm ơn', 'CoronaCovid19', 'Ho', 'Khẩu trang', 'Lây lan', 'Mọi người', 'Rửa tay', 'Sốt', 'Xà phòng']
    label = {}  #khởi tạo 1 dict
    count = 1
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label
    
def main(output_data_path):
    output_dir=output_data_path
    x_test,Y=load_data(output_dir)
    new_model = tf.keras.models.load_model('model.h5')
    labels=load_label()
    print(labels)

    xhat = x_test
    yhat = new_model.predict(xhat)

    predictions = np.array([np.argmax(pred) for pred in yhat])
    print(predictions)
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    print(rev_labels)
    s=0
    count = 0
    txtpath=output_data_path+"result.txt"
    with open(txtpath, "w") as f:
        for i in predictions:
            f.write("true_label: ")
            f.write(Y[s])
            f.write(" === ")
            f.write("predict_label: ")
            f.write(rev_labels[i])
            f.write("\n")
            if rev_labels[i] == Y[s]:
                count+=1
            s+=1
    print("So luong tu du doan trung: " + str(count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Sign language with Mediapipe')
    parser.add_argument("--output_data_path",help=" ")
    args=parser.parse_args()
    output_data_path=args.output_data_path
    main(output_data_path)
