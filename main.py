''' 
Ho Chi Minh University of Technology (HCMUT) 
	Khoa Điện - Điện tử 
Luận văn tốt nghiệp: Nhận dạng ngôn ngữ ký hiệu bằng phương pháp học sâu  
					(Sign Language Recognition by deep learning)

	Nguyen Thanh Sang - 1612933
###
Contact:
github: https://github.com/thanhsang298
gmail: thanhsang98.nguyen@gmail.com
###
'''
# -*- coding: utf-8 -*-
from tkinter import Tk, RIGHT, LEFT, BOTH, X, filedialog, StringVar, FLAT, SUNKEN, GROOVE, RIDGE, RAISED
from tkinter.ttk import Frame, Button, Style, Entry, Label
import tkinter.font as TkFont
from tkinter.font import *
import tkinter as tk, threading
import imageio
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import tensorflow as tf

class Window(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='#b3b3b3')
        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("Viet Nam Sign Language Translator")
        self.font0 = TkFont.Font(self, size=12)
        self.font = TkFont.Font(self, size=14)
        self.style = Style()
        self.style.theme_use("clam")
        self.pack(fill = BOTH, expand = 1)
        self.inputfilepath = StringVar()
        self.inputvideofile = StringVar()
        self.outputfilepath = StringVar()
        self.outputvideofile = StringVar()
        self.display_sign = StringVar()
        self.display_sequence = StringVar()


        #input file path
        frame1 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        frame1.pack(fill=X)
        input_button = tk.Button(frame1, text = 'Input directory', bg='#b3b3b3', font=self.font,
                                 command = self.input_browser)
        input_button.pack(side=LEFT, padx=5, pady=5)
        self.inputfilepathText = Entry(frame1, textvariable = self.inputfilepath, font=self.font)
        self.inputfilepathText.pack(fill=X,padx=5, expand=True)

        #input video file
        frame2 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        frame2.pack(fill=X)
        inputvideo_button = tk.Button(frame2, text = 'Input video file', bg="#b3b3b3", font=self.font,
                                      command = self.inputvideo_browser)
        inputvideo_button.pack(side=LEFT, padx=5, pady=5)
        self.inputvideofileText = Entry(frame2, textvariable = self.inputvideofile, font=self.font)
        self.inputvideofileText.pack(fill=X,padx=5, expand=True)

        #output file path
        frame3 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        frame3.pack(fill=X)
        output_button = tk.Button(frame3, text = 'Output directory', bg="#b3b3b3", font=self.font,
                                  command = self.output_browser)
        output_button.pack(side=LEFT, padx=5, pady=5)
        self.outputfilepathText = Entry(frame3, textvariable = self.outputfilepath, font=self.font)
        self.outputfilepathText.pack(fill=X,padx=5, expand=True)

        #output video file
        frame9 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        frame9.pack(fill=X)
        outputvideo_button = tk.Button(frame9, text = 'Output video file', bg="#b3b3b3", font=self.font,
                                      command = self.outputvideo_browser)
        outputvideo_button.pack(side=LEFT, padx=5, pady=5)
        self.outputvideofileText = Entry(frame9, textvariable = self.outputvideofile, font=self.font)
        self.outputvideofileText.pack(fill=X,padx=5, expand=True)

        #Mediapipe & open video & reset button
        frame4= tk.Frame(self)
        frame4.pack(fill=X)
        inputvideo_button = tk.Button(frame4, text = "Open input video", bg="#b3b3b3", font=self.font,
                                      command=self.open_invideo)
        inputvideo_button.grid(row=0, column=0)
        mediapipe_button = tk.Button(frame4, text = "Hand Mediapipe Process", bg="#b3b3b3", font=self.font,
                                     command=self.hand_mediapipe)
        mediapipe_button.grid(row=0, column=1)
        outputvideo_button = tk.Button(frame4, text = "Open output video", bg="#b3b3b3", font=self.font,
                                       command=self.open_outvideo)
        outputvideo_button.grid(row=0, column=2)
        reset_button=tk.Button(frame4, text = "Reset", bg="#b3b3b3", font=self.font,
                                       command=self.reset)
        reset_button.grid(row=0, column=3)

        #Predict sign
        frame5 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        frame5.pack(fill=X)
        predict_button = tk.Button(frame5, text = "Predict sign", bg="#b3b3b3", font=self.font, command=self.sign_predict)
        predict_button.pack(side=LEFT, padx=5, pady=5)
        self.signText = Entry(frame5, textvariable = self.display_sign, font=self.font)
        self.signText.pack(fill=X,padx=5, expand=True)

        #Predict sequence
        frame6 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        frame6.pack(fill=X)
        sequence_button = tk.Button(frame6, text = "Predict sequence", bg="#b3b3b3", font=self.font,
                                    command=self.sequence_predict)
        sequence_button.pack(side=LEFT, padx=5, pady=5)
        self.sequenceText = Entry(frame6, textvariable = self.display_sequence, font=self.font)
        self.sequenceText.pack(fill=X,padx=5, expand=True)

        frame7 = Frame(self, relief=GROOVE, borderwidth=1)
        frame7.pack(fill=BOTH)
        self.my_label = tk.Label(frame7)
        self.my_label.pack()

        frame8 = tk.Frame(self)
        frame8.pack(fill=X)
        quit_button = tk.Button(frame8, text = 'Close', font=self.font0, command = self.close_window, bg="#b3b3b3")
        quit_button.pack(side=RIGHT, padx=5, pady=5)

    def show_directory_browser(self):
        self.directory = filedialog.askdirectory()
        return self.directory
    def show_videofile_browser(self):
    	#run code nhớ thay đổi đường dẫn của bạn nhé ^^ 
        init_dir = "/home/shayneysang98/HCMUT/Thesis/Sign-language-recognition-with-RNN-and-Mediapipe/"
        ftypes = [("mp4 files","*.mp4"),("all files","*.*")]
        self.filename = filedialog.askopenfilename(initialdir = init_dir,filetypes = ftypes)
        return self.filename

    def input_browser(self):
        directory = self.show_directory_browser()
        self.inputfilepath.set(directory)
    def output_browser(self):
        directory = self.show_directory_browser()
        self.outputfilepath.set(directory)

    def inputvideo_browser(self):
        file = self.show_videofile_browser()
        self.inputvideofile.set(file)
    def outputvideo_browser(self):
        file = self.show_videofile_browser()
        self.outputvideofile.set(file)

    def open_invideo(self):
        video_name = self.inputvideofile.get()
        video = imageio.get_reader(video_name)
        def stream(label):
            for image in video.iter_data():
                image = cv2.resize(image, (1200, 650))
                frame_image = ImageTk.PhotoImage(Image.fromarray(image))
                label.config(image=frame_image)
                label.image = frame_image
        thread = threading.Thread(target=stream, args=(self.my_label,))
        thread.daemon = 1
        thread.start()
    def open_outvideo(self):
        video_name = self.outputvideofile.get()
        video = imageio.get_reader(video_name)
        def stream(label):
            for image in video.iter_data():
                image = cv2.resize(image, (1200, 650))
                frame_image = ImageTk.PhotoImage(Image.fromarray(image))
                label.config(image=frame_image)
                label.image = frame_image
        thread = threading.Thread(target=stream, args=(self.my_label,))
        thread.daemon = 1
        thread.start()

    def reset(self):
        self.inputfilepathText.delete(first=0,last=180)
        self.inputvideofileText.delete(first=0,last=180)
        self.outputfilepathText.delete(first=0,last=180)
        self.outputvideofileText.delete(first=0, last=180)
        self.signText.delete(first=0, last=100)
        self.sequenceText.delete(first=0, last=100)


    def hand_mediapipe(self):
        cmd = 'GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_cpu \
        --calculator_graph_config_file=mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt'
        input_data_path = self.inputfilepathText.get()
        output_data_path = self.outputfilepathText.get()
        if input_data_path[-1] != '/':
            input_data_path = input_data_path+'/'
        if output_data_path[-1] != '/':
            output_data_path = output_data_path+'/'

        listfile = os.listdir(input_data_path)
        if not (os.path.isdir(output_data_path + "Relative/")):
            os.mkdir(output_data_path + "Relative/")
        if not (os.path.isdir(output_data_path + "Absolute/")):
            os.mkdir(output_data_path + "Absolute/")
        for file in listfile:
            if not (os.path.isdir(input_data_path + file)):  # ignore .DS_Store
                continue
            word = file + "/"
            fullfilename = os.listdir(input_data_path + word)
            if not (os.path.isdir(output_data_path + "_" + word)):
                os.mkdir(output_data_path + "_" + word)
            if not (os.path.isdir(output_data_path + "Relative/" + word)):
                os.mkdir(output_data_path + "Relative/" + word)
            if not (os.path.isdir(output_data_path + "Absolute/" + word)):
                os.mkdir(output_data_path + "Absolute/" + word)
            for mp4list in fullfilename:
                if ".DS_Store" in mp4list:
                    continue
                inputfilen = '   --input_video_path=' + input_data_path + word + mp4list
                outputfilen = '   --output_video_path=' + output_data_path + '_' + word + mp4list
                cmdret = cmd + inputfilen + outputfilen
                os.system(cmdret)

    def load_label(self):
        listfile = ['Cách ly', 'Cảm ơn', 'CoronaCovid19', 'Ho', 'Khẩu trang', 'Lây lan', 'Mọi người', 'Rửa tay', 'Sốt', 'Xà phòng']
        label = {}  # khởi tạo 1 dict
        count = 1
        for l in listfile:
            if "_" in l:
                continue
            label[l] = count
            count += 1
        return label

    ########################### Dự đoán 1 từ ##############################
    def load_data(self, dirname):
        listfile = os.listdir(dirname)
        X = []
        Y = []
        for file in listfile:
            if "_" in file:
                continue
            wordname = file
            textlist = os.listdir(dirname + wordname)
            for text in textlist:
                if "DS_" in text:
                    continue
                textname = dirname + wordname + "/" + text
                numbers = []
                with open(textname, mode='r') as t:
                    numbers = [float(num) for num in t.read().split()]
                    while numbers[0] == 0:
                        numbers = numbers[1:]
                    for i in range(len(numbers), 4200):
                        numbers.extend([0.000])
                landmark_frame = []
                row = 0
                for i in range(0, 35):
                    landmark_frame.extend(numbers[row:row + 84])
                    row += 84
                landmark_frame = np.array(landmark_frame)
                landmark_frame = landmark_frame.reshape(-1, 84)
                X.append(np.array(landmark_frame))
                Y.append(wordname)
        X = np.array(X)
        Y = np.array(Y)
        x_train = X
        x_train = np.array(x_train)
        return x_train, Y

    def sign_predict(self):
        output_dir = self.outputfilepathText.get()
        if output_dir[-1] != '/':
            output_dir = output_dir+'/'
        x_test, Y = self.load_data(output_dir)
        new_model = tf.keras.models.load_model('model.h5')
        labels = self.load_label()
        xhat = x_test
        yhat = new_model.predict(xhat)
        predictions = np.array([np.argmax(pred) for pred in yhat])
        print(predictions)
        rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
        print(rev_labels)
        result = rev_labels[predictions[0]]
        self.display_sign.set(result)

    def get_idx(self, numbers, split_idx):
        while (numbers[split_idx] != 0 or numbers[split_idx + 1] != 0 or numbers[split_idx + 2] != 0 or
                numbers[split_idx + 3] != 0
        or numbers[split_idx + 4] != 0 or numbers[split_idx + 5] != 0 or numbers[split_idx + 6] != 0 or
               numbers[split_idx + 7] != 0
        or numbers[split_idx + 8] != 0 or numbers[split_idx + 9] != 0 or numbers[split_idx + 10] != 0 or
               numbers[split_idx + 11] != 0
        or numbers[split_idx + 12] != 0 or numbers[split_idx + 13] != 0 or numbers[split_idx + 14] != 0 or
               numbers[split_idx + 15] != 0
        or numbers[split_idx + 16] != 0 or numbers[split_idx + 17] != 0 or numbers[split_idx + 18] != 0 or
               numbers[split_idx + 19] != 0
        or numbers[split_idx + 20] != 0 or numbers[split_idx + 21] != 0 or numbers[split_idx + 22] != 0 or
               numbers[split_idx + 23] != 0
        or numbers[split_idx + 24] != 0 or numbers[split_idx + 25] != 0 or numbers[split_idx + 26] != 0 or
               numbers[split_idx + 27] != 0
        or numbers[split_idx + 28] != 0 or numbers[split_idx + 29] != 0 or numbers[split_idx + 30] != 0 or
               numbers[split_idx + 31] != 0
        or numbers[split_idx + 32] != 0 or numbers[split_idx + 33] != 0 or numbers[split_idx + 34] != 0 or
               numbers[split_idx + 35] != 0
        or numbers[split_idx + 36] != 0 or numbers[split_idx + 37] != 0 or numbers[split_idx + 38] != 0 or
               numbers[split_idx + 39] != 0
        or numbers[split_idx + 40] != 0 or numbers[split_idx + 41] != 0 or numbers[split_idx + 42] != 0 or
               numbers[split_idx + 43] != 0
        or numbers[split_idx + 44] != 0 or numbers[split_idx + 45] != 0 or numbers[split_idx + 46] != 0 or
               numbers[split_idx + 47] != 0
        or numbers[split_idx + 48] != 0 or numbers[split_idx + 49] != 0 or numbers[split_idx + 50] != 0 or
               numbers[split_idx + 51] != 0
        or numbers[split_idx + 52] != 0 or numbers[split_idx + 53] != 0 or numbers[split_idx + 54] != 0 or
               numbers[split_idx + 55] != 0
        or numbers[split_idx + 56] != 0 or numbers[split_idx + 57] != 0 or numbers[split_idx + 58] != 0 or
               numbers[split_idx + 59] != 0
        or numbers[split_idx + 60] != 0 or numbers[split_idx + 61] != 0 or numbers[split_idx + 62] != 0 or
               numbers[split_idx + 63] != 0
        or numbers[split_idx + 64] != 0 or numbers[split_idx + 65] != 0 or numbers[split_idx + 66] != 0 or
               numbers[split_idx + 67] != 0
        or numbers[split_idx + 68] != 0 or numbers[split_idx + 69] != 0 or numbers[split_idx + 70] != 0 or
               numbers[split_idx + 71] != 0
        or numbers[split_idx + 72] != 0 or numbers[split_idx + 73] != 0 or numbers[split_idx + 74] != 0 or
               numbers[split_idx + 75] != 0
        or numbers[split_idx + 76] != 0 or numbers[split_idx + 77] != 0 or numbers[split_idx + 78] != 0 or
               numbers[split_idx + 79] != 0
        or numbers[split_idx + 80] != 0 or numbers[split_idx + 81] != 0 or numbers[split_idx + 82] != 0 or
               numbers[split_idx + 83] != 0):
            split_idx += 1
        return split_idx
    ########################### Chuỗi 2 hành động ##############################
    def split_list2(self, numbers):
        while numbers[0] == 0:
            numbers = numbers[1:]
        split_idx = self.get_idx(numbers, 0)
        number2 = numbers[split_idx:]
        number1 = numbers[:split_idx]
        while number2[0] == 0:
            number2 = number2[1:]
        return number1, number2

    def load_data2(self, dirname):
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
                    number1, number2 = self.split_list2(numbers)
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
        x1_train = np.array(X1)
        x2_train = np.array(X2)
        Y = np.array(Y)
        print(Y)
        return x1_train, x2_train, Y

    ######################### Chuỗi 3 hành động ##############################
    def split_list3(self, numbers):
        while numbers[0] == 0:
            numbers = numbers[1:]
        split_idx = self.get_idx(numbers, 0)
        number2 = numbers[split_idx:]
        number1 = numbers[:split_idx]
        while number2[0] == 0:
            number2 = number2[1:]
        split_idx1 = self.get_idx(number2, 0)
        number3 = number2[split_idx1:]
        number2 = number2[:split_idx1]
        while number3[0] == 0:
            number3 = number3[1:]
        return number1, number2, number3

    def load_data3(self, dirname):
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
                    number1, number2, number3 = self.split_list3(numbers)
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
        x1_train = np.array(X1)
        x2_train = np.array(X2)
        x3_train = np.array(X3)
        Y = np.array(Y)
        print(Y)
        return x1_train, x2_train, x3_train, Y

    def sequence_predict(self):
        dirname = self.outputfilepathText.get()
        if dirname[-1] != "/":
            dirname = dirname +"/"
        listfile = os.listdir(dirname)
        for file in listfile:
            if "_" in file:
                continue
            wordname = file
            textlist = os.listdir(dirname + wordname)
            for text in textlist:
                if "DS_" in text:
                    continue
                textname = dirname + wordname + "/" + text
                numbers = []
                with open(textname, mode='r') as t:
                    numbers = [float(num) for num in t.read().split()]
                    print("Do dai file txt ban dau: " + str(len(numbers)))
                    while numbers[0] == 0:
                        numbers = numbers[1:]
                    print("Do dai file txt luc sau: " + str(len(numbers)))
                    y = len(numbers)

        if y <= 8400:
            x1_test, x2_test, Y = self.load_data2(dirname)
        elif y>8400:
            x1_test, x2_test, x3_test, Y = self.load_data3(dirname)

        new_model = tf.keras.models.load_model('model.h5')
        labels = self.load_label()
        print(labels)

        y1hat = new_model.predict(x1_test)
        y2hat = new_model.predict(x2_test)
        if y > 8400:
            y3hat = new_model.predict(x3_test)
        predictions1 = np.array([np.argmax(pred) for pred in y1hat])
        predictions2 = np.array([np.argmax(pred) for pred in y2hat])
        if y > 8400:
            predictions3 = np.array([np.argmax(pred) for pred in y3hat])
        print("pre1 va pre2")
        print(predictions1)
        print(predictions2)
        if y > 8400:
            print(predictions3)
        rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
        print("rev_labels:")
        print(rev_labels)
        s1 = rev_labels[predictions1[0]]
        s2 = rev_labels[predictions2[0]]
        if y<=8400:
            result =  s1 + " " + s2
        elif y>8400:
            s3 = rev_labels[predictions3[0]]
            result =  s1 + " " + s2 + " " + s3
        self.display_sequence.set(result)

    def close_window(self):
        form.destroy()

if __name__ == '__main__':
    form = Tk()
    form.geometry("1200x1000")
    app = Window(form)
    form.mainloop()