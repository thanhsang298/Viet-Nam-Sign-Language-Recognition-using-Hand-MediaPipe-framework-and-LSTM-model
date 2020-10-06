# Viet Nam Sign Language Recognition using Hand MediaPipe framework and LSTM model

Sign language with hand gesture recognition using Long Short Term Memory network (LSTM) with [MediaPipe Hand tracking](https://google.github.io/mediapipe/solutions/hands) on desktop (CPU)

This code is built upon [rabBit64's](https://github.com/rabBit64/Sign-language-recognition-with-RNN-and-Mediapipe)

Thank Google's MediaPipe team for great framework

### CUSTOMIZE:

- Using video input files instead of Webcam to train with video data
- Get hand landmark features for every frame per second (fps) per one video input per one word and make it into one txt file

## 1. Set up Hand MediaPipe framework

- Installing and building MediaPipe examples

You can see it details in [here](https://google.github.io/mediapipe/getting_started/getting_started.html)

- Modify MediaPipe

You must change 3 files in "modified_mediapipe" folder like step 1 of [rabBit64's](https://github.com/rabBit64/Sign-language-recognition-with-RNN-and-Mediapipe)

## 2. Create your own training data

Make **trainvideosset** for each sign language word in one folder. Run **build.py** file to get txt file and mp4 output videos with hand tracking. You must have at least 150 videos per one word (one sign) to train

```
python3 build.py --input_data_path=[INPUT_PATH] --output_data_path=[OUTPUT_PATH]

```

For example: *input_data_path=/.../trainvideosset/*  and *output_data_path=/.../traintxtset/* 

```
trainvideosset
|-- Cachly
|	|-- 01_05_01.mp4
|	|-- 01_05_02.mp4
|	|-- 01_05_03.mp4
|	...
|	|-- 01_05_20.mp4
|
|-- Camon
|	|-- 01_09_01.mp4
|	|-- 01_09_02.mp4
|	|-- 01_09_03.mp4
|	...
|	|-- 01_09_20.mp4
|...
```

The output path is initially an empty directory, and when the build is complete, mp4 and txt folders are extracted to your folder path

Created folder example:

```
traintxtset
|-- Absolute
|	|-- Cachly
|		|-- 01_05_01.txt
|		|-- 01_05_02.txt
|		|-- 01_05_03.txt
|		...
|		|-- 01_05_20.txt
|	...
||-- Relative
|	|-- Cachly
|		|-- 01_05_01.txt
|		|-- 01_05_02.txt
|		|-- 01_05_03.txt
|		...
|		|-- 01_05_20.txt
|	...
||-- _Cachly
|	|-- 01_05_01.mp4
|	|-- 01_05_02.mp4
|	|-- 01_05_03.mp4
|	...
|	|-- 01_05_20.mp4
|...
```
**Important**: Name the folder carefully as the folder name will be the label itself for the video data. (DO NOT use space bar or '_' to your folder name, ex *train_videos_set* or *train videos set*)


## 3. Training LSTM model

Open **model.ipynb** file on your Jupyter Notebook enviroment to train LSTM model. The model is saved as **model.h5** in the current directory.

## 4. Result

Run 
```
python3 main.py
```

Result is displayed in a GUI with Tkinter library

- Predict a sign

![alt](https://user-images.githubusercontent.com/51918703/94219042-09f78f00-ff10-11ea-9ead-3f321f7cebdb.png)


- Predict a sequence (this project just recognize a sequence with 2 or 3 continuous signs) 

![alt](https://user-images.githubusercontent.com/51918703/94219168-6064cd80-ff10-11ea-97d2-8a9ff96c20d7.png)

Watch [testing video](https://www.youtube.com/watch?v=sGEKajiANH0) for detail.

