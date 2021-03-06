```

```

# Face Recognition with InsightFace

Recognize and manipulate faces with Python and its support libraries.
The project uses [MTCNN](https://github.com/ipazc/mtcnn) for detecting faces, then applies a simple alignment for each detected face and feeds those aligned faces into embeddings model provided by [InsightFace](https://github.com/deepinsight/insightface). Finally, a softmax classifier was put on top of embedded vectors for classification task.

## Getting started

### Requirements

- Python 3.6
- Virtualenv
- python-pip
- mx-net
- tensorflow
- macOS, Linux or Windows 10

### Installing in Linux or MacOs

Check update:

```
sudo apt-get update
```

Install python:

```
sudo apt-get install python3.6
```

Install pip:

```
sudo apt install python3-pip
```

Most of the necessary libraries were installed and stored in `env/` folder, so what we need is installing `virtualenv` to use this enviroment.
Install virtualenv:

```
sudo pip3 install virtualenv virtualenvwrapper
```

### Installing in Windows 10

Install python 3.6 at [here](https://www.python.org/downloads/release/python-360/).
Clone repos at github:

```angular2html
git clone https://github.com/s3lab-sectic/ai_face_recognition_system_python.git
```

Install Pycharm community version
Run Pycharm and open project by open the code folder
Setting Python interpreter:

```angular2html
Setting -> Project: ABC -> Python Interpreter -> C:\Users\[USER NAME]\AppData\Local\Programs\Python\Python36 
```

Install requirements.txt, run terminal inside pycharm

```angular2html
pip install -r requirements.txt
```

## Usage

1. Resgiter face for person
   File get_faces_from_image.py in folder application
   ```python
   from get_faces_from_image import get_faces_from_image
   get_faces_from_image(imagePath="../datasets/test/001.jpg", nameFolder="binh")
   ```
2. Retrain model
   File train_softmax.py.py in folder application
   ```python
   from train_softmax import train
   status = train()
   print(status)
   ```
3. Recognizer people in image
   File recognizer_image.py in folder application
   ```python
   from recognizer_image import recognizer_image
   result = recognizer_image(pathImageIn="../datasets/test/002.jpg", pathImageOut="../datasets/test/img_test_2.jpg")
   print(result)
   ```
