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
Setup external Tool: QTDesign and PyUIC
```angular2html
 Setting -> Tool -> External Tools -> Name: QTDesigner, Program: [Project folder]\venv\Lib\site-packages\qt5_applications\Qt\bin\designer.exe, working directory: $ProjectFileDir$
 Setting -> Tool -> External Tools -> Name: PyUIC, Program: [Project folder]\venv\Scripts\pyuic5.exe, working directory: $ProjectFileDir$
```
Run main.py to launch GUI.
```angular2html
Select main.py on pycharm -> run
```
## Usage
After launch GUI of ArcFace, we have this interface.
![Image](https://github.com/s3lab-sectic/ai_face_recognition_system_python/blob/main/datasets/images/GUI.jpg)
## Elements

* Button:
    - Capture: 
        - We have to input both text input(name, id) and click Capture.
        - If exist face in frame and after 10 frames. When the red rectangle appear, it captures a frame.
        - The numbers of image of each staff are 5 images. 
        - After capturing 5 images of employee, it saves those images into folder with Employee' name.
      
    - Re-train: base on the face vector and label, train again the model.
    - Re-embedding: generate the vector for all face, do it after adding new employee.
    - Run: Start to recognize with camera stream.
    - Exit: Exit the apps.
    
* Widget:
    - Left widget: Capture widget for registering new employee. 
    - Right widget: Record widget for recognizing with camera stream.
    
* Text Input:
    - Name: Name of employee.
    - Id: Identification of employee.
