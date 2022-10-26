import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from playsound import playsound

st.title('Driver Drowziness Detection')
st.sidebar.subheader('About')
st.sidebar.write('A computer vision system made with the help of opencv that can automatically detect driver drowsiness in a real-time video stream and then play an alarm if the driver appears to be drowsy.')

dir_path= (r'models')
model = load_model(dir_path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
st.header("Webcam Live Feed")
run = st.checkbox('Click to Run/Off the cam',value=True)
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(1)
Score = 0
eye_cond = 1

st.subheader('Rules')
st.write('The more focused you are on your ride, the lower your drowziness score')
st.write('Alarm clock sounds when score reaches 25')
st.markdown('To Stop the Alarm Just **Focus on Your Drive**')


while run:  
    col1,col2 = st.sidebar.columns(2)
    with col1:
        st.subheader('Score = ' + str(Score))
    with col2:
        pass
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height,width = frame.shape[0:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, scaleFactor= 1.2, minNeighbors=3)
    eyes= eye_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=1)
    frame2 = cv2.rectangle(frame, (0,height-50),(200,height),(0,0,0),thickness=cv2.FILLED)
    sc = st.empty()
    def on_update():
        data = getNewData()
        sc.text('Score :' + str(data))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0), thickness=3 )
    for (ex,ey,ew,eh) in eyes:
        # cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (255,0,0), thickness=5)
        # preprocessing steps
        eye= frame[ey:ey+eh,ex:ex+ew]
        eye= cv2.resize(eye,(80,80))
        eye= eye/255
        eye= eye.reshape(80,80,3)
        eye= np.expand_dims(eye,axis=0)
        # preprocessing is done now model prediction
        prediction = model.predict(eye)
        
        # if eyes are closed
        print(prediction)
        if prediction[0][0]>0.25:
            eye_cond=0
            Score=Score+1
            if(Score>25):
                try:
                    playsound('alarm.wav')
                except:
                    pass
            
        # if eyes are open  
        elif prediction[0][1]>0.75:
            eye_cond=1
            Score = Score-1
            if (Score<0):
                Score=0
    cv2.putText(frame,'Score'+str(Score),(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')
