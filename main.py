import streamlit as st
import av
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from sample_utils import get_ice_servers


class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        # Load the models and initialize the classifier
        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.classifier = load_model('Emotion_Detection.h5')
        self.class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img=self.detect_emotions(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def detect_emotions(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)
        # print(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = self.classifier.predict(roi)[0]
                label = self.class_labels[preds.argmax()]
                label_position = (x, y)
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(img, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        return img


# Streamlit UI setup
st.title("Real-Time Emotion Detection")
st.write("**This application detects emotions in real-time using your webcam or an uploaded image.**")
st.write('This works on 5 types of emotions :''Angry', 'Happy', 'Neutral', 'Sad', 'Surprise')
# Option to choose between Webcam or Image Upload (in sidebar)
option = st.sidebar.selectbox("**Choose input method:**", ("Webcam (WebRTC)", "Upload Image"))

if option == "Webcam (WebRTC)":
    webrtc_streamer(key="emotion-detection", 
                    mode=WebRtcMode.SENDRECV, 
                    rtc_configuration={
                        "iceServers": get_ice_servers(),
                        "iceTransportPolicy": "relay",
                    },
                    media_stream_constraints={
                        "video": True,
                        "audio": False,
                    },
                    video_processor_factory=EmotionDetector,
                    async_processing=True)
elif option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Load the image
        img = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))
        
        # Initialize the emotion detector
        detector = EmotionDetector()
        
        # Detect emotions
        processed_img = detector.detect_emotions(img)
        
        # Convert the processed frame back to an image format for display
        st.image(processed_img, channels="BGR")
