import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import base64
import pandas as pd
import imageio as iio
import numpy as np
import cv2
import io as BytesIO
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# Set page configuration
icon = Image.open('web_logo2.png')
st.set_page_config(page_title="CAPSTONE PROJECT - 20BML0014 20BML0045",
                   layout="wide",
                   page_icon= icon)

# getting the working directory for application developed
working_dir = os.path.dirname(os.path.abspath(__file__))

# # Add css to make text bigger
st.markdown(
        """
        <style>
        textarea {
            font-size: 1.5rem !important;
            }
        input {
            font-size: 1.5rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
)


# loading the saved models

heart_disease_model = pickle.load(open('F_heart_disease_model.sav', 'rb'))

asthma_model = pickle.load(open('F_ASTHMA_disease_model.sav', 'rb'))

parkinson_model = tf.keras.models.load_model(('my_parkinson_model.h5'), custom_objects={'KerasLayer':hub.KerasLayer})

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# sidebar for navigation

with st.sidebar:
    logo = Image.open('logo (11).png')
    st.image(logo)
    selected = option_menu('Predictive Health Analytics using Machine Learning for Senior Citizens',

                           ['About us','Cardiovascular Disease Prediction (Arrhythmia & Stroke)',
                            'Parkinson\'s Disease Prediction', 'Asthma Disease Prediction', 
                            'Diabetes Prediction'],
                           menu_icon= 'clipboard-pulse',
                           icons=['bookmark','activity', 'person-walking', 'lungs-fill', 'capsule'],
                           default_index=0)


# About us Page
if selected == 'About us':

    # page title
    
    new_title = '<p style="font-family:sans-serif ; color:#122620; text-align: center; font-size: 45px;"><b> “Caring for Seniors: SeniorCare Enriching lives with Heart & Expertise” </b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write("\n\n")
    st.write("\n\n")    
    col1,col2 = st.columns(2, gap = 'large')
    with col2:     
        image = Image.open('eld (3).png')
        st.image(image, width=330) 
    with col1:
        st.write("\n\n")
        st.write("\n\n")
        st.write("\n\n")
        sub = '<p style="font-family:Taviraj; color:#E1A140; text-align: center;  font-size: 36px;"><b>Welcome to SeniorCare, <br> where cutting-edge technology meets compassionate care for our aging population. </b></p>'
        st.markdown(sub, unsafe_allow_html=True)    
    
    # st.write("This is <b>bold</b> text, and this is <em>italicized</em> text.")
    
    # progress_text = '<span style="font-size: 24px;">:orange[Operation in progress. Please wait.]</span>'
    # st.markdown(progress_text, unsafe_allow_html=True)

    # st.markdown("This text is :red[colored red], and this is **:blue[colored]** and bold.")
    st.write('\n\n')
    st.write('\n\n')
    st.write('\n\n')
    st.write('\n\n')
    new_title = '<p style="font-family:Taviraj; color:#3B0404; text-align: center; font-size: 42px; "><b>Empowering Health: What Sets Us Apart ?</b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    new_title = '<p style="font-family:Taviraj; color:#3B0404; text-align: center; font-size: 28px;"><b><em>“Providing simpler solutions to complex health issues, specifically designed to aid the elderly” </em></b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    new_title = '<p style="font-family:Taviraj; color:#914110; text-align: justify; font-size: 25px;">Our platform offers comprehensive assessments for a range of prevalent conditions including cardiovascular disease, diabetes, hypertension, and Parkinson\'s disease. Using advanced algorithms and medical expertise, our intuitive interface guides users through a personalized evaluation process, providing insightful results in moments. Whether you\'re seeking peace of mind or proactive health management, our goal is to empower you with reliable information and support. </p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    new_title = '<p style="font-family:Taviraj; color:#914110; text-align: justify; font-size: 25px;">Join us in taking proactive steps towards better health. Click on your left-side tabs and get started with your free diagnosis today. </p>'

    st.markdown(new_title, unsafe_allow_html=True)
        
    st.write('\n\n')
    st.write('\n\n')
    st.write('\n\n')
    new_title = '<p style="font-family:Taviraj; color:#3B0404; text-align: center; font-size: 45px; "><b>Mission </b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    new_title = '<p style="font-family:Taviraj; color:#3B0404; text-align: center; font-size: 28px;"><b><em>“Empowering Seniors Through Predictive Health Solutions” </em></b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    new_title = '<p style="font-family:Taviraj; color:#914110; text-align: justify; font-size: 25px;">Our mission is to bridge this divide by integrating predictive approaches into the daily lives of seniors, empowering them with the knowledge to proactively manage their health. Through our innovative platform, we offer early detection and monitoring for prevalent diseases such as  Arrhythmia, Stroke, Parkinson\'s Disease, Hypertension and Diabetes.  </p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    st.write('\n\n')
    st.write('\n\n')
    st.write('\n\n')
    st.write('\n\n')
    new_title = '<p style="font-family:Taviraj; color:#3B0404; text-align: center; font-size: 45px;"><b>Vision </b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    new_title = '<p style="font-family:Taviraj; color:#3B0404; text-align: center; font-size: 28px;"><b><em>“Transforming Senior Healthcare: Where Technology Meets Compassion for a Healthier Tomorrow”</em></b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    new_title = '<p style="font-family:Taviraj; color:#914110; text-align: justify; font-size: 25px;">Our vision is to redefine senior healthcare by providing accessible, user-friendly tools that enable timely intervention and improved quality of life. Join us on our journey to revolutionize elderly care, where technology and compassion converge to empower individuals and communities with proactive health management. </p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    
    st.write('\n\n')
    st.write('\n\n')
    st.write('\n\n')
    st.write('\n\n')
    new_title = '<p style="font-family:Taviraj; color:#3B0404; text-align: center; font-size: 40px;"><b>Creators\' Credits </b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    new_title = '<p style="font-family:Taviraj; color:#3B0404; text-align: justify; font-size: 26px;">SeniorCare owes its inception and success to the dedication and expertise of our talented team -</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap = "large")
    with col1:
        logo = Image.open('kee.jpeg')
        st.image(logo, width=200)
        new_title = '<p style="font-family:Taviraj; color:#3B0404; font-size: 25px;"><b> KEERTHANA B </b> </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        new_title = '<p style="font-family:Taviraj; color:#914110; font-size: 22px;"> Co-founder </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        new_title = '<p style="font-family:Taviraj; color:#914110; font-size: 20px;"> B.Tech. ECE with Specialization in Biomedical Engineering Undergraduate Student</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        new_title = '<p style="font-family:Taviraj; color:#914110; font-size: 20px;"> VIT Vellore, Tamil Nadu</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        new_title = '<p style="font-family:Taviraj; color:#914110; font-size: 20px;"> E-mail: keerthana.balaji2020@vitstudent.ac.in   </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        new_title = '<p style="font-family:Taviraj; color:#914110; font-size: 20px;"> LinkedIn: https://www.linkedin.com/in/keerthanabalaji2108/ </p>'
        st.markdown(new_title, unsafe_allow_html=True)
    
    
    with col2:            
        logo = Image.open('vish.jpeg')
        st.image(logo, width=200)
        new_title = '<p style="font-family:Taviraj; color:#3B0404; font-size: 25px;"><b> VISHALINI P </b> </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        new_title = '<p style="font-family:Taviraj; color:#914110; font-size: 22px;"> Co-founder </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        new_title = '<p style="font-family:Taviraj; color:#914110; font-size: 20px;"> B.Tech. ECE with Specialization in Biomedical Engineering Undergraduate Student </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        new_title = '<p style="font-family:Taviraj; color:#914110; font-size: 20px;"> VIT Vellore, Tamil Nadu </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        new_title = '<p style="font-family:Taviraj; color:#914110; font-size: 20px;"> E-mail: vishalini.p2020@vitstudent.ac.in   </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        new_title = '<p style="font-family:Taviraj; color:#914110; font-size: 20px;"> LinkedIn: https://www.linkedin.com/in/vishalini2709/  </p>'
        st.markdown(new_title, unsafe_allow_html=True)

    st.write('\n\n')
    st.write('\n\n')
    new_title = '<p style="font-family:Taviraj; color:#2f0404; text-align: justify; font-size: 28px;"><b>We are immensely glad in shaping SeniorCare into a platform that strives to make a meaningful impact in senior healthcare. </b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
     

###################################################################################################################


# Cardiac Disease Prediction Page
if selected == 'Cardiovascular Disease Prediction (Arrhythmia & Stroke)':

    # page title
    st.title('Cardiovascular Disease Prediction')
    st.subheader('Accuracy : 78%')
    

    
    ########################################
    ## 2 COLUMN SETUP 
    
    # st.text_area("Write some text")
    # st.text_input("Write some text")
    # st.number_input("Write some number")


    col1, col2 = st.columns(2)
    

    with col1:
        age = st.text_input('Age')

    with col2:
        # p = float(0)
        # q = float(1)
        sex = st.text_input('Gender (0: Female / 1: Male)')
           
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results \n\n\n(0: Normal / 1: having ST-T wave abnormality / 2: showing probable or definite left ventricular hypertrophy)')

    with col2:
        thalach = st.text_input('Maximum Heart Rate (Numeric value between 60 and 202)')

    with col1:
        exang = st.text_input('Exercise Induced Angina chest pain (0: NO / 1: YES)')

    with col2:
        oldpeak = st.text_input('ST depression induced by exercise')

    
   
    # code for Prediction
    heart_diagnosis = ''
    

    # creating a button for Prediction
    btn = st.button('Cardiovascular Disease Test Result')
    if (btn):
        st.markdown(
                   '<style> .streamlit-button.primary-button{visibility: hidden;} .streamlit-button.primary-button:after{content: "Clicked"; visibility: visible; position:relative;-webkit-tap-highlight-color: rgba(38,39,48,0);box-sizing: border-box;font-family: inherit;font-size: inherit;overflow: visible;text-transform: none;display: inline-flex;align-items: center;justify-content: center;font-weight: 400;border-radius: .25rem;margin: 0;line-height: 1.6;color: #262730;cursor: pointer;padding: .25rem .75rem;background-color: #fff;border: 1px solid #e6eaf1; left: 0;}</style>', unsafe_allow_html=True)

        user_input = [age, sex, restecg, thalach, exang, oldpeak]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])
        # Prediction and confidence scores
        
        confidence_scores = heart_disease_model.predict_proba([user_input])
        confidence_scores = "{:.2f}%".format(np.max(confidence_scores)*100)
        confidence_scores = str(confidence_scores)

        if heart_prediction[0] == 1:
            heart_diagnosis = 'You are SUSPECTED to have a Cardiovascular Disease \n\n CONFIDENCE SCORE : ' + confidence_scores
        else:
            heart_diagnosis = 'You are NOT SUSPECTED to have any Cardiovascular Disease \n\n CONFIDENCE SCORE : ' + confidence_scores

    st.success(heart_diagnosis)

######################################################################################
# Parkinson's Disease Prediction Page
if selected == 'Parkinson\'s Disease Prediction':
    # page title
    st.title('Parkinson\'s Disease Prediction')
    st.subheader('Accuracy : 73%')
    st.subheader('Steps : \n\n 1. Draw a waveform OR a spiral on a plain piece of paper as shown below and store as .png/.jpg image on your desktop. \n\n2. Upload the saved file below and get your results. \n\n')

  # Example image
    # col1, col2 = st.columns(2)
    

    # with col1:
    image = Image.open('eg1.png')
    st.image(image, caption='Example images') 
    
  #Browser upload of image
    file = st.file_uploader("Upload your hand-drawn waveform image here: ", type=["png","jpg"])
    show_file = st.empty()
    if file:
        show_file.image(file, caption='Your uploaded image')
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        input_image = opencv_image

        input_image_resize = cv2.resize(input_image, (224,224))

        input_image_scaled = input_image_resize/255

        image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])


   # code for Prediction
    park_diagnosis = ''

   # creating a button for Prediction

    if st.button('Parkinson\'s Disease Test Result'):

        park_prediction = parkinson_model.predict(image_reshaped)

        input_pred_label = np.argmax(park_prediction)
        
        confidence_scores = tf.nn.softmax(park_prediction[0])
        confidence_scores = "{:.2f}%".format(np.max(confidence_scores)*100)
        confidence_scores = str(confidence_scores)


        if input_pred_label == 0:
            park_diagnosis = 'You are SUSPECTED to have Parkinson\'s Disease \n\n CONFIDENCE SCORE : ' + confidence_scores
       
        else:
            park_diagnosis = 'You are NOT SUSPECTED to have Parkinson\'s Disease \n\n CONFIDENCE SCORE : ' + confidence_scores
    
    st.success(park_diagnosis)
    
######################################################################################
# Asthma Disease Prediction Page
if selected == 'Asthma Disease Prediction':

    # page title
    st.title('Asthma Disease Prediction')
    st.subheader('Accuracy : 51%') 
    
    col1, col2 = st.columns(2)
    

    with col1:
        Gender = st.text_input('Gender (0: Female / 1: Male)')

    with col2:
        Tiredness = st.text_input('Do you get tired very often ? (0: No / 1: Yes)')

    with col1:
        Dry_Cough = st.text_input('Do you have dry cough ? (0: No / 1: Yes)' )
        
    with col2:
        Difficulty_in_Breathing = st.text_input('Do you experience difficulty in breathing ?  (0: No / 1: Yes)' )
    
    with col1:
        Sore_Throat = st.text_input('Do you have a sore throat ?  (0: No / 1: Yes)')

    with col2:
        Pains = st.text_input('Do you experience throat or chest pains very often ?  (0: No / 1: Yes)')

    with col1:
        Nasal_Congestion = st.text_input('Do you have nasal congestion ? (0: No / 1: Yes)')

    with col2:
        Runny_Nose = st.text_input('Do you often get runny nose ? (0: No / 1: Yes)')

    with col1:
        Age_25_59 = st.text_input('Are you aged between 25 and 59 years old ? (0: No / 1: Yes)')

    with col2:
        Age_60 = st.text_input('Are you 60 or above years old ? (0: No / 1: Yes)')
        
   
    # code for Prediction
    asthma_diagnosis = ''

    # creating a button for Prediction

    if st.button('Asthma Test Result'):

        user_input = [Gender, Tiredness, Dry_Cough, Difficulty_in_Breathing, Sore_Throat, Pains, Nasal_Congestion, Runny_Nose, Age_25_59, Age_60]

        user_input = [float(x) for x in user_input]

        asthma_prediction = asthma_model.predict([user_input])

        # Prediction and confidence scores
        
        confidence_scores = asthma_model.predict_proba([user_input])
        confidence_scores = "{:.2f}%".format(np.max(confidence_scores)*100)
        confidence_scores = str(confidence_scores)

        if asthma_prediction[0] == 0:
            asthma_diagnosis = 'You are SUSPECTED to have Hypertension Disease \n\n CONFIDENCE SCORE : ' + confidence_scores
        else:
            asthma_diagnosis = 'You are NOT SUSPECTED to have Hypertension Disease \n\n CONFIDENCE SCORE : ' + confidence_scores

    st.success(asthma_diagnosis)
    
######################################################################################
# Diabetes Disease Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction')

    st.subheader('Accuracy : 83%')
    
    
    col1, col2 = st.columns(2)
    

    with col1:
        Age = st.text_input('Age')

    with col2:
        Gender = st.text_input('Gender (0: Female / 1: Male)')
        
    with col1:
        Polydipsia = st.text_input('(Condition - Polydipsia) Do you feel thirsty for prolonged period of time despite drinking adequate amount of water everyday ? \n\n(0: No / 1: Yes)' )
    
    with col2:
        sudden_weight_loss = st.text_input('Did you experience any sudden weightloss recently? \n\n(0: No / 1: Yes)')

    with col1:
        Polyphagia = st.text_input('(Condition - Polyphagia) Do you have an increased appetite / extreme hunger recently ? \n\n(0: No / 1: Yes)')

    with col2:
        visual_blurring = st.text_input('Do you experience visual blurring ? \n\n(0: No / 1: Yes)')

    with col1:
        Itching = st.text_input('Do you have itching sensation ? \n\n(0: No / 1: Yes)')
        
    with col2:
        delayed_healing = st.text_input('Do you experience delayed healing of injuries ? \n\n(0: No / 1: Yes)')

    with col1:
        partial_paresis = st.text_input('(Condition - Partial paresis) Do you find any muscle weakness or difficulty to move voluntarily ? \n\n(0: No / 1: Yes)')

    with col2:
        muscle_stiffness = st.text_input('Do you experience muscle stiffness ? \n\n(0: No / 1: Yes)')

    with col1:
        Alopecia = st.text_input('(Condition - Alopecia) Do you experience extreme hair loss on the scalp or the entire body lately ? \n\n(0: No / 1: Yes)')

    with col2:
        Obesity = st.text_input('Do you have obesity ? \n\n(0: No / 1: Yes)')
    
   
    # code for Prediction
    diabetes_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Age, Gender, Polydipsia, sudden_weight_loss, Polyphagia, visual_blurring, Itching, delayed_healing, partial_paresis, muscle_stiffness, Alopecia, Obesity]

        user_input = [float(x) for x in user_input]

        diabetes_prediction = diabetes_model.predict([user_input])
        
        # Prediction and confidence scores
        
        confidence_scores = diabetes_model.predict_proba([user_input])
        confidence_scores = "{:.2f}%".format(np.max(confidence_scores)*100)
        confidence_scores = str(confidence_scores)

        if diabetes_prediction[0] == 1:
            diabetes_diagnosis = 'You are SUSPECTED to have Diabetes Disease \n\n CONFIDENCE SCORE : ' + confidence_scores
        else:
            diabetes_diagnosis = 'You are NOT SUSPECTED to have Diabetes Disease \n\n CONFIDENCE SCORE : ' + confidence_scores

    st.success(diabetes_diagnosis)

############################################
#END
