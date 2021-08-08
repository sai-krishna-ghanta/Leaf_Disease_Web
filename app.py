import streamlit as st
from PIL import Image
from numpy import asarray
import numpy as np

def lab():
    from pathlib import Path
    p = Path("Dataset")
    dirs = p.glob("*")
    labels_dict = {}
    for i,folder_dir in enumerate(dirs):
        label = str(folder_dir).split("/")[-1][:]
        labels_dict[label] = i  
    print(labels_dict)
    p = Path("Dataset")
    dirs = p.glob("*")
    image_data = []
    labels = []
    for i,folder_dir in enumerate(dirs):
        label = str(folder_dir).split("/")[-1][:]
        for img_path in folder_dir.glob("*.JPG"):
            img =  Image.open(img_path)
            img = img.resize((32,32))
            from numpy import asarray
            img_array = asarray(img)
            image_data.append(img_array)
            labels.append(labels_dict[label])
        
    image_data = np.array(image_data, dtype='float32')/255.0
    labels = np.array(labels)

    print(image_data.shape, labels.shape)
    print(len(image_data))
    print(len(labels))
    M = image_data.shape[0]
    image_data = image_data.reshape(M,-1)
    return image_data,labels

def app():
    images = ["https://hmp.me/dovt","https://hmp.me/dovs","https://hmp.me/dovu","https://hmp.me/dovu"]
    st.image(
        images,
        width=150, # Manually Adjust the width of the image as per requirement
        )
    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:Teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Leaf Disease Detection</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)
    if st.button("Detect"):   
        
        input_data = []
        img = our_image.resize((32,32))
        data = asarray(img)
        input_data.append(data)
        input_data = np.array(input_data, dtype='float32')/255.0
        M = input_data.shape[0]
        input_data = input_data.reshape(M,-1)
        for i in range(0,1) :
            from sklearn import svm
            image_data, labels = lab()
            svm_classifier = svm.SVC(kernel='linear', C=1.0)
            svm_classifier.fit(image_data, labels)
            prediction = svm_classifier.predict([input_data[i]])
        if prediction == 0:
            var = "It might be bacterial spot or septoria spot"
            pre = " Use certified disease free seeds and plants."
            a = "Avoid areas that were planted with peppers or tomatoes during previous year. "
            b = "Spraying with a copper  fungicide. "
            c = "Prune plants to promote air circulation."
            d = "Do not use overhead irrigation."
        if prediction == 1:
            var = "It might be late Blight or Early Blight"
            pre = "Allow extra room between the plants, and avoid overhead watering."
            a = "Plant resistant cultivars"
            b = "maintain a sufficient level of potassium,"
            c = " use one of the following fungicides: chlorothalonil , copper fungicide, or mancozeb "
            d = ""

        if prediction == 2:
            var = "It is Healthy"
            pre = "Leaf is healthy maintain the health "
            a = ""
            b = ""
            c = ""
            d = ""
        st.text(var)
        st.markdown("""Preventions and Precautions : :""")
        st.text(pre)
        st.text(a)
        st.text(b)
        st.text(c)
        st.text(d)



    st.text("This is a minor service added to the farm with app")
    st.success("Only disease detection available in website")
    st.text("For more services contact us")
    if st.button("Architecture"):
        st.success("This was made by SVM(sklearn),Opencv-Python,Streamlit and Heroku")
        st.text("The service consits additionlly YOLOV3 leaf detector and Disease detection with openCV")
    if st.button("Github Repository"):
        link1 = '[GithubRepo](https://github.com/sai-krishna-ghanta/Leaf_disease_Detection)'
        st.markdown(link1, unsafe_allow_html=True)
    if st.button("About Developer and Contact Us"):
        link2 = '[Linkedin Profile](https://www.linkedin.com/in/ghanta-sai-krishna-320ab0211/)'
        st.markdown(link2, unsafe_allow_html=True)

app()