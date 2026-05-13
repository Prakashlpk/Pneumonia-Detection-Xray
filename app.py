import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    r"D:\Pneumonia_Detection\pneumonia_densenet_model.keras"
)


# Page title
st.title("Chest X-Ray Pneumonia Detection")

st.write(
    "Upload a chest X-ray image to predict "
    "NORMAL or PNEUMONIA."
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded X-ray Image",
        use_container_width=True
    )

    # Preprocess image
    image = image.convert("RGB")

    image = image.resize((224,224))

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(image_array)

    probability = prediction[0][0]

    # Result
    if probability > 0.5:

        st.error("Prediction: PNEUMONIA")

        st.write(
            f"Confidence: {probability:.2%}"
        )

    else:

        st.success("Prediction: NORMAL")

        st.write(
            f"Confidence: {(1-probability):.2%}"
        )