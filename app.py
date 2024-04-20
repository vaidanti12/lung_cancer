import pickle
import streamlit as st

lung_cancer = pickle.load(open("lung_cancer_model.pkl", "rb"))

st.title("Lung Cancer Prediction using Machine Learning")

col1, col2, col3, col4 = st.columns(4)
arr_rate = [x for x in range(0, 3)]
with col1:
    GENDER = st.selectbox("GENDER", ["Male", "Female"])
    in_GENDER = 0 if GENDER == "Male" else 1

with col2:
    AGE = st.number_input("AGE")

with col3:
    SMOKING = st.selectbox("SMOKING", arr_rate)
    # SMOKING = st.number_input("SMOKING")

with col4:
    YELLOW_FINGERS = st.selectbox("YELLOW_FINGERS", arr_rate)

with col1:
    ANXIETY = st.selectbox("ANXIETY", arr_rate)

with col2:
    PEER_PRESSURE = st.selectbox("PEER_PRESSURE", arr_rate)

with col3:
    CHRONIC_DISEASE = st.selectbox("CHRONIC DISEASE", arr_rate)

with col4:
    FATIGUE = st.selectbox("FATIGUE", arr_rate)

with col1:
    ALLERGY = st.selectbox("ALLERGY", arr_rate)

with col2:
    WHEEZING = st.selectbox("WHEEZING", arr_rate)

with col3:
    ALCOHOL_CONSUMING = st.selectbox("ALCOHOL CONSUMING", arr_rate)

with col4:
    COUGHING = st.selectbox("COUGHING", arr_rate)

with col1:
    SHORTNESS_OF_BREATH = st.selectbox("SHORTNESS OF BREATH", arr_rate)

with col2:
    SWALLOWING_DIFFICULTY = st.selectbox("SWALLOWING DIFFICULTY", arr_rate)

with col3:
    CHEST_PAIN = st.selectbox("CHEST PAIN", arr_rate)

# code for Prediction
lung_cancer_result = " "

# creating a button for Prediction
if st.button("Lung Cancer Test Result"):
    lung_cancer_report = lung_cancer.predict(
        [
            [
                in_GENDER,
                SMOKING,
                YELLOW_FINGERS,
                ANXIETY,
                PEER_PRESSURE,
                CHRONIC_DISEASE,
                FATIGUE,
                ALLERGY,
                WHEEZING,
                ALCOHOL_CONSUMING,
                COUGHING,
                SHORTNESS_OF_BREATH,
                SWALLOWING_DIFFICULTY,
                CHEST_PAIN,
            ]
        ]
    )

    if lung_cancer_report[0] == 0:
        lung_cancer_result = "You have no Lung Cancer."
        st.success(lung_cancer_result)
    else:
        lung_cancer_result = "You have Lung Cancer."
        st.error(lung_cancer_result)
