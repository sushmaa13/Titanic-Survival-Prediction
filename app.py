import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("titanic_model.pkl")

# Function to make predictions
def predict_survival(Pclass, Sex, SibSp, Parch, Embarked, FamilySize, AgeGroup, FareGroup):
    input_data = np.array([[Pclass, Sex, SibSp, Parch, Embarked, FamilySize, AgeGroup, FareGroup]])
    prediction = model.predict(input_data)[0]
    return "Survived" if prediction == 1 else "Did not survive"

# Streamlit UI
st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# User Inputs
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, step=1)
Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, step=1)
Embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
FamilySize = SibSp + Parch + 1
AgeGroup = st.selectbox("Age Group", ["Child (0-12)", "Teenager (13-18)", "Young Adult (19-35)", "Middle Aged (36-60)", "Senior (60+)"])
FareGroup = st.selectbox("Fare Group", ["Lowest", "Low", "High", "Highest"])

# Convert inputs to match model encoding
Sex = 0 if Sex == "Male" else 1
Embarked = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[Embarked]
AgeGroup = {"Child (0-12)": 0, "Teenager (13-18)": 1, "Young Adult (19-35)": 2, "Middle Aged (36-60)": 3, "Senior (60+)": 4}[AgeGroup]
FareGroup = {"Lowest": 0, "Low": 1, "High": 2, "Highest": 3}[FareGroup]

# Predict
if st.button("Predict Survival"):
    result = predict_survival(Pclass, Sex, SibSp, Parch, Embarked, FamilySize, AgeGroup, FareGroup)
    st.subheader(f"Prediction: {result}")
