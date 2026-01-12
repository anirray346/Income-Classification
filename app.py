import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Custom CSS for Fonts & Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Montserrat:wght@700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
}

h1, h2, h3 {
    font-family: 'Montserrat', sans-serif;
    color: #2c3e50;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    results_df = joblib.load('results.pkl')
    return model, scaler, encoders, results_df

try:
    model, scaler, encoders, results_df = load_artifacts()
except FileNotFoundError:
    st.error("Artifacts not found. Please run 'train_model.py' first.")
    st.stop()

# --- App Layout ---
st.title("💰 Income Classification Pro")
st.markdown("### Predict outcome classes with Machine Learning")

# Sidebar
st.sidebar.markdown("## 🔧 Input Parameters")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=17, max_value=90, value=30)
    workclass = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)
    education = st.sidebar.selectbox("Education", encoders['education'].classes_)
    marital_status = st.sidebar.selectbox("Marital Status", encoders['marital-status'].classes_)
    occupation = st.sidebar.selectbox("Occupation", encoders['occupation'].classes_)
    relationship = st.sidebar.selectbox("Relationship", encoders['relationship'].classes_)
    race = st.sidebar.selectbox("Race", encoders['race'].classes_)
    sex = st.sidebar.selectbox("Sex", encoders['sex'].classes_)
    hours_per_week = st.sidebar.number_input("Hours per Week", min_value=1, max_value=100, value=40)
    native_country = st.sidebar.selectbox("Native Country", encoders['native-country'].classes_)
    
    capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
    
    data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': 189778, 
        'education': education,
        'education-num': 10,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main Panel tabs
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Model Conclusion"])

with tab1:
    st.info("Adjust settings in the sidebar to predict income level.")
    st.write(input_df)

    if st.button("Predict Income"):
        # Preprocessing
        input_encoded = input_df.copy()
        
        # Strip whitespace from input if strings might have them (selectboxes use stripped classes so should be fine)
        # But let's be safe for object cols
        for col in input_encoded.select_dtypes(include=['object']).columns:
             input_encoded[col] = input_encoded[col].str.strip()

        for col, le in encoders.items():
            if col in input_encoded.columns and col != 'income':
                try:
                    input_encoded[col] = le.transform(input_encoded[col])
                except ValueError:
                    st.warning(f"Unknown category in {col}, using default.")
                    input_encoded[col] = 0
        
        expected_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        input_encoded = input_encoded[expected_cols]
        input_scaled = scaler.transform(input_encoded)
        
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.success(f"### Predicted Income: **>50K**")
        else:
            st.warning(f"### Predicted Income: **<=50K**")
            
        st.write(f"Confidence: **{np.max(prob)*100:.2f}%**")

with tab2:
    st.markdown("## 🏆 Model Performance Conclusion")
    st.write("We trained multiple models to find the best performer. Here are the results:")
    
    # Display Table
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score']), use_container_width=True)
    
    # Chart
    st.bar_chart(results_df.set_index('Model')[['Accuracy', 'F1 Score']])
    
    best_model_name = results_df.loc[results_df['Accuracy'].idxmax()]['Model']
    best_acc = results_df['Accuracy'].max()
    
    st.markdown(f"""
    <div class="metric-card">
        <h3>Best Performing Model</h3>
        <h2 style="color: #4CAF50;">{best_model_name}</h2>
        <p>Accuracy: {best_acc:.2%}</p>
    </div>
    """, unsafe_allow_html=True)
