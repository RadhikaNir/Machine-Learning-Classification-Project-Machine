import streamlit as st
import joblib
import pandas as pd

# Set the tab title
st.set_page_config("Deployment Project")

# Set the page title
st.title("Machine Failure Prediction Project")

# Set header
st.subheader("By Radhika Nirmalkar")

# Load the pipeline (data cleaning, preprocessing) and model
pre = joblib.load("pipeline.joblib")
model = joblib.load("model.joblib")
st.write("Expected columns:")
st.write(pre.feature_names_in_.tolist())

# Create Input boxes that takes input from the user 
ID = st.number_input("ID")
Product_ID = st.text_input("Product ID", value="L53257")
Type = st.selectbox("Type",options=['L','M','H'])
Air_Temperature = st.number_input("Ait Temperature [K]")
Process_Temperature = st.number_input("Process Temperature [K]")
Rotational_Speed = st.number_input("Rotational Speed [RPM]")
Torque = st.number_input("Torque [Nm]")
Tool_Wear = st.number_input("Tool Wear")
TWF = st.selectbox("TWF", options=[0,1])
HDF = st.selectbox("HDF", options=[0,1])
PWF = st.selectbox("PWF", options=[0,1])
OSF = st.selectbox("OSF", options=[0,1])
RNF = st.selectbox("RNF", options=[0,1])

# Include a button. After providing all the inputs, user will click on the button. The button should provide the necessary predictions
submit = st.button("Predict Machine Failure  Status")

if submit:
    data = {
        'Product ID':[Product_ID],
        'Type':[Type],
        'Air Temperature[K]':[Air_Temperature],
        'Process Temperature[K]':[Process_Temperature],
        'Rotational speed[rpm]':[Rotational_Speed],
        'Torque[Nm]':[Torque],
        'Tool wear[min]':[Tool_Wear],
        'TWF':[TWF],
        'HDF':[HDF],
        'PWF':[PWF],
        'OSF':[OSF],
        'RNF':[RNF]
    }
    # Convert above dictionary into dataframe first
    xnew = pd.DataFrame(data)
     try:
        # Apply the saved preprocessing pipeline
        xnew_pre = pre.transform(xnew)

        # Convert preprocessed output to a DataFrame
        expected_columns = [
            "con_Air temperature _K",
            "con_Process temperature _K",
            "con_Rotational speed _rpm",
            "con_Torque _Nm",
            "con_Tool wear _min"
        ]

        if isinstance(xnew_pre, pd.DataFrame):
            xnew_pre = xnew_pre.copy()
            xnew_pre.columns = expected_columns
        else:
            xnew_pre = pd.DataFrame(
                xnew_pre,
                columns=expected_columns
            )

        # Check for missing or infinite values
        if not np.isfinite(xnew_pre.to_numpy(dtype=float)).all():
            st.error(
                "Preprocessing generated missing or infinite values. "
                "Please check the saved preprocessing pipeline."
            )

        else:
            # Predict using the trained model
            preds = model.predict(xnew_pre)
    
    if preds[0]==1:
        op = 'Machine has Failed'
        # st.subheader('Machine has Failed')
    else:
        op = 'Machine has Not Failed'
        # st.subheader('Machine has Not Failed')
    
    st.subheader(op)
