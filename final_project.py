#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Linda Kim
# ### 12/10/22


# PACKAGES

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Read in the data, call the dataframe "s"
s = pd.read_csv("social_media_usage.csv")


# Define a function called clean_sm 
# Takes input x (if 1, x = 1, else 0), Return x
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return(x)


# Create a new dataframe called "ss"
# Use clean_sm to create target column sm_li
# The new dataframe should contain a target column called sm_li (binary)
# Drop any missing values. 

ss = pd.DataFrame({
    "income": np.where(s['income'] > 9, np.nan, s['income']),
    "education": np.where(s['educ2'] > 8, np.nan, s['educ2']),
    "parent": clean_sm(s['par']),
    "married": clean_sm(s['marital']),
    "female": np.where(s['gender'] == 2, 1, 0),
    "age": np.where(s['age'] > 98, np.nan, s['age']),
    "sm_li": clean_sm(s['web1h'])
})

ss = ss.dropna()


# Converting variables
ss2 = ss
ss2.income = ss2.income.convert_dtypes('category')
ss2.education = ss2.education.convert_dtypes('category')
ss2.age = ss2.age.convert_dtypes('integer')


# Create a target vector (y) and feature set (X)
y = ss2["sm_li"]
X = ss2[["income", "education", "parent", "married", "female", "age"]]


# Split the data into training and test sets. 
# Hold out 20% of the data for testing.
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility


# Instantiate a logistic regression model 
# Set class_weight to balanced. Fit the model with the training data.
log_reg = LogisticRegression(class_weight = 'balanced')
log_reg.fit(X_train, y_train)


# Evaluate the model using the testing data. 
# Use the model to make predictions
y_pred = log_reg.predict(X_test)



# Application: Main
st.markdown("# LinkedIn User Predictions")

st.markdown("### This app will predict whether a person is a LinkedIn user based on the parameters in the sidebar.")

st.markdown("##### Change the settings on the left and then click Predict to see the prediction and probability.")





# Application: Inputs
with st.sidebar:
    inc = st.selectbox("Income",
        options = ["Less than $10,000",
        "10 to under $20,000",
        "20 to under $30,000",
        "30 to under $40,000",
        "40 to under $50,000",
        "50 to under $75,000",
        "75 to under $100,000",
        "100 to under $150,000",
        "$150,000 or more"])
    edu = st.selectbox("Education",
        options = ["Less than high school",
        "High school incomplete",
        "High school graduate",
        "Some college, no degree",
        "Two-year associate degree from a college or university",
        "Four-year college or university degree/Bachelor’s degree ",
        "Some postgraduate or professional schooling",
        "Postgraduate or professional degree"])
    par = st.radio("Parent", ('Yes', 'No'), index = 0)
    mar = st.radio("Married", ('Yes', 'No'), index = 0)
    gen = st.radio("Gender", ('Female', 'Male', 'Other'), index = 0)
    age = st.number_input("Age", min_value = 18, max_value = 98)


# Input: Income
if inc == "Less than $10,000":
    inc = 1
elif inc == "10 to under $20,000":
    inc = 2
elif inc == "20 to under $30,000":
    inc = 3
elif inc == "30 to under $40,000":
    inc = 4
elif inc == "40 to under $50,000":
    inc = 5
elif inc == "50 to under $75,000":
    inc = 6
elif inc == "75 to under $100,000":
    inc = 7
elif inc == "100 to under $150,000":
    inc = 8
elif inc == "$150,000 or more":
    inc = 9

# Input: Education
if edu == "Less than high school":
    edu = 1
elif edu == "High school incomplete":
    edu = 2
elif edu == "High school graduate":
    edu = 3
elif edu == "Some college, no degree":
    edu = 4
elif edu == "Two-year associate degree from a college or university":
    edu = 5
elif edu == "Four-year college or university degree/Bachelor’s degree ":
    edu = 6
elif edu == "Some postgraduate or professional schooling":
    edu = 7
elif edu == "Postgraduate or professional degree":
    edu = 8

# Input: Parent
if par == 'Yes':
    par = 1
else:
    par = 0

# Input: Married
if mar == 'Yes':
    mar = 1
else:
    mar = 0

# Input: Gender
if gen == 'Female':
    gen = 1
else:
    gen = 0

# DataFrame for inputs
person = pd.DataFrame({
    "income": [inc],
    "education": [edu],
    "parent": [par],
    "married": [mar],
    "female": [gen],
    "age": [age]
})

# Predict based on inputs
prediction = log_reg.predict(person)

# Probability
probability = log_reg.predict_proba(person)

# Make the Probability a percentage
prob = (probability[0][1])
proba = "{:.0%}".format(prob)


# Predict Button
if st.button('Predict'):
    if prediction == 1:
        pred = "likely"
        st.write(f"This person is {pred} a LinkedIn user.")
        st.write(f"Probability that this person is a LinkedIn user: {proba}")
    else:
        pred = "not likely"
        st.write(f"This person is {pred} a LinkedIn user.")
        st.write(f"Probability that this person is a LinkedIn user: {proba}")

