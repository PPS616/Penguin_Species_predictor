import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

# Prediction function
def prediction( model , island , bill_length_mm , bill_depth_mm , flipper_length_mm , body_mass_g , sex ):
    p_spec_pred = model.predict( [[island , bill_length_mm , bill_depth_mm , flipper_length_mm , body_mass_g , sex]] )
    p_spec_pred = p_spec_pred[0]

    if p_spec_pred == 0:
        return 'Adelie'.upper()
    elif p_spec_pred == 1:
        return 'Chinstrap'.upper()
    elif p_spec_pred == 2:
        return 'Gentoo'.upper()

#Streamlit Design
st.sidebar.title('Display Data')

if st.sidebar.checkbox("Show raw data"):
    st.title(' Penguin Species Dataset')
    st.dataframe(df)

else:
    st.title('Penguin Species Predictor')
    st.write('-' * 40)

st.sidebar.write('-' * 20)

st.sidebar.title( 'Select Features')
bl = st.sidebar.slider('Bill length' , df['bill_length_mm'].min() , df['bill_length_mm'].max() , 0.1 )
bd = st.sidebar.slider('Bill depth' , df['bill_depth_mm'].min() , df['bill_depth_mm'].max() , 0.1)
fl = st.sidebar.slider('Flipper length' , df['flipper_length_mm'].min() , df['flipper_length_mm'].max() , 0.1 )
bm = st.sidebar.slider('Body mass' , df['body_mass_g'].min() , df['body_mass_g'].max() , 0.1)

s = st.sidebar.selectbox('Sex', (0,1) )
i = st.sidebar.selectbox('Island', (0,1,2) )


st.sidebar.write('-' * 20)

st.sidebar.subheader('Choose Classifier')
classifier = st.sidebar.selectbox('Classifier' , ( 'Support Vector Machine' , 'Logistic Regression' , 'Random Forest Classifier') )

if classifier == 'Support Vector Machine':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    kernel_input = st.sidebar.radio("Kernel", ("linear", "rbf", "poly"))
    gamma_input = st. sidebar.number_input("Gamma", 1, 100, step = 1)

    if st.sidebar.button('Classify'):
        st.subheader("Support Vector Machine")
        svc_model = SVC(C=c_value, kernel=kernel_input, gamma=gamma_input)
        svc_model.fit(X_train, y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = svc_model.score(X_test, y_test)
        pred = prediction(model = svc_model , island = i , bill_length_mm = bl, bill_depth_mm = bd, flipper_length_mm = fl , body_mass_g = bm, sex = s )
        st.write("Penguin predicted species is:", pred)
        st.write("Accuracy", accuracy.round(2))

elif classifier =='Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
    max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rf_clf= RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
        rf_clf.fit(X_train,y_train)
        accuracy = rf_clf.score(X_test, y_test)
        pred = prediction(model = rf_clf, island = i , bill_length_mm = bl, bill_depth_mm = bd, flipper_length_mm = fl , body_mass_g = bm, sex = s )
        st.write("Penguin predicted species is:", pred)
        st.write("Accuracy", accuracy.round(2))

else:
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 0.1, 100.0, step = 0.1)
    max_iter_input = st.sidebar.number_input("No. of iterations", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    if st.sidebar.button('Classify'):
        st.subheader("Logistic Regression")
        lg_model= LogisticRegression(C = c_value , max_iter= max_iter_input)
        lg_model.fit(X_train,y_train)
        accuracy = lg_model.score(X_test, y_test)
        pred = prediction(model = lg_model, island = i , bill_length_mm = bl, bill_depth_mm = bd, flipper_length_mm = fl , body_mass_g = bm, sex = s )
        st.write("Penguin predicted species is:", pred)
        st.write("Accuracy", accuracy.round(2))