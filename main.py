import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


st.title("Exploring different Classifiers")
st.write("Made by Zeaan Pithawala")
st.write("""
    ## Select any dataset and then see which classifier is best!
""")
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris","Breast Cancer","Wine Dataset","Digits"))
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN","SVM","Random Forest"))
metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

def getDataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    if dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    if dataset_name == 'Wine Dataset':
        data = datasets.load_wine()
    if dataset_name == 'Digits':
        data = datasets.load_digits()
    x = data.data
    y = data.target
    return x,y

x,y = getDataset(dataset_name)
st.write("The shape of the dataset is",x.shape)
st.write("Number of Classes", len(np.unique(y)))

def add_parameter_ui(classifier_name):
    d = {}
    if classifier_name == 'KNN':
        K = st.sidebar.slider("K",1,15)
        d["K"] = K
    if classifier_name == "SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        d["C"] = C
    if classifier_name == "Random Forest":
        max_depth = st.sidebar.slider("Max Depth",2,15)
        n_estimators = st.sidebar.slider("Number of Estimators",1,100)
        d["Max Depth"] = max_depth
        d["Number of Estimators"] = n_estimators
    return d

d = add_parameter_ui(classifier_name)

def get_classifier(classifier_name,d):
    if classifier_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors = d['K'])
    if classifier_name == "SVM":
        classifier = SVC(C = d['C'])
    if classifier_name == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=d["Number of Estimators"], max_depth=d["Max Depth"], random_state = 1234)
    return classifier

classifier = get_classifier(classifier_name,d)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=1234)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test,y_pred)

def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(classifier, x_test, y_test)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

st.write("Classifier =",classifier_name)
st.write("Accuracy =",str(acc*100)+"%")

pca = PCA(2)
x_projected = pca.fit_transform(x)
x1 = x_projected[:,0]
x2 = x_projected[:,1]
fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
plot_metrics('Confusion Matrix')

st.write("If you want to know more about me, visit my Website or LinkedIn and feel free to connect with me!")
st.write("Website- https://zeaan.github.io/website/")
st.write("LinkedIn- https://www.linkedin.com/in/zeaan-pithawala/")
