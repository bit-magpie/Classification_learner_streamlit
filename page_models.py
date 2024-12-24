import streamlit as st
import pandas as pd
import numpy as np
import data_functions
import learner_module
from learner_module import Learner


acc_placeholders = dict()

def list_params(params):
    param_string = ""
    for k, v in params.items():
        param_string += f"{k}:`{v}` "
    st.write(param_string)        

def model_element(id, name, params):
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 9, 2])
        with col1:
            chkbox = st.checkbox("|", key=id, value=True)
        with col2:            
            st.write("##### " + name)
            list_params(params)
            # param_string = ["`{}`".format(p) for p in params]
            # st.write(" ".join(param_string))
        with col3:
            st.write(" ")
            acc_placeholders[id] = st.empty()
            # st.write("##### 100%")
            # st.button("Reults", key="btn"+id)
    
    return chkbox

def tablulate_models():    
    model_list = learner_module.classification_algorithms
    
    for k,v in model_list.items():
        model_element(k, v["long_name"], v["parameters"])
        
def train_model():
    train, test = data_functions.data_file.get_process_data()
    
    model_list = learner_module.classification_algorithms
    for k in model_list.keys():
        if st.session_state[k]:
            with acc_placeholders[k].container(): 
                with st.spinner("Training..."):
                    model = Learner(k, model_list[k]["function"], model_list[k]["parameters"])
                    model.num_cls = len(data_functions.data_file.c_names)
                    model.set_train_test(train, test)
                    model.train_model()
                    model.eval_model()                    
                    
                    # model = model_list[k]["function"]()
                    # model.fit(X,y) 
                    # accuracy = learner_module.get_metrics(model, X, y) 
            
                st.write(f"#### {model.accuracy*100:.1f}%")
                learner_module.trained_models[k] = model

def main():
    _, col, _ = st.columns([2,8,2])
    with col:    
        st.header("Model training")    
        if data_functions.data_file is not None:
            with st.container(border=True): 
                col1, col2 = st.columns([10,2])
                with col1:
                    st.write("Selects models to train")
                with col2:
                    btnTrain = st.button("Train all", key="btnTrain", type="primary")
                
                tablulate_models()
                if btnTrain:
                    train_model()
        else:
            st.text("No dataset found. Please upload a CSV formatted data file.")
            if st.button("Go to upload page"):
                st.switch_page("page_start.py")

main()