import streamlit as st
import pandas as pd
import data_functions
import learner_module
import numpy as np

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
            chkbox = st.checkbox("|", key=id, value=False)
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
            
        
        # df = pd.DataFrame.from_dict(learner_module.classification_algorithms)
        # df = df.T
        # df["selected"] = [True] * len(df)
        # st.data_editor(
        #     df[["long_name", "selected"]],
        #     column_config={
        #         "model": st.column_config.CheckboxColumn(
        #             "Select to train",
        #             help="Select models to train",
        #             default=True,
        #         )
        #     },
        #     disabled=["widgets"],
        #     hide_index=True,
        #     width=800
        # )
        
def train_model():
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 0, 1, 1])
    
    model_list = learner_module.classification_algorithms
    for k in model_list.keys():
        if st.session_state[k]:
            with acc_placeholders[k].container(): 
                with st.spinner("Training..."):
                    model = model_list[k]["function"]()
                    model.fit(X,y) 
                    accuracy = learner_module.get_metrics(model, X, y)
            
                st.write(f"#### {accuracy*100}%")

def main():    
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