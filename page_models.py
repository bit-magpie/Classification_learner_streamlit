import streamlit as st
import plotly.express as px
from itertools import groupby
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
        with col3:
            st.write(" ")
            acc_placeholders[id] = st.empty()
    
    return chkbox

def show_pbars(names, values):
    
    for name, val in zip(names, values):
        text = f"{name} - `{val * 100:.1f}%`"
        st.progress(val, text=text)

def get_cdistribution_train():
    dataset = st.session_state["Dataset"]
    split = st.session_state["sldSplit"]
    dataset.get_process_data(train_split=split)
    _, train = dataset.train_data
    
    classes = dataset.c_names
    vals = []
    
    for _,v in groupby(sorted(train)):
        vals.append(len([*v]) / len(train))
        
    show_pbars(classes, vals)
    
def get_cdistribution_all():
    dataset = st.session_state["Dataset"]
    _, all_data = dataset.get_train_data()    
    
    classes = dataset.c_names    
    vals = []
    
    for _,v in groupby(sorted(all_data)):
        vals.append(len([*v]) / len(all_data))
    
    show_pbars(classes, vals)    

def tablulate_models():    
    model_list = learner_module.classification_algorithms
    
    for k,v in model_list.items():
        if "n_components" in v["parameters"].keys():
            model_list[k]["parameters"]["n_components"] = len(st.session_state["Dataset"].c_names)
        model_element(k, v["long_name"], v["parameters"])
        
def train_model():    
    model_list = learner_module.classification_algorithms
    trained_models = dict()
    
    for k in model_list.keys():
        if st.session_state[k]:
            with acc_placeholders[k].container(): 
                with st.spinner("Training..."):
                    dataset = st.session_state["Dataset"]                    
                    model = Learner(k, model_list[k]["function"], model_list[k]["parameters"])
                    model.num_cls = len(dataset.c_names)
                    model.train_model(dataset.train_data)
                    model.eval_model(dataset.test_data)                    
                                
                st.write(f"#### {model.accuracy*100:.1f}%")
                trained_models[k] = model
    
    st.session_state["Trained_models"] = trained_models

def main():
    st.header("Model training")
    
    if st.session_state["Dataset_loaded"]:
        col1, col2 = st.columns([7, 5], gap='large')
        with col1:                 
            with st.container(border=False): 
                trnCol1, trnCol2 = st.columns([10,2])
                with trnCol1:
                    st.write("Select models to train")
                with trnCol2:
                    btnTrain = st.button("Train all", key="btnTrain", type="primary")
                
                with st.container(border=True, height=540):
                    tablulate_models()
                if btnTrain:
                    train_model()
        
        with col2:
            st.write("##### Common configurations")
            split = st.slider("Train-test split", 0.1, 0.9, 0.8, key="sldSplit")
            
            setCol1, setCol2, setCol3 = st.columns([6, 3, 3], vertical_alignment='center') 
            with setCol1:
                st.radio("Model selection", ["All", "All linear", "All non-linear", "Custom"])
                
            if "n_samples" in st.session_state:
                with setCol2:
                    train_percent = st.session_state["sldSplit"]
                    n_samples = st.session_state["n_samples"]
                    n_train = int(n_samples * train_percent)
                    st.metric(label="Training", value=f"{n_train}", delta= f"{train_percent*100:.1f} %")
                
                with setCol3:
                    st.metric(label="Testing", value=f"{n_samples - n_train}", delta= f"{(1 - train_percent)*100:.1f} %")
            
            st.write("##### Class distributions")
            plotCol1, plotCol2 = st.columns(2, border=True)
            
            with plotCol1:
                st.write("All data")
                
                with st.container(border=False, height=170):
                    get_cdistribution_all()
            
            with plotCol2:
                st.write("Train split") 
                
                with st.container(border=False, height=170):
                    get_cdistribution_train()               

    else:
        st.warning('No dataset found. Please upload a CSV formatted data file or load existing dataset and click the "Load Dataset and Proceed" button.')
        
        if st.button("Go to upload page"):
            st.switch_page("page_start.py")

main()