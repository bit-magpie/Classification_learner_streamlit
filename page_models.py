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
            # param_string = ["`{}`".format(p) for p in params]
            # st.write(" ".join(param_string))
        with col3:
            st.write(" ")
            acc_placeholders[id] = st.empty()
            # st.write("##### 100%")
            # st.button("Reults", key="btn"+id)
    
    return chkbox

def show_pbars(names, values):
    for name, val in zip(names, values):
        text = f"{name} - `{val * 100:.1f}%`"
        st.progress(val, text=text)

def get_cdistribution_train():
    dataset = st.session_state["Dataset"]
    split = st.session_state["sldSplit"]
    train, _ = dataset.get_process_data(train_split=split)
    
    classes = dataset.c_names
    vals = []
    for _,v in groupby(sorted(train[1])):
        vals.append(len([*v]) / len(train[1]))
        
    show_pbars(classes, vals)
    
def get_cdistribution_all():
    dataset = st.session_state["Dataset"]
    _, all_data = dataset.get_train_data()
    
    classes = dataset.c_names    
    vals = []
    
    for _,v in groupby(sorted(all_data)):
        vals.append(len([*v]) / len(all_data))
    
    show_pbars(classes, vals)
    
    # fig = px.bar(x=x, y=y, height=300)
    # event = st.plotly_chart(fig, key="cdist" , on_select="rerun")
    

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
    st.header("Model training")
    if "Dataset" in st.session_state:
        col1, _, col2 = st.columns([6,1,5])
        with col1:                 
            with st.container(border=False): 
                trnCol1, trnCol2 = st.columns([10,2])
                with trnCol1:
                    st.write("Select models to train")
                with trnCol2:
                    btnTrain = st.button("Train all", key="btnTrain", type="primary")
                
                with st.container(border=True, height=500):
                    tablulate_models()
                if btnTrain:
                    train_model()
        with col2:
            st.write("##### Common configurations")
            split = st.slider("Train-test split", 0.1, 0.9, 0.8, key="sldSplit")
            st.radio("Model selection", ["All", "All linear", "All non-linear", "Custom"])
            
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
        st.text("No dataset found. Please upload a CSV formatted data file.")
        if st.button("Go to upload page"):
            st.switch_page("page_start.py")

main()