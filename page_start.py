import streamlit as st
import pandas as pd
from data_functions import DataFile, sk_datasets

def get_dataset_df(dataset_loader):
    dataset = dataset_loader()
    X = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
    y = pd.Series(dataset['target'], name='target')
    target_names = {index: name for index, name in enumerate(dataset['target_names'])}
    y = y.map(target_names)
    df = pd.concat([y, X], axis=1)
    
    return df, dataset['feature_names']

def load_file_data(uploaded_file):
    if "Dataset" not in st.session_state:
        df = pd.read_csv(uploaded_file)
        dataset = DataFile(df=df, name=uploaded_file.name)
        dataset.target = df.columns[0]
        dataset.features = df.columns[1:]
        st.session_state["Dataset"] = dataset

def load_sk_dataset(df, key, feature_cols):
    dataset = DataFile(df=df, name=key)
    dataset.features = feature_cols
    dataset.target = "target" 
    st.session_state["Dataset"] = dataset

def show_exaplaination():
    st.write("##### Dataset structure")
    st.write("- Only CSV format is supported by this app.")
    st.write("- Make sure the CSV file includes a header row, as shown in the example.")
    st.write("- Including class names is recommended to make the visualizations more understandable.")
    st.write("- Ensure the file size is less than 200MB.")
    st.image("class_learn_exp.png")

def set_target():
    st.session_state["Dataset"].target = st.session_state["target"]

def feature_selection():
    st.write("##### Select features and target")
    if "Dataset" in st.session_state:
        dataset = st.session_state["Dataset"]
        headers =  dataset.df.columns
        
        target = st.selectbox("Target (y)", headers, key="target", on_change=set_target)
        
        st.text("Features (X)")
        with st.container(height=300):
            col1, col2 = st.columns(2)
            st.session_state["Dataset"].selection = []
            for i, feature in enumerate(headers):
                if feature != target:
                    if i%2 == 0:
                        with col1:
                            st.session_state["Dataset"].selection.append(st.checkbox(feature, value=True, key="feat"+str(i)))
                    else:
                        with col2:
                            st.session_state["Dataset"].selection.append(st.checkbox(feature, value=True, key="feat"+str(i)))
                    
# def select_data():    
#     if data_file is not None:
#         headers =  data_file.df.columns
        
#         target = st.selectbox("Target (y)", headers)
#         data_file.target = target
#         st.text("Features (X)")
                
#         for i, feature in enumerate(headers):
#             if feature != target:
#                 data_file.selection.append(st.checkbox(feature, value=True, key="feat"+str(i)))
        
#         if st.button("Next"):
#             st.switch_page("page_visualizer.py")           

def dataset_desc(key, value):
    with st.container(border=False):
        _, col1, col2 = st.columns([1, 7, 4], vertical_alignment='center')
        with col1:
            data = value()
            n_samples, n_features, n_classes = len(data["data"]), len(data["feature_names"]), len(data["target_names"])
            st.markdown(f"**{key} Dataset**")
            st.write(f"Instances: `{n_samples}` Features: `{n_features}` Classes: `{n_classes}`")
            st.html("<hr>")
        with col2:
            if st.button("Load", key="btn" + key):
                st.session_state["Dataset_loaded"] = False
                df, feature_cols = get_dataset_df(value)
                load_sk_dataset(df, key, feature_cols)
        
def main():    
    col1, col2 = st.columns([7, 5], gap="large")    
    with col1:
        st.header("Welcome to Classic Learner")
        # with st.container():
        st.markdown("##### Upload your dataset.")
        
        uploaded_file = st.file_uploader(label="Upload a CSV file", key="dataset_uploder", type=["csv"])
        if uploaded_file is not None:
            st.session_state["Dataset_loaded"] = False
            # del st.session_state["Dataset"]
            load_file_data(uploaded_file)
            # set_data.clear()
            # set_data(uploaded_file)
            # select_data()  

        st.markdown("##### Or use sample dataset.")

        with st.container(border=False, height=350):
            # df = None
            # btn_cols = st.columns(len(sk_datasets))
            
            for i, (key, value) in enumerate(sk_datasets.items()):                
                dataset_desc(key, value)
                # with btn_cols[i]:
                #     if st.button(key + " dataset"):
                #         df, feature_cols = get_dataset_df(value)                       
                #         load_sk_dataset(df, key, feature_cols)
                    
        if "Dataset" in st.session_state:
            btnNext = st.button("Load Dataset and Proceed", key="btnGoViz", use_container_width=True, type="primary")
            if btnNext:                
                st.session_state["Dataset"].set_features()
                st.session_state["Dataset_loaded"] = True
                dataset = st.session_state["Dataset"]
                st.session_state["n_samples"] = len(dataset.df)
                st.write(dataset.target)
                st.switch_page("page_visualizer.py")
                        
    with col2:
        st.header("")
        if "Dataset" in st.session_state:       
            tab1, tab2 = st.tabs(["Feature selection", "Data table"])
            with tab1:
                feature_selection()
            with tab2:
                st.dataframe(st.session_state["Dataset"].df, height=550)
        else:
            show_exaplaination()
            
main()
