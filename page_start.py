import streamlit as st
import pandas as pd
from data_functions import DataFile
import data_functions

@st.cache_resource
def set_data(uploaded_file):
    data_functions.data_file = DataFile(upload_file=uploaded_file)
    
def set_features():
    data_functions.data_file.set_features()

def get_dataset_df(dataset_loader):
    dataset = dataset_loader()
    X = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
    y = pd.Series(dataset['target'], name='target')
    target_names = {index: name for index, name in enumerate(dataset['target_names'])}
    y = y.map(target_names)
    df = pd.concat([X, y], axis=1)
    return df, dataset['feature_names']

def select_data():    
    if data_functions.data_file is not None:
        headers =  data_functions.data_file.df.columns
        
        target = st.selectbox("Target (y)", headers)
        data_functions.data_file.target = target
        st.text("Features (X)")
                
        for i, feature in enumerate(headers):
            if feature != target:
                data_functions.data_file.selection.append(st.checkbox(feature, value=True, key="feat"+str(i)))

        # st.write(data_functions.data_file.selection)
        
        if st.button("Next"):
            st.switch_page("page_visualizer.py")           
        
def main():    
    _, col1, _ = st.columns([2,8,2])    
    with col1:
        st.header("Welcome to Classic Learner")
        with st.container():
            st.markdown("#### Start by uploading your dataset.")
            
            uploaded_file = st.file_uploader(label="Upload a CSV file", key="dataset_uploder", type=["csv"])
            if uploaded_file is not None:
                set_data.clear()
                set_data(uploaded_file)
                select_data()  

            # st.markdown("#### Or try loading sample dataset.")
            with st.expander("#### Or try loading sample dataset."):
                df = None
                btn_cols = st.columns(len(data_functions.sk_datasets))
                for i, (k, v) in enumerate(data_functions.sk_datasets.items()):
                    with btn_cols[i]:
                        if st.button(k + " dataset"):
                            df, feature_cols = get_dataset_df(v)
                            data_functions.data_file = DataFile(df=df, name=k)
                            data_functions.data_file.features = feature_cols
                            data_functions.data_file.target = "target"                        
                        
                if df is not None:
                    st.dataframe(df)  
        
main()