import streamlit as st
import pandas as pd
from data_functions import DataFile
import data_functions

@st.cache_resource
def set_data(uploaded_file):
    data_functions.data_file = DataFile(uploaded_file)
    
def set_features():
    data_functions.data_file.set_features()

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
        
    # st.write(dataset)
    
    # d = sns.load_dataset("penguins")
    # plot = sns.pairplot(d, hue="species")
    # st.pyplot(plot.figure)

def main():
    st.header("Machine Learning UI")    
    with st.container(border=True):
        st.markdown("### Start by uploading your dataset.")
        # with st.expander("What should be the structure?"):
        #     st.write('''
        #         The chart above shows some numbers I picked for you.
        #         I rolled actual dice for these, so they're *guaranteed* to
        #         be random.
        #     ''')
        uploaded_file = st.file_uploader(label="Upload a CSV file", key="dataset_uploder", type=["csv"])
        if uploaded_file is not None:
            set_data.clear()
            set_data(uploaded_file)
            select_data()
            
        
main()