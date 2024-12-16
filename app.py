import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_icon="🎓", page_title="Leaner App")


def get_upload(uploaded_file):    
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

def main():    
    st.header("Machine Learning UI")
    with st.container(border=True):
        st.markdown("### Start by uploading your dataset.")
        with st.expander("What should be the structure?"):
            st.write('''
                The chart above shows some numbers I picked for you.
                I rolled actual dice for these, so they're *guaranteed* to
                be random.
            ''')
        uploaded_file = st.file_uploader(label="Upload a CSV file", key="dataset_uploder", type=["csv"])
        if uploaded_file is not None:
            get_upload(uploaded_file)

main()