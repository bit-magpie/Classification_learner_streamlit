import streamlit as st
import data_functions
import learner_module

def tablulate_models():
    pass

def main():    
    st.header("Training models")    
    if data_functions.data_file is not None:  
        with st.container(border=True):
            st.text("models")

main()