import streamlit as st
import pandas as pd
import data_functions
import learner_module

def tablulate_models():
    st.markdown("""<style>.reportview-container .markdown-text-container { flex: 1; height: 500px; }</style>""", unsafe_allow_html=True)
    with st.container(border=True):
        df = pd.DataFrame.from_dict(learner_module.classification_algorithms)
        df = df.T
        df["selected"] = [True] * len(df)
        st.data_editor(
            df[["long_name", "selected"]],
            column_config={
                "model": st.column_config.CheckboxColumn(
                    "Select to train",
                    help="Select models to train",
                    default=True,
                )
            },
            disabled=["widgets"],
            hide_index=True,
            width=800
        )

def main():    
    st.header("Training models")    
    if data_functions.data_file is not None:  
        tablulate_models()
    else:
        st.text("No dataset found. Please upload a CSV formatted data file.")
        if st.button("Go to upload page"):
            st.switch_page("page_start.py")

main()