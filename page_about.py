import streamlit as st

def main():
    _, col, _ = st.columns([3,6,3])
    with col:
        st.header("About")
        
        with st.container():
            st.write("#### Classic Learner V1.0.0")
            st.write("**Author:** Isuru Jayarathne")

main()