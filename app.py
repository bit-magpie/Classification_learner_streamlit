import streamlit as st

st.set_page_config(
    page_icon="🎓", 
    page_title="Classic Learner",
    layout="wide"
    )

if "Dataset_loaded" not in st.session_state:
    st.session_state["Dataset_loaded"] = False

def main():  
    
    start_page = st.Page("page_start.py", title="Data selection")
    visualizer = st.Page("page_visualizer.py", title="Visualize data")
    models = st.Page("page_models.py", title="Model training")
    results = st.Page("page_results.py", title="Evaluvation summary")
    about = st.Page("page_about.py", title="About")
    page_nav = st.navigation([start_page, visualizer, models, results, about])
    page_nav.run()                  

main()