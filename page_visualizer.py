import streamlit as st
import plotly.express as px
import seaborn as sns
import data_functions


def show_meta_data():
    with st.expander("Meta data", expanded=True):
        data_functions.data_file.set_features()
        feature_string = ["`{}`".format(feat) for feat in data_functions.data_file.features]

        st.write("**File name:** `" + data_functions.data_file.name + "`")
        st.write("**Target column name:** `" + data_functions.data_file.target + "`")
        st.write("**Feature columns:** " + " ".join(feature_string))        

def visualize_categories():    
    with st.expander("Scatter plot"):
        # st.markdown("### Scatter plot")
        col1, col2 = st.columns(2)
        with col1:
            feat1 = st.selectbox("Feature 1", data_functions.data_file.features)
        with col2:
            feat2 = st.selectbox("Feature 2", data_functions.data_file.features)
        
        fig = px.scatter(
            data_functions.data_file.df,
            x=feat1,
            y=feat2,
            color=data_functions.data_file.target,
        )

        event = st.plotly_chart(fig, key="plot_scatter", on_select="rerun")

def visualize_boxplots():
    pass

def visualize_pairplot():
    with st.expander("Scatter plot"):
        # plot = sns.pairplot(data_functions.data_file.df, hue=data_functions.data_file.target)
        # st.pyplot(plot.figure)
        fig = px.scatter_matrix(data_functions.data_file.df,
            dimensions=data_functions.data_file.features,
            color=data_functions.data_file.target)
        event = st.plotly_chart(fig, key="plot_pair", on_select="rerun")

def main():    
    st.header("Visualize data")   
    if data_functions.data_file is not None:  
        show_meta_data()
        with st.expander("Show data file"):
            data_functions.data_file.df
        visualize_categories()
        visualize_pairplot()
    else:
        st.text("No dataset found. Please upload a CSV formatted data file.")
        if st.button("Go to upload page"):
            st.switch_page("page_start.py")
    

main()