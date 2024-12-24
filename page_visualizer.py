import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from data_functions import data_file, get_tsne


def show_meta_data():
    # with st.expander("Meta data", expanded=True):
    with st.container(border=True):
        data_file.set_features()
        feature_string = ["`{}`".format(feat) for feat in data_file.features]

        st.write("**File name:** `" + data_file.name + "`")
        st.write("**Target column name:** `" + data_file.target + "`")
        st.write("**Feature columns:** " + " ".join(feature_string))        

def visualize_scatter():    
    # with st.expander("Scatter plot"):
    # st.markdown("### Scatter plot")
    feat_list = data_file.features
    col1, col2 = st.columns(2)
    with col1:
        feat1 = st.selectbox("Feature 1", feat_list)
    with col2:
        feat2 = st.selectbox("Feature 2", feat_list, index=1)
    
    fig = px.scatter(
        data_file.df,
        x=feat1,
        y=feat2,
        color=data_file.target,
    )

    event = st.plotly_chart(fig, key="plot_scatter", on_select="rerun")

def visualize_boxplots():
    # with st.expander("Boxplots"):        
    feat = st.selectbox("Select feature", data_file.features, key="ft_bplot")
    fig = px.box(data_file.df, 
                    x=data_file.target, 
                    y=feat)
    event = st.plotly_chart(fig, key="plot_box", on_select="rerun")

# def visualize_distplot():
#     feat = st.selectbox("Select feature", data_file.features, key="ft_dplot")
#     fig = ff.create_distplot(data_file.df, 
#                     x=data_file.target, 
#                     y=feat)
#     event = st.plotly_chart(fig, key="plot_box", on_select="rerun")

def visualize_pairplot():
    # with st.expander("Pairplot"):
    # plot = sns.pairplot(data_file.df, hue=data_file.target)
    # st.pyplot(plot.figure)
    if len(data_file.features) > 5:
        tsne_df = get_tsne()
        fig = px.scatter(tsne_df, x="t-SNE1", y="t-SNE2",
            color=data_file.target,
        )

        event = st.plotly_chart(fig, key="plot_tsne", on_select="rerun")
    
    else:
        fig = px.scatter_matrix(data_file.df,
            dimensions=data_file.features,
            color=data_file.target)
        event = st.plotly_chart(fig, key="plot_pair", on_select="rerun")

def main():    
    st.header("Visualize data")   
    if data_file is not None: 
        col1, _, col2 = st.columns([8, 1, 3])     

        with col1:
            tab1, tab2, tab3, tab4 = st.tabs(["Scatter plot", "Box plots","t-SNE/Pair plot", "Data table"])
            with tab1:            
                visualize_scatter()
            with tab2:
                visualize_boxplots()
            with tab3:
                visualize_pairplot()
            with tab4:
                st.dataframe(data_file.df, use_container_width=True)
        with col2: 
            show_meta_data()
    else:
        st.text("No dataset found. Please upload a CSV formatted data file.")
        if st.button("Go to upload page"):
            st.switch_page("page_start.py")
    

main()