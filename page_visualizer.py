import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from data_functions import dataset, get_tsne

if "Dataset" in st.session_state:
    dataset = st.session_state["Dataset"]

def show_meta_data():
    # with st.expander("Meta data", expanded=True):
    with st.container(border=True):
        dataset.set_features()
        feature_string = ["`{}`".format(feat) for feat in dataset.features]

        st.write("**File name:** `" + dataset.name + "`")
        st.write("**Target column name:** `" + dataset.target + "`")
        st.write("**Feature columns:** " + " ".join(feature_string))        

def visualize_scatter():    
    # with st.expander("Scatter plot"):
    # st.markdown("### Scatter plot")
    feat_list = dataset.features
    col1, col2 = st.columns(2)
    with col1:
        feat1 = st.selectbox("Feature 1", feat_list)
    with col2:
        feat2 = st.selectbox("Feature 2", feat_list, index=1)
    
    fig = px.scatter(
        dataset.df,
        x=feat1,
        y=feat2,
        color=dataset.target,
    )

    event = st.plotly_chart(fig, key="plot_scatter", on_select="rerun")

def visualize_boxplots():
    # with st.expander("Boxplots"):        
    feat = st.selectbox("Select feature", dataset.features, key="ft_bplot")
    fig = px.box(dataset.df, 
                    x=dataset.target, 
                    y=feat)
    event = st.plotly_chart(fig, key="plot_box", on_select="rerun")

def visualize_distplot():
    feat = st.selectbox("Select feature", dataset.features, key="ft_dplot")
    groups = dataset.df[[feat, dataset.target]].groupby([dataset.target])
    values = []
    classes = []
    for name, value in groups:
        values.append(value[feat].tolist())
        classes.append(name[0])
        
    fig = ff.create_distplot(values, classes, bin_size=.2)
    event = st.plotly_chart(fig, key="plot_dist", on_select="rerun")

def visualize_pairplot():
    st.info("If the dataset contains more than 5 features, a t-SNE plot will be displayed; otherwise, a pairplot will be shown.", icon="ℹ️")
    if len(dataset.features) > 5:
        tsne_df = get_tsne()
        fig = px.scatter(tsne_df, x="t-SNE1", y="t-SNE2",
            color=dataset.target,
        )

        event = st.plotly_chart(fig, key="plot_tsne", on_select="rerun")
    
    else:
        fig = px.scatter_matrix(dataset.df,
            dimensions=dataset.features,
            color=dataset.target)
        event = st.plotly_chart(fig, key="plot_pair", on_select="rerun")

def main():    
    st.header("Visualize data")   
    if st.session_state["Dataset_loaded"]:
    # if dataset is not None: 
        col1, _, col2 = st.columns([8, 1, 3])     

        with col1:
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Scatter plot", "t-SNE/Pair plot", "Box plots", "Distributions"])
            with tab1:            
                visualize_scatter()
            with tab2:
                visualize_pairplot()                
            with tab3:
                visualize_boxplots()
            with tab4:
                visualize_distplot()
        with col2: 
            show_meta_data()
    else:
        st.warning('No dataset found. Please upload a CSV formatted data file or load existing dataset and click the "Load Dataset and Proceed" button.')
        if st.button("Go to upload page"):
            st.switch_page("page_start.py")
    

main()