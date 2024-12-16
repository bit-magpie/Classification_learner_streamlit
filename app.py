import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px

st.set_page_config(page_icon="🎓", page_title="Leaner App")


def get_upload(uploaded_file):    
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    
    d = sns.load_dataset("penguins")
    plot = sns.pairplot(d, hue="species")
    st.pyplot(plot.figure)

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


        df = px.data.iris()
        fig = px.scatter(
            df,
            x="sepal_width",
            y="sepal_length",
            color="species",
            # size="petal_length",
            # hover_data=["petal_width"],
        )

        event = st.plotly_chart(fig, key="iris", on_select="rerun")

main()