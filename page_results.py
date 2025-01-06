import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from learner_module import classification_algorithms
import data_functions

def get_eval_metrics():
    results = dict()
    ids, names, accs, f1s, aucs = [], [], [], [], []
    trained_models = st.session_state["Trained_models"]
    
    for k, v in trained_models.items():
        ids.append(k)
        names.append(classification_algorithms[k]["long_name"])
        accs.append(v.accuracy)
        f1s.append(v.f1)
        aucs.append(v.auc)
    
    results["id"] = ids
    results["Model"] = names
    results["Accuracy"] = accs
    results["F1 Score"] = f1s
    results["AUC"] = aucs
    
    return pd.DataFrame(data=results)

# @st.dialog("More details")
def show_confusion_matrix(id, name):
    model = st.session_state["Trained_models"][id]
    c_names = st.session_state["Dataset"].c_names
    # st.write(f"#### {name}")
    # with st.container(border=True):        
    #     st.write(f"Test samples: `{len(model.test_data[1])}` Classes: `{len(c_names)}` Accuracy: `{model.accuracy*100:.1f}%`")
        
    fig = px.imshow(model.c_matrix, 
                    text_auto=True, 
                    x=c_names, 
                    y=c_names, 
                    labels=dict(x="Predicted classes", y="True classes"))
    
    fig.update_coloraxes(showscale=False)
    event = st.plotly_chart(fig, key="plot_cmat", on_select="rerun")

def show_results_table(df):     
    cols = ['Accuracy','F1 Score', 'AUC']
    df[cols] = df[cols].map(lambda x: '{0:.4f}'.format(x))           
    # df.style.format({"Accuracy": "{:.4f}", "F1 Score": "{:,.4f}", "AUC": "{:,.4f}"})
    # df = df.astype(str)
    st.dataframe(df[["Model", "Accuracy", "F1 Score", "AUC"]], use_container_width=True, height=500)
    # event = st.dataframe(
    #     df[["Model", "Accuracy", "F1 Score", "AUC"]],
    #     on_select='rerun',
    #     selection_mode='single-row',
    #     use_container_width=True,
    #     hide_index=True,
    #     height=500
    # )

    # if len(event.selection['rows']):
    #     selected_row = event.selection['rows'][0]
    #     model_id = df.iloc[selected_row]['id']
    #     model_name = df.iloc[selected_row]['Model']
        # show_details(model_id, model_name)

def download_models(model_id):
    code = f'''
    import pickle
    
    # load model
    with open('{model_id}_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # pass feature set
    model.predict(X)
    '''
    st.info("Downalod pickel file and use following code for predictions")
    st.code(code, language="python")
    
    _, col = st.columns([9,3], gap="large")
    with col:
        st.download_button(
            "Download",
            data=pickle.dumps(st.session_state["Trained_models"][model_id].model),
            file_name=f"{model_id}_model.pkl",
            type='primary'
        )
    
   
def plot_bar(df, plot):
    fig = px.bar(df, x="id", y=plot)
    event = st.plotly_chart(fig, key="plot_bar_" + plot , on_select="rerun")

def main():    
    st.header("Evaluation summary")   
    
    # if len(learner_module.trained_models) > 0:
    if "Trained_models" in st.session_state:
        col1, col2 = st.columns([6,6], gap="medium")
        
        with col1:
            tab1, tab2, tab3, tab4 = st.tabs(["All results", "Accuary", "F1-Score", "AUC"])
            df = get_eval_metrics()
            with tab1:          
                show_results_table(df)                    
            with tab2:
                plot_bar(df, "Accuracy")
            with tab3:
                plot_bar(df, "F1 Score")
            with tab4:
                plot_bar(df, "AUC")
        
        with col2:
            ptab1, ptab2 = st.tabs(["Confusion Matrix", "Download models"])
            with ptab1:
                model_name = st.selectbox("Select model", df["Model"].to_list(), key="plot_mdl_select")
                model_id = df.loc[df['Model'] == model_name]["id"].values[0]
                # print(model_id)
                show_confusion_matrix(model_id, model_name)
            with ptab2:
                st.write("##### Download trained model")
                dwon_model_name = st.selectbox("Select model", df["Model"].to_list(),  key="down_mdl_select")
                model_id = df.loc[df['Model'] == dwon_model_name]["id"].values[0]
                download_models(model_id)

    else:
        st.warning("No trained models found. Please trained the models first.")
        if st.button("Go to training page"):
            st.switch_page("page_models.py")

main()