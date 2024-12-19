import streamlit as st
import pandas as pd
import plotly.express as px
import learner_module

def get_eval_metrics():
    results = dict()
    ids, names, accs, f1s, aucs = [], [], [], [], []
    
    for k, v in learner_module.trained_models.items():
        ids.append(k)
        names.append(learner_module.classification_algorithms[k]["long_name"])
        accs.append(v.accuracy)
        f1s.append(v.f1)
        aucs.append(v.auc)
    
    results["id"] = ids
    results["Model"] = names
    results["Accuracy"] = accs
    results["F1 Score"] = f1s
    results["AUC"] = aucs
    
    return pd.DataFrame(data=results)

@st.dialog("More details")
def show_details(id, name):
    model = learner_module.trained_models[id]
    st.write(f"#### {name}")
    st.write(f"Test samples: {len(model.test_data[1])}")
    fig = px.imshow(model.c_matrix, text_auto=True)
    event = st.plotly_chart(fig, key="plot_cmat", on_select="rerun")
    

def main():    
    st.header("Evaluation summary")
    if len(learner_module.trained_models) > 0:
        with st.container(border=True):
            df = get_eval_metrics()
            
            event = st.dataframe(
                df[["Model", "Accuracy", "F1 Score", "AUC"]],
                on_select='rerun',
                selection_mode='single-row',
                use_container_width=True
            )

            if len(event.selection['rows']):
                selected_row = event.selection['rows'][0]
                model_id = df.iloc[selected_row]['id']
                model_name = df.iloc[selected_row]['Model']

                if st.button("Show details", key="btnDetails"):
                    show_details(model_id, model_name)
    else:
        st.text("No trained models found. Please trained the models first.")
        if st.button("Go to training page"):
            st.switch_page("page_models.py")

main()