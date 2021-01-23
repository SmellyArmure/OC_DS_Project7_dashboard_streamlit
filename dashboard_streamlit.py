# TO RUN: $streamlit run dashboard/dashboard.py
# Local URL: http://localhost:8501
# Network URL: http://192.168.0.50:8501
# Online URL: http://15.188.179.79

import streamlit as st
from PIL import Image
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import shap
import time
# sys.path.insert(0, '..\\NOTEBOOKS')
from P7_functions import CustTransformer
from P7_functions import plot_boxplot_var_by_target
from P7_functions import plot_scatter_projection


def main():
    # local API (à remplacer par l'adresse de l'application déployée)
    API_URL = "http://127.0.0.1:5000/api/"
    # API_URL = "https://oc-api-flask-mm.herokuapp.com/api/"

    ##################################
    # LIST OF API REQUEST FUNCTIONS

    # Get list of SK_IDS (cached)
    @st.cache
    def get_sk_id_list():
        # URL of the sk_id API
        SK_IDS_API_URL = API_URL + "sk_ids/"
        # Requesting the API and saving the response
        response = requests.get(SK_IDS_API_URL)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of SK_IDS from the content
        SK_IDS = pd.Series(content['data']).values
        return SK_IDS

    # Get Personal data (cached)
    @st.cache
    def get_data_cust(select_sk_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        PERSONAL_DATA_API_URL = API_URL + "data_cust/?SK_ID_CURR=" + str(select_sk_id)
        # save the response to API request
        response = requests.get(PERSONAL_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.Series
        data_cust = pd.Series(content['data']).rename("SK_ID {}".format(select_sk_id))
        data_cust_proc = pd.Series(content['data_proc']).rename("SK_ID {}".format(select_sk_id))
        return data_cust, data_cust_proc

    # Get data from 20 nearest neighbors in train set (cached)
    @st.cache
    def get_data_neigh(select_sk_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        NEIGH_DATA_API_URL = API_URL + "neigh_cust/?SK_ID_CURR=" + str(select_sk_id)
        # save the response of API request
        response = requests.get(NEIGH_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        X_neigh = pd.DataFrame(content['X_neigh'])
        y_neigh = pd.Series(content['y_neigh']['TARGET']).rename('TARGET')
        return X_neigh, y_neigh

    # Get all data in train set (cached)
    @st.cache
    def get_all_proc_data_tr():
        # URL of the scoring API
        ALL_PROC_DATA_API_URL = API_URL + "all_proc_data_tr/"
        # save the response of API request
        response = requests.get(ALL_PROC_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.Series
        X_tr_proc = pd.DataFrame(content['X_tr_proc'])
        y_tr = pd.Series(content['y_train']['TARGET']).rename('TARGET')
        return X_tr_proc, y_tr

    # Get scoring of one applicant customer (cached)
    @st.cache
    def get_cust_scoring(select_sk_id):
        # URL of the scoring API
        SCORING_API_URL = API_URL + "scoring_cust/?SK_ID_CURR=" + str(select_sk_id)
        # Requesting the API and save the response
        response = requests.get(SCORING_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # getting the values from the content
        score = content['score']
        thresh = content['thresh']
        return score, thresh

    # Get the list of features
    @st.cache
    def get_features_descriptions():
        # URL of the aggregations API
        FEAT_DESC_API_URL = API_URL + "feat_desc"
        # Requesting the API and save the response
        response = requests.get(FEAT_DESC_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        features_desc = pd.Series(content['data']['Description']).rename("Description")
        return features_desc

    # Get the shap values of the customer and 20 nearest neighbors (cached)
    @st.cache
    def get_shap_values(select_sk_id):
        # URL of the scoring API
        GET_SHAP_VAL_API_URL = API_URL + "shap_values/?SK_ID_CURR=" + str(select_sk_id)
        # save the response of API request
        response = requests.get(GET_SHAP_VAL_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame
        shap_val_df = pd.DataFrame(content['shap_val'])
        expected_value = content['exp_val']
        X_neigh_ = pd.DataFrame(content['X_neigh_'])
        return shap_val_df, expected_value, X_neigh_

    #################################
    #################################
    #################################
    # Configuration of the streamlit page
    st.set_page_config(page_title='Loan application scoring dashboard',
                       page_icon='random',
                       layout='centered',
                       initial_sidebar_state='auto')

    # Display the title
    st.title('Loan application scoring dashboard')
    st.header("Maryse MULLER - Data Science project 7")
    # st.subheader("Maryse MULLER - Parcours Data Science projet 7 - OpenClassrooms")
    # # change the color of the background
    # st.markdown("""<style> body {color: #fff;
    #                              background-color: #000066;} </style> """,
    #                              unsafe_allow_html=True)

    # Display the logo in the sidebar
    # path = os.path.join('dashboard','logo.png')
    path = "logo.png"
    image = Image.open(path)
    st.sidebar.image(image, width=180)

    ###############################################
    # # request to fetch the local background image 
    # @st.cache(allow_output_mutation=True)
    # def get_base64_of_bin_file(bin_file):
    #     with open(bin_file, 'rb') as f:
    #         data = f.read()
    #     return base64.b64encode(data).decode()

    # def set_png_as_page_bg(png_file):
    #     bin_str = get_base64_of_bin_file(png_file)
    #     page_bg_img = '''
    #     <style>
    #     body {
    #     background-image: url("data:image/png;base64,%s"); # 'banking_background.jpeg'
    #     background-size: cover;
    #     }
    #     </style>
    #     ''' % bin_str
        
    #     st.markdown(page_bg_img, unsafe_allow_html=True)
    #     return
    ################################################

    # change color of the sidebar
    # background-color: #011839;
    # background-image: url("https://img.wallpapersafari.com/desktop/1536/864/49/82/M3WxOo.jpeg");
    # st.markdown( """ <style> .css-1aumxhk {
    #                                        background-color: #011839;
    #                                        color: #ffffff} } </style> """,
    #                                        unsafe_allow_html=True, )

    #################################
    #################################
    #################################

    # ------------------------------------------------
    # Select the customer's ID
    # ------------------------------------------------

    SK_IDS = get_sk_id_list()
    select_sk_id = st.sidebar.selectbox('Select SK_ID from list:', SK_IDS, key=1)
    st.write('You selected: ', select_sk_id)

    # ------------------------------------------------
    # Get All Data relative to customer 
    # ------------------------------------------------

    # Get personal data (unprocessed and preprocessed)
    X_cust, X_cust_proc = get_data_cust(select_sk_id)

    # Get 20 neighbors personal data (preprocessed)
    X_neigh, y_neigh = get_data_neigh(select_sk_id)
    y_neigh = y_neigh.replace({0: 'repaid (neighbors)',
                               1: 'not repaid (neighbors)'})

    # Get all preprocessed training data
    X_tr_all, y_tr_all = get_all_proc_data_tr()  # X_tr_proc, y_proc
    y_tr_all = y_tr_all.replace({0: 'repaid (global)',
                                 1: 'not repaid (global)'})

    # ------------------------------------------------
    # Default value for main columns
    # ------------------------------------------------

    main_cols = ['binary__CODE_GENDER', 'high_card__OCCUPATION_TYPE',
                 'high_card__ORGANIZATION_TYPE', 'INCOME_CREDIT_PERC',
                 'EXT_SOURCE_2', 'ANNUITY_INCOME_PERC', 'EXT_SOURCE_3',
                 'AMT_CREDIT', 'PAYMENT_RATE', 'DAYS_BIRTH']

    # ##################################################
    # PERSONAL DATA
    # ##################################################

    if st.sidebar.checkbox("Customer's data"):

        st.header("Customer's data")

        format_dict = {'cust prepro': '{:.2f}',
                       '20 neigh (mean)': '{:.2f}',
                       '20k samp (mean)': '{:.2f}'}

        if st.checkbox('Show comparison with 20 neighbors and random sample'):
            # Concatenation of the information to display
            df_display = pd.concat([X_cust.rename('cust'),
                                    X_cust_proc.rename('cust prepro'),
                                    X_neigh.mean().rename('20 neigh (mean)'),
                                    X_tr_all.mean().rename('20k samp (mean)')
                                    ], axis=1)
            # subset = ['cust prepro', '20 neigh (mean)', '20k samp (mean)']
        else:
            # Display only personal_data
            df_display = pd.concat([X_cust.rename('cust'),
                                    X_cust_proc.rename('cust prepro')], axis=1)
            # subset = ['cust prepro']

        # Display at last 
        st.dataframe(df_display.style.format(format_dict)
                                     .background_gradient(cmap='seismic',
                                                          axis=0, subset=None,
                                                          text_color_threshold=0.2,
                                                          vmin=-1, vmax=1)
                                     .highlight_null('lightgrey'))

        expander = st.beta_expander("Concerning the graph...")
        # format de la première colonne objet ?

        expander.write("Here my explanation of the graphs")

    # #################################################
    # BOXPLOT FOR MAIN 10 VARIABLES
    # ##################################################

    if st.sidebar.checkbox('Boxplots of the main features'):

        st.header('Boxplots of the main features')

        # with st.spinner('Boxplot creation in progress...'):

        # ----------------------------
        # place to choose main_cols
        # ----------------------------
        fig = plot_boxplot_var_by_target(X_tr_all, y_tr_all, X_neigh, y_neigh,
                                         X_cust_proc, main_cols, figsize=(15, 4))

        st.write(fig)  # st.pyplot(fig) # the same
        st.markdown('_Dispersion of the main features for random sample,\
            20 nearest neighbors and applicant customer_')

        expander = st.beta_expander("Concerning the graph...")

        expander.write("Here my explanation of the graphs")

        # st.success('Done!')

    # #################################################
    # SCATTERPLOT TWO OR MORE FEATURES
    # ##################################################

    if st.sidebar.checkbox('Scatterplot comparison'):
    
        st.header('Scatterplot comparison')
        st.write(X_cust_proc[['EXT_SOURCE_2', 'EXT_SOURCE_3']].head())
        fig = plot_scatter_projection(X=X_tr_all,
                                      ser_clust=y_tr_all.replace({0: 'repaid', 1: 'not repaid'}),
                                      n_display=200,
                                      plot_highlight=X_neigh,
                                      X_cust=X_cust_proc,
                                      figsize=(10, 5),
                                      size=40,
                                      fontsize=12,
                                      columns=['EXT_SOURCE_2', 'EXT_SOURCE_3'])
        st.write(fig)  # st.pyplot(fig)
        st.markdown('_Scatter plot of random sample, 20 nearest neighbors and applicant customer_')

    ##################################################
    # SCORING
    ##################################################

    if st.sidebar.checkbox("Scoring and model's decision"):

        st.header("Scoring and model's decision")

        #  Get score
        score, thresh = get_cust_scoring(select_sk_id)

        # Display score (default probability)
        st.write('Default probability: {:.0f}%'.format(score*100))
        # Display default threshold
        st.write('Default model threshold: {:.0f}%'.format(thresh*100))

        # Compute decision according to the best threshold (True: loan refused)
        bool_cust = (score >= thresh)

        if bool_cust is False:
            decision = "Loan granted" 
            # st.balloons()
            # st.warning("The loan has been accepted but be careful...")
        else:
            decision = "LOAN REJECTED"
        
        st.write('Decision:', decision)

        if st.checkbox('Show explanations'):
            # proportion among nearest neighbors
            st.write("proportion among nearest neighbors")

    # #################################################
    # FEATURES' IMPORTANCE (SHAP VALUES) for 20 nearest neighbors
    # ##############################################

    if st.sidebar.checkbox("Relative importance of features"):

        st.header("Relative importance of features")

        # get shap's values for customer and 20 nearest neighbors
        shap_val_df, expected_value, X_neigh_ = get_shap_values(select_sk_id)

        # nb_features = 

        # draw the graph
        shap.plots._waterfall.waterfall_legacy(expected_value,
                                               shap_val_df.values[-1],
                                               X_neigh_.values.reshape(-1),
                                               feature_names=list(X_neigh_.columns),
                                               max_display=10, show=False)
        plt.gcf().set_size_inches((14, 3))
        # plt.show()

        # Plot the graph on the dashboard
        st.pyplot(plt.gcf())

        if st.checkbox('Show details'):  # .style.format(format_dict)\
            st.dataframe(shap_val_df.style.background_gradient(cmap='seismic',
                                                               axis=0, subset=None,
                                                               text_color_threshold=0.2,
                                                               vmin=-1, vmax=1)
                         .highlight_null('lightgrey'))

    # #################################################
    # FEATURES DESCRIPTIONS
    # #################################################

    features_desc = get_features_descriptions()

    if st.sidebar.checkbox('Features descriptions'):

        st.header("Features descriptions")

        list_features = features_desc.index.to_list()

        feature = st.selectbox('List of the features:', list_features, key=1)
        # st.write("Feature's name: ", feature)
        # st.write('Description: ', str(features_desc.loc[feature]))
        st.table(features_desc.loc[feature:feature])

        if st.checkbox('show all'):
            # Display features' descriptions
            st.table(features_desc)
    
    ################################################


if __name__ == '__main__':
    main()
