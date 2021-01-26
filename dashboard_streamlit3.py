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
from custtransformer import CustTransformer
from dashboard_functions import plot_boxplot_var_by_target
from dashboard_functions import plot_scatter_projection


def main():
    # local API (à remplacer par l'adresse de l'application déployée)
    # API_URL = "http://127.0.0.1:5000/api/"
    API_URL = "https://oc-api-flask-mm.herokuapp.com/api/"

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
        data_cust = pd.Series(content['data']).rename(select_sk_id)
        data_cust_proc = pd.Series(content['data_proc']).rename(select_sk_id)
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
    
    # Get the list of feature importances (according to lgbm classification model)
    @st.cache
    def get_features_importances():
        # URL of the aggregations API
        FEAT_IMP_API_URL = API_URL + "feat_imp"
        # Requesting the API and save the response
        response = requests.get(FEAT_IMP_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        feat_imp = pd.Series(content['data']).sort_values(ascending=False)
        return feat_imp

    # Get the shap values of the customer and 20 nearest neighbors (cached)
    @st.cache
    def get_shap_values(select_sk_id):
        # URL of the scoring API
        GET_SHAP_VAL_API_URL = API_URL + "shap_values/?SK_ID_CURR=" + str(select_sk_id)
        # save the response of API request
        response = requests.get(GET_SHAP_VAL_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame or pd.Series
        shap_val_df = pd.DataFrame(content['shap_val'])
        shap_val_trans = pd.Series(content['shap_val_cust_trans'])
        exp_value = content['exp_val']
        exp_value_trans = content['exp_val_trans']
        X_neigh_ = pd.DataFrame(content['X_neigh_'])
        return shap_val_df, shap_val_trans, exp_value, exp_value_trans, X_neigh_

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
    select_sk_id = st.sidebar.selectbox('Select SK_ID from list:', SK_IDS, key=18)
    st.write('You selected: ', select_sk_id)
    # get shap's values for customer and 20 nearest neighbors
    shap_val, shap_val_trans, exp_val, exp_val_trans, X_neigh_ = \
        get_shap_values(select_sk_id)

    # ------------------------------------------------
    # Get All Data relative to customer 
    # ------------------------------------------------

    # Get personal data (unprocessed and preprocessed)
    X_cust, X_cust_proc = get_data_cust(select_sk_id)  # pd.Series !!!!

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
    feat_imp = get_features_importances()
    main_cols = list(feat_imp.sort_values(ascending=False)\
                                               .iloc[:12].index)
    # st.write(main_cols)

    # main_cols = ['binary__CODE_GENDER', 'high_card__OCCUPATION_TYPE',
    #              'high_card__ORGANIZATION_TYPE', 'INCOME_CREDIT_PERC',
    #              'EXT_SOURCE_2', 'ANNUITY_INCOME_PERC', 'EXT_SOURCE_3',
    #              'AMT_CREDIT', 'PAYMENT_RATE', 'DAYS_BIRTH']

    # #############################
    
    def get_list_display_features(shap_val_trans, def_n, key):
    
        all_feat = X_tr_all.columns.to_list()
        
        n = st.slider("Nb of features to display",
                      min_value=2, max_value=42,
                      value=def_n, step=None, format=None, key=key)
        
        if st.checkbox('Choose main features according to SHAP local interpretation for the applicant customer', key=key):
            disp_cols = list(shap_val_trans.abs()
                                .sort_values(ascending=False)
                                .iloc[:n].index)
        else:
            disp_cols = list(feat_imp.sort_values(ascending=False)\
                                            .iloc[:n].index)
            
        disp_box_cols = st.multiselect('Choose the features to display (default: order of general importance for lgbm calssifier):',
                                        sorted(all_feat),
                                        default=disp_cols, key=key)
        return disp_box_cols

    
    # ##########################


    ##################################################
    # SCORING
    ##################################################

    if st.sidebar.checkbox("Scoring and model's decision", key=38):

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
            # st.warning("The loan has been accepted but...")
        else:
            decision = "LOAN REJECTED"
        
        st.write('Decision:', decision)
        
        expander = st.beta_expander("Concerning the classification model...")

        expander.write("The prediction was made using a LGBM (Light Gradient Boosting Model) \
classification model.")

        expander.write("The default model threshold is tuned to maximize a gain function that penalizes \
'false negative'/type II errors (i.e. granted loans that would not actually not be repaid) \
10 times more than 'false positive'/type I errors (i.e. rejected loans that would actually be repaid).")


        if st.checkbox('Show local interpretation', key=37):

            with st.spinner('SHAP waterfall plot creation in progress...'):
                
                nb_features = st.slider("Nb of features to display",
                                        min_value=2, max_value=42,
                                        value=10, step=None, format=None, key=14)
                # # get shap's values for customer and 20 nearest neighbors
                # shap_val, shap_val_trans, exp_val, exp_val_trans, X_neigh_ = \
                #     get_shap_values(select_sk_id)
                
                # draw the graph (only for the customer with scaling)
                shap.plots._waterfall.waterfall_legacy(exp_val_trans,
                                                       shap_val_trans,
                                                       feature_names=list(shap_val_trans.index),
                                                       max_display=nb_features,
                                                       show=False)
                plt.gcf().set_size_inches((14, nb_features/2))
                # Plot the graph on the dashboard
                st.pyplot(plt.gcf())

                st.markdown('_SHAP waterfall plot for the applicant customer._')

                expander = st.beta_expander("Concerning the SHAP waterfall plot...")

                expander.write("The above waterfall plot displays \
explanations for the individual prediction of the applicant customer.\
The bottom of a waterfall plot starts as the expected value of the model output \
(i.e. the value obtained if no information (features) were provided), and then \
each row shows how the positive (red) or negative (blue) contribution of \
each feature moves the value from the expected model output over the \
background dataset to the model output for this prediction.")
                expander.write("NB: for LGBM classification model, the sum of the SHAP values is NOT \
usually the final probability prediction but log odds values. \
On this graph, a simple scaling has been performed so that the base value \
represents the probability obtained if no particular information is given, and the sum of \
the values on the arrows is the predicted probability of default on the loan (non repayment).")

            if st.checkbox('Show detail of the shap values for 20 nearest neighbors (without re-scaling)', key=35):
                st.dataframe(shap_val.style.format("{:.2}")
                 .background_gradient(cmap='seismic', axis=0, subset=None,
                                     text_color_threshold=0.2, vmin=-1, vmax=1)
                 .highlight_null('lightgrey'))

    # ##################################################
    # CUSTOMER'S DATA
    # ##################################################

    if st.sidebar.checkbox("Customer's data"):

        st.header("Customer's data")

        format_dict = {'cust prepro': '{:.2f}',
                       '20 neigh (mean)': '{:.2f}',
                       '20k samp (mean)': '{:.2f}'}
        all_feat = list(set(X_cust.index.to_list() + X_cust_proc.index.to_list()))
        disp_cols = st.multiselect('Choose the features to display:',
                                   sorted(all_feat),#.sort(),
                                   default=sorted(main_cols))
        
        disp_cols_not_prepro = [col for col in disp_cols \
                                if col in X_cust.index.to_list()]
        disp_cols_prepro = [col for col in disp_cols \
                            if col in X_cust_proc.index.to_list()]

        if st.checkbox('Show comparison with 20 neighbors and random sample', key=31):
            # Concatenation of the information to display
            df_display = pd.concat([X_cust.loc[disp_cols_not_prepro].rename('cust'),
                                    X_cust_proc.loc[disp_cols_prepro].rename('cust prepro'),
                                    X_neigh[disp_cols_prepro].mean().rename('20 neigh (mean)'),
                                    X_tr_all[disp_cols_prepro].mean().rename('20k samp (mean)')
                                    ], axis=1)  # all pd.Series
        else:
            # Display only personal_data
            df_display = pd.concat([X_cust.loc[disp_cols_not_prepro].rename('cust'),
                                    X_cust_proc.loc[disp_cols_prepro].rename('cust prepro')], axis=1)  # all pd.Series

        # Display at last 
        st.dataframe(df_display.style.format(format_dict)
                                     .background_gradient(cmap='seismic',
                                                          axis=0, subset=None,
                                                          text_color_threshold=0.2,
                                                          vmin=-1, vmax=1)
                                     .highlight_null('lightgrey'))
        
        st.markdown('_Data used by the model, for the applicant customer,\
            for the 20 nearest neighbors and for a random sample_')

        expander = st.beta_expander("Concerning the data table...")
        # format de la première colonne objet ?

        expander.write("The above table shows the value of each feature:\
  \n- _cust_: values of the feature for the applicant customer,\
unprocessed  /n- _cust prepro_: values of the feature for the \
 applicant customer after categorical encoding and standard scaling\
  \n- _20 neigh (mean)_: mean of the preprocessed values of each feature \
  for the 20 nearest neighbors of the applicant customer in the training \
set  \n- _20k samp (mean)_: mean of the preprocessed values of each feature \
for a random sample of customers from the training set.")

    # #################################################
    # BOXPLOT FOR MAIN 10 VARIABLES
    # ##################################################

    if st.sidebar.checkbox('Boxplots of the main features', key=23):

        st.header('Boxplots of the main features')

        plt.clf()

        with st.spinner('Boxplot creation in progress...'):
            
            disp_box_cols = get_list_display_features(shap_val_trans, 10, key=43)
            
            fig = plot_boxplot_var_by_target(X_tr_all, y_tr_all, X_neigh, y_neigh,
                                             X_cust_proc, disp_box_cols, figsize=(10, 4))

            st.write(fig)  # st.pyplot(fig) # the same
            st.markdown('_Dispersion of the main features for random sample,\
 20 nearest neighbors and applicant customer_')

            expander = st.beta_expander("Concerning the dispersion graph...")

            expander.write("These boxplots show the dispersion of the preprocessed features values\
 used by the model to make a prediction. The green boxplot are for the customers that repaid \
their loan, and red boxplots are for the customers that didn't repay it.Over the boxplots are\
 superimposed (markers) the values\
 of the features for the 20 nearest neighbors of the applicant customer in the training set. The \
 color of the markers indicate whether or not these neighbors repaid their loan. \
 Values for the applicant customer are superimposed in yellow.")

    # #################################################
    # SCATTERPLOT TWO OR MORE FEATURES
    # ##################################################

    if st.sidebar.checkbox('Scatterplot comparison', key=27):
    
        st.header('Scatterplot comparison')
        
        plt.clf()
        
        disp_box_cols = get_list_display_features(shap_val_trans, 2, key=41)
        if len(disp_box_cols)>2:
            proj = st.radio("Choose projection method", ['PCA', 't-SNE'])
        else:
            proj = 'PCA'
        
        with st.spinner('Scatterplot creation in progress... may take some time if n>2 (computes projection)...'):
            fig = plot_scatter_projection(X=X_tr_all,
                                          ser_clust=y_tr_all.replace({0: 'repaid',
                                                                      1: 'not repaid'}),
                                          n_display=200,
                                          plot_highlight=X_neigh,
                                          X_cust=X_cust_proc,  # pd.Series
                                          proj=proj,
                                          figsize=(10, 5),
                                          size=40,
                                          fontsize=12,
                                          columns=disp_box_cols)
            
            st.write(fig)  # st.pyplot(fig)
            st.markdown('_Scatter plot of random sample, nearest neighbors and applicant customer_')

            expander = st.beta_expander("Concerning the scatterplot graph...")

            expander.write("This scatterplot graph shows the preprocessed features values \
used by the model to make a prediction. The green markers are for the customers that repaid \
their loan, and the red markers are for the customers that didn't repay it. Small markers are \
customers among a random sample of the training set. Bigger markers are the 20 nearest neighbors \
of the applicant customer in the training set. \
Values for the applicant customer are superimposed in yellow.")
            expander.write("If the selected features exceed 2, a PCA or t-SNE projection of the data \
 is shown.")

    # #################################################
    # FEATURES' IMPORTANCE (SHAP VALUES) for 20 nearest neighbors
    # ##############################################

    if st.sidebar.checkbox("Importance of the features", key=29):

        st.header("Comparison of local and global feature importances")
        
        plot_choice = []
        if st.checkbox('Add global feature importance', value=True, key=25):
            plot_choice.append(0)
        if st.checkbox('Add local feature importance for the nearest neighbors', key=28):
            plot_choice.append(1)
        if st.checkbox('Add local feature importance for the applicant cutomer', key=26):
            plot_choice.append(2)
        
        # # get shap's values for customer and 20 nearest neighbors
        # shap_val_df, _, _, _, X_neigh_ = get_shap_values(select_sk_id)
            
        disp_box_cols = get_list_display_features(shap_val_trans, 10, key=42)
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 3))
    
        global_imp = feat_imp.loc[disp_box_cols]
        mean_shap_neigh = shap_val.mean().loc[disp_box_cols]
        shap_val_cust = shap_val.iloc[-1].loc[disp_box_cols]
        
        from sklearn.preprocessing import MinMaxScaler
        minmax = MinMaxScaler()
        
        df_disp = pd.concat([global_imp.to_frame('global'),
                             mean_shap_neigh.to_frame('neigh'),
                             shap_val_cust.to_frame('cust')],
                            axis=1)
        df_disp_sc = pd.DataFrame(minmax.fit_transform(df_disp),
                                  index=df_disp.index,
                                  columns=df_disp.columns)
        plot_choice = [0] if plot_choice == [] else plot_choice
        disp_df_choice = df_disp_sc.iloc[:,plot_choice]
        disp_df_choice.sort_values(disp_df_choice.columns[0]).plot.barh(width=0.8, ec='k',
                                                                       color=['navy', 'red', 'orange'],
                                                                       ax=ax2)
        
        plt.legend()
        fig2.set_size_inches((8, len(disp_box_cols)/2))
        # # Plot the graph on the dashboard
        st.pyplot(fig2)
        
        st.markdown('_Relative global and local feature importances_')
        
        expander = st.beta_expander("Concerning the comparison of local and global feature importances...")
        
        expander.write("The global feature importances in blue (computed globally for the lgbm model when trained on the training set)\
 are compared in the above bar chart to the local importance (SHAP values) of each features for the 20 nearest neighbors (red) \
 or for the applicant customer (orange).")
        expander.write("NB: For easier visualization the values in each bar plot has be scaled with min-max \
scaling (0 stands for min and 1 for max value).") 
        
        if st.checkbox('Detail of SHAP feature importances for the applicant customer neighbors', key=24):
            
            st.header("Local feature importance of the features for the nearest neighbors")
            
            plt.clf()
            nb_features = st.slider("Nb of features to display",
                        min_value=2, max_value=42,
                        value=10, step=None, format=None, key=13)
            
            # draw the graph
            shap.summary_plot(shap_val.values,  # shap values
                              X_neigh_.values,  # data (np.array)
                              feature_names=list(X_neigh_.columns),  # features name of data (order matters)
                              max_display=nb_features,  # nb of displayed features
                              show=True)  # enables setting of plot size later using matplotlib)  
            fig = plt.gcf()
            fig.set_size_inches((10, nb_features/2))
            # Plot the graph on the dashboard
            st.pyplot(fig)
            
            st.markdown('_Beeswarm plot showing the SHAP values for each feature and for \
the nearest neighbors of the applicant customer_')

            expander = st.beta_expander("Concerning the SHAP waterfall plot...")

            expander.write("The above beeswarm plot displays \
the SHAP values for the individual prediction of the applicant customer and his \
20 nearest neighbors for each feature, as well a the corresponding value of this feature (colormap).")

    # #################################################
    # FEATURES DESCRIPTIONS
    # #################################################

    features_desc = get_features_descriptions()

    if st.sidebar.checkbox('Features descriptions', key=22):

        st.header("Features descriptions")

        list_features = features_desc.index.to_list()

        feature = st.selectbox('List of the features:', list_features, key=15)
        # st.write("Feature's name: ", feature)
        # st.write('Description: ', str(features_desc.loc[feature]))
        st.table(features_desc.loc[feature:feature])

        if st.checkbox('show all', key=20):
            # Display features' descriptions
            st.table(features_desc)
    
    ################################################


if __name__ == '__main__':
    main()
