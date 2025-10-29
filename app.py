import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error,r2_score, root_mean_squared_error, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVR


#Visualization Imports
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st



#page config
st.set_page_config( 
    page_title="Insurance Charges Dashboard",
    page_icon="📊", 
    layout="wide"
    )

#styling tables
def style_data():
    return [
        {   #body
            'selector': 'td',
            'props': [
                ('font-family', 'Roboto, sans-serif'), 
                ('font-size', '14px'),                 
                ('background-color', '#282C34'),       
                ('color', '#E0E0E0')                  
            ]
        },
        {
            #headers
            'selector': 'th',
            'props': [
                ('font-family', 'Roboto, sans-serif'), 
                ('font-size', '16px'),                  
                ('background-color', '#1C1E21'),        
                ('color', '#00BCD4'),                   
                ('border', '1px solid #00BCD4')          
            ]
        }
    ]




@st.cache_data
def load_data():
    data = pd.read_csv("medical insurance.csv")
    
    #Preprocess data
    data['sex'] = pd.factorize(data['sex'])[0]
    data['smoker'] = pd.factorize(data['smoker'])[0]
    data['region'] = pd.factorize(data['region'])[0]

    group_mean = data.groupby('sex')['bmi'].transform('mean')
    group_std = data.groupby('sex')['bmi'].transform('std')
    data['bmi_normal'] = (data['bmi'] - group_mean) / group_std

    return data

@st.cache_data
def raw_data():
    data = pd.read_csv("medical insurance.csv")
    return data

df = load_data()
raw = raw_data()


#Navigation bar options

PAGE_OPTIONS = {
    "Homepage (SDE)": "homepage",
    "Model Evaluation": "model_evaluation",
    "Pairwise Plots": "pairwise_plots",
    "K-Fold Cross Validation": "CV_page"
}

with st.sidebar:
    st.title("Navigation")
    
    # Navigation Radio Button
    selection = st.radio("Go to", list(PAGE_OPTIONS.keys()))


#Homepage 

def homepage():
    
    st.title("Insurance Charges Dashboard")
    st.markdown("Analyze and visualize insurance charges data.")
    
    raw_styled = raw.style.set_table_styles(style_data())

    st.write("Insurance Charges Dataset Overview",)
    st.dataframe(round(raw_styled.head(), 2), use_container_width=True)

    st.subheader("Dataset Summary Statistics")
    st.dataframe(round(raw_styled.drop(columns=['children']).describe(), 2), use_container_width=True)


    #drop box to draw feature vs charges plot

    st.subheader("Relationship between each feature and charges")
    feature_selected = st.selectbox("Select Feature for Visualization", df.columns.drop('charges'))
     
    
    fig1, ax1 = plt.subplots(figsize = (7, 4))
    ax1.set_title(f"Charges vs {feature_selected}")
    ax1.set_xlabel(feature_selected)
    ax1.set_ylabel("Charges ($)")
    sns.scatterplot(x = feature_selected, y = 'charges', data = df, ax = ax1, alpha = 1.0, color = "orange")
    st.pyplot(fig1)

    fig, ax = plt.subplots(1, 3, figsize = (10, 4))
    ax[0].hist(raw["age"], bins = 20, color = 'skyblue', edgecolor = 'black')
    ax[0].set_title("Age Distribution")
    ax[0].set_xlabel("Age (Years)")
    ax[0].set_ylabel("Frequency")
    ax[1].hist(raw["bmi"], bins = 20, color = 'lightcoral', edgecolor = 'black')
    ax[1].set_title("BMI Distribution")
    ax[1].set_xlabel("BMI")
    ax[1].set_ylabel("Frequency")
    ax[2].hist(raw["charges"], bins = 40, color = 'lightgreen', edgecolor = 'black')
    ax[2].set_title("Charges Distribution")
    ax[2].set_xlabel("Insurance Charges ($)")
    ax[2].set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)


    




#Pair-wise plots Page
@st.cache_data
def pairwise_plots():
    data = raw
    PLOT_HEIGHT = 2.0
    st.subheader("Figure 1: Pairwise Relationships Hued by SMOKER Status ")
    sns.pairplot(
        data.drop(columns=['children']),
        height=PLOT_HEIGHT,
        hue="smoker",
        diag_kind="kde",
        plot_kws={'alpha': 0.6, 's': 20}
    )
    # plt.suptitle("Pairwise Relationships of Medical Data, Colored by Smoker Status",
    #             y=1.02)
    st.pyplot(plt)
    st.subheader("Figure 2: Pairwise Relationships Hued by SEX")
    sns.pairplot(
        data.drop(columns=['children']),
        height=PLOT_HEIGHT,
        hue="sex",
        diag_kind="kde",
        plot_kws={'alpha': 0.6, 's': 20}
    )
    plt.suptitle("Pairwise Relationships of Medical Data, Colored by Sex",
                y=1.02)
    st.pyplot(plt)
    st.subheader("Figure 3: Pairwise Relationships Hued by REGION")
    sns.pairplot(
        data.drop(columns=['children']),
        height=PLOT_HEIGHT,
        hue="region",
        diag_kind="kde",
        plot_kws={'alpha': 0.6, 's': 20}
    )
    plt.suptitle("Pairwise Relationships of Medical Data, Colored by Region",
                y=1.02)
    st.pyplot(plt)
    st.subheader("Figure 4: Pairwise Relationships Hued by CHILDREN")
    sns.pairplot(
        data.drop(columns=['children']),
        height=PLOT_HEIGHT,
        hue="region",
        diag_kind="kde",
        plot_kws={'alpha': 0.6, 's': 20}
    )
    plt.suptitle("Pairwise Relationships of Medical Data, Colored by Region",
                y=1.02)
    st.pyplot(plt)



#ML model Training and Evaluation Page


@st.cache_resource
def RF_model(X_tr, y_tr):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    return model

@st.cache_resource

def Linear_model(X_tr, y_tr):
    
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model

@st.cache_resource
def GradientBoost_model(X_tr, y_tr):
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    return model

@st.cache_resource
def PCA_model(X_tr, y_tr):
    
    pca = PCA(n_components=6)
    X_reduced = pca.fit_transform(X_tr)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_reduced, y_tr)
    return model

def SVM_model(X_tr, y_tr):
    model = SVR(kernel='rbf')
    model.fit(X_tr, y_tr)
    return model

MODELS = {
    "Random Forest": RF_model,
    "Linear Regression": Linear_model,
    "Gradient Boosting": GradientBoost_model,
    "PCA + Random Forest": PCA_model,
    "Support Vector Machine": SVM_model
}

MODELS1 = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "PCA + Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Machine": SVR(kernel='rbf')
}

#ML Model Evaluation Page

def model_evaluation():
    X = df.drop(columns=['charges', 'bmi'])
    y = df['charges']
    scaler = StandardScaler()
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_Train_scaled = scaler.fit_transform(X_Train)
    X_Test_scaled = scaler.transform(X_Test)

    selected_model = st.selectbox("Select Model for Evaluation", list(MODELS.keys()))
    
    train_model = MODELS[selected_model]
    model = train_model(X_Train_scaled, y_Train)
    y_Pred = model.predict(X_Test_scaled)
    rmse = root_mean_squared_error(y_Test, y_Pred)
    mape = mean_absolute_percentage_error(y_Test, y_Pred)

    st.subheader(f"{selected_model} Model Performance")
    st.write(f"Root Mean Squared Error: {rmse:.2f}")
    st.write(f"Mean Absolute Percentage Error: {mape:.2f}")
    st.write(f"R2 Score: {r2_score(y_Test, y_Pred):.4f}")

    # Visualization of Predictions vs Actual
    st.subheader("Model Predictions vs Actual Charges")

    plot_df = pd.DataFrame({'Actual Charges': y_Test, 'Predicted Charges': y_Pred})

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x = 'Actual Charges',
                    y = 'Predicted Charges', 
                    data=plot_df,
                    ax=ax,
                    hue='Actual Charges',
                    palette='viridis')

    max_charge = max(y_Test.max(), y_Pred.max())
    min_charge = min(y_Test.min(), y_Pred.min())
    ax.plot([min_charge, max_charge], 
            [min_charge, max_charge],
            color = "red",
            linestyle='--',
            linewidth=2,
            label='Perfect Prediction'
    )

    ax.set_title('Actual vs Predicted Charges')
    ax.set_xlabel('Actual Insurance Charges($)')
    ax.set_ylabel('Predicted Insurance Charges($)')
    ax.legend(loc = 'lower right')
    plt.tight_layout()
    st.pyplot(fig)

    if selected_model in ['Gradient Boosting', 'Random Forest']:
        feature_importances = model.feature_importances_
        features = X.columns

        fi_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)

        st.subheader("Feature Importances")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax2, palette='viridis')
        ax2.set_title(f'Feature Importances from {selected_model} Model')
        plt.tight_layout()
        st.pyplot(fig2)






with st.spinner("Generating visualizations..."):
    def CV_page():
        st.title("K-Fold Cross Validation")
        selected_model = st.selectbox("Select Model for K-Fold Cross Validation", list(MODELS.keys()))

        @st.cache_resource
        def n_fold_cross_validation(model, K):
            X = df.drop(columns=['charges', 'bmi'])
            y = df['charges']
            scaler = StandardScaler()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            kfold = KFold(n_splits=K, shuffle=True, random_state=42)       
            selected_model1 = MODELS1[model]
            # train_model = MODELS[selected_model]

            avg_mae = 0
            avg_rmse = 0
            avg_mape = 0

            for train_index, val_index in kfold.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
                pipeline = Pipeline([('scaler',scaler), ('model', selected_model1)])
                y_train_log = np.log(y_train_fold)
                model = pipeline.fit(X_train_fold, y_train_log)
                y_pred_log = model.predict(X_val_fold)
                y_pred = np.exp(y_pred_log)
                mae = mean_absolute_error(y_val_fold, y_pred)
                rmse = root_mean_squared_error(y_val_fold, y_pred)
                mape = mean_absolute_percentage_error(y_val_fold, y_pred)
                r2_score_val = r2_score(y_val_fold, y_pred)
                avg_mae += mae
                avg_rmse += rmse
                avg_mape += mape

                avg_rmse /= K
                avg_mape /= K

            
            # st.subheader(f"{selected_model} K_fold Cross Validation Performance")
            # st.write(f"Average RMSE: {avg_rmse:.2f}")
            # st.write(f"Average MAPE: {avg_mape:.2f}")
            # st.write(f"R2 Score: {r2_score_val:.4f}")
            
            return [avg_rmse, avg_mape, r2_score_val]


        EVAL_METRICS = {}
        st.subheader("plot of CV results")
        

        EVAL_METRICS = {
            'RMSE': 0,
            'MAPE': 1
        }
        eval_metric = st.selectbox("Select Evaluation Metric for Visualization", ['RMSE', 'MAPE'])

        with st.spinner("Generating visualizations..."):    
            CV_results = []
            for k in range(2, 15):
                metric = n_fold_cross_validation(selected_model, k)[EVAL_METRICS[eval_metric]]
                CV_results.append({'K' : k, eval_metric : metric})
            
            CV_df = pd.DataFrame(CV_results)
            fig, ax = plt.subplots(figsize=(8, 5))

            sns.lineplot(x = 'K', y = eval_metric, data = CV_df, marker='o', ax=ax)
            ax.set_title(f"{selected_model} K-fold Cross Validation {eval_metric} vs K")
            ax.set_xlabel("Number of Folds (K)")
            ax.set_ylabel(eval_metric)
            ax.legend()
            st.pyplot(fig)

        
        def slider_metrics():
            st.subheader("Select number of folds (K) to see CV metrics")
            K_value = st.slider("select number of folds (K)", min_value=2, max_value=15, value=5, step=1)
            metric_values = n_fold_cross_validation(selected_model, K_value)
            st.dataframe({
                'Metric':['RMSE', 'MAPE', 'R2 Score'],
                'Value' :[round(metric_values[0], 2), round(metric_values[1], 2), round(metric_values[2], 4)],
            }, use_container_width=True)
        slider_metrics()
    



#Page Selection

if selection == "Homepage (SDE)":
    homepage()
elif selection == "Model Evaluation":
    model_evaluation()
elif selection == "Pairwise Plots":
    pairwise_plots()
elif selection == "K-Fold Cross Validation":
    CV_page()





