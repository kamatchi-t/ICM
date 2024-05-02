import streamlit as st
from streamlit_option_menu import option_menu
import mysql.connector
st.title("""
         Industrial Copper Modeling
         """)
def main():
    import streamlit as st
    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.express as px
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from streamlit_extras.colored_header import colored_header
    from streamlit_option_menu import option_menu
    from streamlit_dynamic_filters import DynamicFilters
    import plotly.express as px
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    with st.sidebar:
        selected = option_menu("Main Menu", ["Regression","Classification"], 
            icons=[], menu_icon="cast", default_index=1)
        df=pd.DataFrame()
        df=pd.read_excel(r"E:\DataScience\Project5_CopperDSModel\Copper_Set.xlsx")
        df_head=df.head(1000)
        df_part=df_head.loc[df_head['status'].isin(['Won','Lost'])]
        df_map=pd.DataFrame()
        df_map['product_ref']=df_part['product_ref']
        df_map['selling_price']=df_part['selling_price']
        sns.boxplot(data=df_map,x="product_ref",y="selling_price")
        plt.show()
        df_part.drop(['id'],axis=1,inplace=True)
        df_part=df_part.dropna(subset=['item_date','thickness','customer','delivery date','application','country'])
        df_part['selling_price']=df_part['selling_price'].fillna(df_part['selling_price'].mean())
        df_part.loc[df_part['material_ref'].str.contains('000000000',na=False),'material_ref']=np.nan
        df_part['status']=df_part['status'].replace("Won",1)
        
        df_part['status']=df_part['status'].replace("Lost",0)
        
        cols=['application','country','customer','item_date','material_ref',]
        df_part.loc[:,cols] = df_part.loc[:,cols].ffill()
        df_part=pd.get_dummies(df_part,columns=['item type','material_ref'],dtype='int')
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
    if selected=='Regression':
        X=df_part.drop(['selling_price'],axis=1)
        Y=df_part['selling_price']
        # model=LinearRegression()
        # model.fit(X,Y)
        # y_cap=model.predict(X)
        # df_part['pred']=y_cap
        # df_part
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

        # Initialize Linear Regression model
        model = LinearRegression()

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Predict on the testing data
        y_pred = model.predict(X_test)
        df_part['pred']=y_pred
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        st.write("Mean Squared Error:", mse)
        x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
        model=Lasso()
        model.fit(x_train,y_train)
        train_pred=model.predict(x_train)
        test_pred=model.predict(x_test)

        st.write(f"TrainMSE: {mean_squared_error(y_train,train_pred)}")
        st.write(f"TestMSE: {mean_squared_error(y_test,test_pred)}")
        st.write(f"co_eff: {model.coef_}")
    
    if selected=='Classification':
        
        X=df_part.drop(['status'],axis=1)
        y=df_part['status']
        #st.write(X)
        x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
        models=[SVC(),LogisticRegression(),
                KNeighborsClassifier(),
                DecisionTreeClassifier(),RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier()]
        for model in models:
            model.fit(x_train,y_train)
            train_pred=model.predict(x_train)
            test_pred=model.predict(x_test)
            st.write(f"-------------{type(model)}___name___-----------")
            st.write(f"acc: {accuracy_score(y_train,train_pred)}")
            st.write(f"pred: {precision_score(y_train,train_pred,average='micro')}")
            st.write(f"rec: {recall_score(y_train,train_pred,average='micro')}")
            st.write(f"f1: {f1_score(y_train,train_pred,average='micro')}")
            st.write(f"acctest: {accuracy_score(y_test,test_pred)}")
            st.write(f"predtest: {precision_score(y_test,test_pred,average='micro')}")
            st.write(f"rectest: {recall_score(y_test,test_pred,average='micro')}")
            st.write(f"f1test: {f1_score(y_test,test_pred,average='micro')}")
        df=pd.read_excel(r"E:\DataScience\Project5_CopperDSModel\Copper_Set.xlsx")
        df_part=df.head(1000)
        #df_part.drop(['id'],axis=1,inplace=True)

        fig1 = px.density_heatmap(df_part, x="item type", y="status",hover_name ="id",
                                width=1200,height=800,color_continuous_scale ='Greens')
        fig1.show()
        st.plotly_chart(fig1,use_container_width=False)
main()
