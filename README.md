# ICM
Industrial Copper Modeling
The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data. 
Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.
Modulesimported:
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
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
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
The Solution Approach:
  1. Read the provided excel file and write it to a pandas dataframe.
  2. Filter records with the value in the status column as either 'Won' or 'Lost'
  3. Identify skewness through an sns boxplot
  4. Dropped unnecessary columns
  5. Handled null values in all independant variables
  6. The null values in Selling_price column is filled with the mean value
  7. The material_ref column has some junk values to be removed and updated with null values
  8. The dataset is cleaned and the train test data split is done
  9. The Regression model to find the predicted Selling _Price value
  10. Linear Regression with ridge is used
  11. The classification algorithm is done using different ML models like Logistic Regression,
      RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,SVC,KNeighborsClassifier,DecisionTreeClassifier.
  It is found that the KNeighborsClassifier gives around 0.95 accuracy value and its better for predicting the status of the customer.
