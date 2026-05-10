import pandas as pd

def create_features(df):

    new_df = df.copy()

    #Glucose-Insulin Balance
    new_df['Glucose_Insulin_Ratio'] = new_df['Glucose'] / new_df['Insulin']
    
    #0: Underweight, 1: Normal Weight, 2: Overweight, 3: Class 1 Obesity, 4: Class 2 Obesity, 5: Class 3 Obesity
    #Nuttall, F. Q. (2015). Body mass index: obesity, BMI, and health: a critical review. Nutrition today, 50(3), 117-128.
    new_df['BMI_Category'] = pd.cut(new_df['BMI'], 
                                    bins=[0, 19.9, 24.9, 29.9, 34.9, 39.9, 100], 
                                    labels=[0, 1, 2, 3, 4, 5]).astype(int)
    
    #0: Young, 1: Middle-aged, 2: Senior
    #Beard, J. R., Officer, A. M., & Cassels, A. K. (2016). The world report on ageing and health. The Gerontologist, 56(Suppl_2), S163-S166.
    new_df['Age_Risk'] = pd.cut(new_df['Age'], 
                                bins=[0, 30, 50, 150], 
                                labels=[0, 1, 2]).astype(int)
    
    #Glucose-BMI Interaction
    new_df['Glucose_BMI_Interaction'] = new_df['Glucose'] * new_df['BMI']

    #Pregnancies-Age Interaction
    new_df['Pregnancies_Age_Interaction'] = new_df['Pregnancies'] * new_df['Age']

    return new_df