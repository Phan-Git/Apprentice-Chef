# timeit

# Student Name : NHAT PHAN
# Cohort       : FMSBA2

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import gender_guesser.detector as gender

import warnings
warnings.filterwarnings("ignore")
from time import process_time
time_start = process_time()  

################################################################################
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

file = 'Apprentice_Chef_Dataset.xlsx'
original_df = pd.read_excel(file)

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# use this space for all of the feature engineering that is required for your
# final model

# if your final model requires dataset standardization, do this here as well
for col in original_df:

    # creating columns with 1s if missing and 0 if not
    if original_df[col].isnull().astype(int).sum() > 0:
        original_df['m_'+col] = original_df[col].isnull().astype(int)

original_df['CLASS_FAMILY_NAME'] = original_df['FAMILY_NAME']
original_df['CLASS_FAMILY_NAME'] = pd.notnull(original_df['CLASS_FAMILY_NAME'])
original_df['CLASS_FAMILY_NAME'] = original_df['CLASS_FAMILY_NAME'].astype(int)

original_df['EXP_PER_ORDER'] = original_df['REVENUE'] / original_df['TOTAL_MEALS_ORDERED']

original_df['LARGEST_EXP'] = original_df['EXP_PER_ORDER'] * original_df['LARGEST_ORDER_SIZE']

original_df['SPEND_ON_CATEGORIES'] = original_df['REVENUE'] / original_df['UNIQUE_MEALS_PURCH']

original_df['ORDERS_OVER_VIEWS'] = original_df['UNIQUE_MEALS_PURCH'] / original_df['PRODUCT_CATEGORIES_VIEWED']

original_df['REV_LOS_FOR_CANCEL'] = original_df['EXP_PER_ORDER'] * (original_df['CANCELLATIONS_BEFORE_NOON'] + 0.5*original_df['CANCELLATIONS_AFTER_NOON'])

placeholder_lst = []

for index, col in original_df.iterrows():
    
    # splitting email domain at '@'
    split_email = original_df.loc[index, 'EMAIL'].split(sep = "@")
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

email_df = pd.DataFrame(placeholder_lst, columns = ['Name','Domain'])
original_df_concat = pd.concat([original_df, email_df.loc[:,'Domain']],
                   axis = 1)
# we now create lists of categories.
personal_email_domains = ['gmail.com','yahoo.com','protonmail.com']
junk_email_domains = ['me.com','aol.com','hotmail.com', 'live.com', 'msn.com', 'passport.com']
professional_email_domains = ['mmm.com', 'amex.com', 'apple.com',
'boeing.com', 'caterpillar.com', 'chevron.com', 
'cisco.com', 'cocacola.com', 'disney.com', 'dupont.com', 
'exxon.com', 'ge.org', 'goldmansacs.com', 'homedepot.com', 
'ibm.com', 'intel.com ','jnj.com', 'jpmorgan.com', 'mcdonalds.com', 
'merck.com', 'microsoft.com', 'nike.com','pfizer.com', 'pg.com', 
'travelers.com', 'unitedtech.com', 'unitedhealth.com', 'verizon.com',
'visa.com', 'walmart.com']

# placeholder list
placeholder_lst = []


# looping is applying to group observations
for domain in original_df_concat['Domain']:
        if domain in personal_email_domains:
            placeholder_lst.append('personal')
            
        elif domain in junk_email_domains :
            placeholder_lst.append('junk')
        
        elif domain in professional_email_domains:
            placeholder_lst.append('professional')



# concatenating with original DataFrame
original_df_concat['domain_group'] = pd.Series(placeholder_lst)

for domain in original_df_concat['Domain'].tail(n = 17):
        if domain in personal_email_domains:
            placeholder_lst.append('personal')
            
        elif domain in junk_email_domains :
            placeholder_lst.append('junk')
        
        elif domain in professional_email_domains:
            placeholder_lst.append('professional')



# concatenating with original DataFrame
original_df_concat['domain_group']= pd.Series(placeholder_lst)

one_hot_email = pd.get_dummies(original_df_concat['domain_group'])
one_hot_follow = pd.get_dummies(original_df_concat['FOLLOWED_RECOMMENDATIONS_PCT'])

#dropping
original_df_concat = original_df_concat.drop('domain_group', axis = 1)
original_df_concat = original_df_concat.drop('FOLLOWED_RECOMMENDATIONS_PCT', axis = 1)

#joining
original_df_concat = original_df_concat.join([one_hot_email,one_hot_follow])

original_df_explanatory = original_df_concat.copy()


# dropping SalePrice and Order from the explanatory variable set
original_df_explanatory = original_df_explanatory.drop(columns = ['REVENUE','EMAIL','FIRST_NAME','FAMILY_NAME','Domain','NAME'], axis = 1)

TOTAL_MEALS_ORDERED_change = 250
AVG_PREP_VID_TIME_change   = 270
LARGEST_ORDER_SIZE_change  = 7
LARGEST_EXP_change         = 400
MEDIAN_MEAL_RATING_change  = 4
CONTACTS_W_CUSTOMER_SERVICE_change = 10
UNIQUE_MEALS_PURCH_change          = 7
MASTER_CLASSES_ATTENDED_change     = 2
EXP_PER_ORDER_change               = 55
REV_LOS_FOR_CANCEL_change          = 200
SPEND_ON_CATEGORIES_change         = 4000               
                  

# greater than sign

original_df_concat['change_TOTAL_MEALS_ORDERED'] = 0
condition = original_df_concat.loc[0:,'change_TOTAL_MEALS_ORDERED'][original_df_concat['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_change]

original_df_concat['change_TOTAL_MEALS_ORDERED'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# REV_LOS_FOR_CANCEL
original_df_concat['change_REV_LOS_FOR_CANCEL'] = 0
condition = original_df_concat.loc[0:,'change_REV_LOS_FOR_CANCEL'][original_df_concat['REV_LOS_FOR_CANCEL'] > REV_LOS_FOR_CANCEL_change]

original_df_concat['change_REV_LOS_FOR_CANCEL'].replace(to_replace = condition,
                                value      = 1,
                                inplace    = True)

# SPEND_ON_CATEGORIES
original_df_concat['change_SPEND_ON_CATEGORIES'] = 0
condition = original_df_concat.loc[0:,'change_SPEND_ON_CATEGORIES'][original_df_concat['SPEND_ON_CATEGORIES'] > SPEND_ON_CATEGORIES_change]

original_df_concat['change_SPEND_ON_CATEGORIES'].replace(to_replace = condition,
                                value      = 1,
                                inplace    = True)


# AVG_PREP_VID_TIME
original_df_concat['change_AVG_PREP_VID_TIME'] = 0
condition = original_df_concat.loc[0:,'change_AVG_PREP_VID_TIME'][original_df_concat['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_change]

original_df_concat['change_AVG_PREP_VID_TIME'].replace(to_replace = condition,
                                value      = 1,
                                inplace    = True)


# LARGEST_ORDER_SIZE
original_df_concat['change_LARGEST_ORDER_SIZE'] = 0

condition = original_df_concat.loc[0:,'change_LARGEST_ORDER_SIZE'][original_df_concat['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_change]

original_df_concat['change_LARGEST_ORDER_SIZE'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)

# LARGEST_EXP
original_df_concat['change_LARGEST_EXP'] = 0

condition = original_df_concat.loc[0:,'change_LARGEST_EXP'][original_df_concat['LARGEST_EXP'] > LARGEST_EXP_change]

original_df_concat['change_LARGEST_EXP'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
original_df_concat['change_CONTACTS_W_CUSTOMER_SERVICE'] = 0

condition = original_df_concat.loc[0:,'change_CONTACTS_W_CUSTOMER_SERVICE'][original_df_concat['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_change]

original_df_concat['change_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)

# EXP_PER_ORDER
original_df_concat['change_EXP_PER_ORDER'] = 0

condition = original_df_concat.loc[0:,'change_EXP_PER_ORDER'][original_df_concat['EXP_PER_ORDER'] > EXP_PER_ORDER_change]

original_df_concat['change_EXP_PER_ORDER'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)

# double-equals sign

# MEDIAN_MEAL_RATING
original_df_concat['change_MEDIAN_MEAL_RATING'] = 0
condition = original_df_concat.loc[0:,'change_MEDIAN_MEAL_RATING'][original_df_concat['MEDIAN_MEAL_RATING'] == MEDIAN_MEAL_RATING_change]

original_df_concat['change_MEDIAN_MEAL_RATING'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# UNIQUE_MEALS_PURCH
original_df_concat['change_UNIQUE_MEALS_PURCH'] = 0
condition = original_df_concat.loc[0:,'change_UNIQUE_MEALS_PURCH'][original_df_concat['UNIQUE_MEALS_PURCH'] == UNIQUE_MEALS_PURCH_change]

original_df_concat['change_UNIQUE_MEALS_PURCH'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)
                   
                   
# MASTER_CLASSES_ATTENDED
original_df_concat['change_MASTER_CLASSES_ATTENDED'] = 0
condition = original_df_concat.loc[0:,'change_MASTER_CLASSES_ATTENDED'][original_df_concat['MASTER_CLASSES_ATTENDED'] == MASTER_CLASSES_ATTENDED_change]

original_df_concat['change_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

original_df_explanatory = original_df_concat.copy()


# dropping SalePrice and Order from the explanatory variable set
original_df_explanatory = original_df_explanatory.drop(columns = ['REVENUE','EMAIL','FIRST_NAME','FAMILY_NAME','Domain','NAME'], axis = 1)

# This is the finall OLS run
lm_full = smf.ols(formula = ''' original_df_concat['REVENUE'] ~ original_df_concat['TOTAL_MEALS_ORDERED']+
                                original_df_concat['EXP_PER_ORDER']+
                                original_df_concat['AVG_PREP_VID_TIME']+
                                original_df_concat['MEDIAN_MEAL_RATING']+
                                original_df_concat['CONTACTS_W_CUSTOMER_SERVICE']+
                                original_df_concat['TOTAL_PHOTOS_VIEWED']+
                                original_df_concat['MASTER_CLASSES_ATTENDED']+
                                original_df_concat['UNIQUE_MEALS_PURCH']+
                                original_df_concat['LARGEST_ORDER_SIZE']+
                                original_df_concat['EXP_PER_ORDER']+
                                original_df_concat['LARGEST_EXP']+
                                original_df_concat['SPEND_ON_CATEGORIES']+
                                original_df_concat['ORDERS_OVER_VIEWS']+
                                original_df_concat['change_TOTAL_MEALS_ORDERED']+
                                original_df_concat['change_AVG_PREP_VID_TIME']+
                                original_df_concat['change_LARGEST_ORDER_SIZE']+
                                original_df_concat['change_LARGEST_EXP']+
                                original_df_concat['change_CONTACTS_W_CUSTOMER_SERVICE']+
                                original_df_concat['change_EXP_PER_ORDER']+
                                original_df_concat['change_MEDIAN_MEAL_RATING']+
                                original_df_concat['change_UNIQUE_MEALS_PURCH']+
                                original_df_concat['change_MASTER_CLASSES_ATTENDED']+
                                original_df_concat['change_SPEND_ON_CATEGORIES']+ 
                                original_df_concat['junk']+
                                original_df_concat['personal']+
                                original_df_concat['professional']+
                                original_df_concat[0]+
                                original_df_concat[10]+
                                original_df_concat[20]+
                                original_df_concat[30]+
                                original_df_concat[40]+
                                original_df_concat[50]+
                                original_df_concat[60]+
                                original_df_concat[70]+
                                original_df_concat[80]+
                                original_df_concat[90]''',
                                data = original_df_concat)


# telling Python to run the data through the blueprint
results_full = lm_full.fit()

         


x_variables = [ 'TOTAL_MEALS_ORDERED','EXP_PER_ORDER','AVG_PREP_VID_TIME',
                'MEDIAN_MEAL_RATING','CONTACTS_W_CUSTOMER_SERVICE',
                'TOTAL_PHOTOS_VIEWED','MASTER_CLASSES_ATTENDED', 
                'UNIQUE_MEALS_PURCH','LARGEST_ORDER_SIZE',
                'LARGEST_EXP',
                'SPEND_ON_CATEGORIES',
                'ORDERS_OVER_VIEWS',
                'change_TOTAL_MEALS_ORDERED',
                'change_AVG_PREP_VID_TIME',
                'change_LARGEST_ORDER_SIZE',
                'change_LARGEST_EXP',
                'change_CONTACTS_W_CUSTOMER_SERVICE',
                'change_EXP_PER_ORDER',
                'change_MEDIAN_MEAL_RATING',
                'change_UNIQUE_MEALS_PURCH',
                'change_MASTER_CLASSES_ATTENDED',
                'change_SPEND_ON_CATEGORIES',
                'junk','personal','professional',
                0,10,20,30,40,50,60,70,80,90]


original_df_data = original_df_concat.loc[:, x_variables]

original_df_target = original_df_concat.loc[:, 'REVENUE']

# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()


# FITTING the scaler with housing_data
scaler.fit(original_df_data)


# TRANSFORMING our data after fit
X_scaled = scaler.transform(original_df_data)


# converting scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)


# checking the results
X_scaled_df.describe().round(2)

X_scaled_df.columns = original_df_data.columns


print(f"""

Dataset after STANDARDIZATION
------------------------------
{pd.np.var(X_scaled_df)}
""")

# correlation matrix
fig, ax = plt.subplots(figsize = (5, 8))

df_scaled_corr = X_scaled_df.loc[ : , ['TOTAL_MEALS_ORDERED','EXP_PER_ORDER','AVG_PREP_VID_TIME',
                                        'MEDIAN_MEAL_RATING','CONTACTS_W_CUSTOMER_SERVICE',
                                        'TOTAL_PHOTOS_VIEWED','MASTER_CLASSES_ATTENDED', 
                                        'UNIQUE_MEALS_PURCH','LARGEST_ORDER_SIZE',
                                        'LARGEST_EXP',
                                        'SPEND_ON_CATEGORIES',
                                        'ORDERS_OVER_VIEWS']].corr().round(2)

bottom, top = plt.ylim() # discover the values for bottom and top
bottom += 0.5            # add 0.5 to the bottom
top -= 0.5               # subtract 0.5 from the top
plt.ylim(bottom, top)    # update the ylim(bottom, top) values
# titling the plot
plt.title("BEFORE Standardization")



# heatmap of SCALED correlations
sns.heatmap(df_scaled_corr,
            cmap = 'Greens',
            square = True,
            annot = True,
            cbar = False) # surpressing the color bar
bottom, top = plt.ylim() # discover the values for bottom and top
bottom += 0.5            # add 0.5 to the bottom
top -= 0.5               # subtract 0.5 from the top
plt.ylim(bottom, top)    # update the ylim(bottom, top) values

# titling the plot
plt.title("AFTER Standardization")
plt.show()
################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# Note: Be sure to set test_size = 0.25

# we now split the data
X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df,
            original_df_target,
            test_size = 0.25,
            random_state = 222)



################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model
# LR
lr = LinearRegression()
lr_fit = lr.fit(X_train, y_train)
lr_pred = lr_fit.predict(X_test)

################################################################################

# developing a residual plot
fig, ax = plt.subplots(figsize = (8,5))
sns.residplot(x = lr_pred,
              y = y_test,
             color = "r") 
plt.xlabel('Linear Regression Model OLS')
plt.tight_layout()
plt.show()


################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)
print('Training Score:', lr.score(X_train, y_train).round(3))
print('Testing Score:', lr.score(X_test, y_test).round(3))

train_score = lr.score(X_train, y_train).round(3)
test_score = lr.score(X_test, y_test).round(3)

print("The final model is Ordinary Least Square _ Linear Regression")
time_stop = process_time() 
   
print("Elapsed time:", time_stop, time_start)  
print("Elapsed time during the file proccessing in seconds is :", 
                                         time_stop-time_start)
print(""" THANK YOU VERY MUCH & HAPPY CHINESE NEW YEAR ! 
      REALLY APPRECIATE YOUR EFFORT FOR US. 
      YOU ARE MY INSPIRATION TO CHOOSE BUSINESS ANALYST CAREER - PHAN NHAT, 2020""")

