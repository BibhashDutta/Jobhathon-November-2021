# Jobhathon-November-2021
Employee attrition problem
# Problem statement
You are working as a data scientist with HR Department of a large insurance company focused on sales team attrition. Insurance sales teams help insurance companies generate new business by contacting potential customers and selling one or more types of insurance. The department generally sees high attrition and thus staffing becomes a crucial aspect.

To aid staffing, you are provided with the monthly information for a segment of employees for 2016 and 2017 and tasked to predict whether a current employee will be leaving the organization in the upcoming two quarters (01 Jan 2018 - 01 July 2018) or not, given:

Demographics of the employee (city, age, gender etc.)
Tenure information (joining date, Last Date)
Historical data regarding the performance of the employee (Quarterly rating, Monthly business acquired, designation, salary)
Private Leaderboard Rank - 61
Public Leaderboard Rank - 138

# Approach
As the objective was to predict if an employee will leave the organization in the upcoming two quarters, the target variable was taken such that if an employee leaves the organization within 180 days of review it was taken was 1 and 0 otherwise. In train data only January, february and march data was given and using that data we needed to predict if the employee is going to leave the company in the next quarter or not. Since last joining date had many null values as many people did not leave the company and we needed to predict for these employees if they'll leave or not, so first we made our target variable out of these column of last joining date. Then filtered those with label as 0 on which we needed to predict churn and used the data with lable 1 and 0 both were used to train our model.

# Feature Engineering
I first grouped the data according to employee id as three rows of data were given for each employee and made a whole new dataframe with the target variable.
After working on the individual columns and then encoding categorical columns we produced 44 columns, in which 9 features were given initially and left were engineered which were:-
'Age', 'Salary', 'Total Business Value', 'rating', 'Gender_Male','City_C10', 'City_C11', 'City_C12', 'City_C13', 'City_C14', 'City_C15','City_C16', 'City_C17', 'City_C18', 'City_C19', 'City_C2', 'City_C20','City_C21', 'City_C22', 'City_C23', 'City_C24', 'City_C25', 'City_C26','City_C27', 'City_C28', 'City_C29', 'City_C3', 'City_C4', 'City_C5','City_C6', 'City_C7', 'City_C8', 'City_C9', 'Education_Level_College','Education_Level_Master', 'Joining Designation_2''Joining Designation_3', 'Joining Designation_4',
'Joining Designation_5', 'Designation_2', 'Designation_3','Designation_4', 'Designation_5', 'Target'.
Note: here we even engineered the target variable out of last working date.

# Model Building
XGBoostClassifier was used for modelling and it gave the highest F1 score out of all the classifiers that i used. Changing models from RandomForestClassier to LGBM clasifier to catboost then XGboost helped in improving the score but even then there was scope for improvement, so i carefully engineered columns again to get better results and used GridSearchCV to tune some hyperparameters to get the best result.
