import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data = pd.read_csv('bank-additional-full.csv')

data.head()

dataset = data['''age;"job";"marital";"education";"default";"housing";"loan";"contact";"month";"day_of_week";"duration";"campaign";"pdays";"previous";"poutcome";"emp.var.rate";"cons.price.idx";"cons.conf.idx";"euribor3m";"nr.employed";"y"'''].str.split(';', expand = True)

dataset.rename(columns = {0:"age",
               1: "job",
               2: "marital",
               3: "education",
               4: "default",
               5: "housing",
               6: "loan",
               7: "contact",
               8: "month",
               9: "day_of_week",
               10: "duration",
               11: "campaign",
               12: "pdays",
               13: "previous",
               14: "poutcome",
               15: "emp.var.rate",
               16: "cons.price.idx",
               17: "cons.conf.idx",
               18: "euribor3m",
               19: "nr.employed",
               20: "y"}, inplace = True)

dataset.isnull().sum()

dataset.dtypes

dataset_int = dataset.iloc[:, [0,10,11,12,13,15,16,17,18,19]]
dataset_cat = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,14,20]]

dataset_int = dataset_int.astype(np.number)
dataset_int.dtypes


job_l = []
for i in dataset_cat["job"]:
    job_l.append(i[1:-1])
    
dataset_cat["job"] = pd.Series(job_l)

marital_l = []
for i in dataset_cat["marital"]:
    marital_l.append(i[1:-1])
    
dataset_cat["marital"] = pd.Series(marital_l)

education_l = []
for i in dataset_cat["education"]:
    education_l.append(i[1:-1])

dataset_cat["education"] = pd.Series(education_l)

default_l = []
for i in dataset_cat["default"]:
    default_l.append(i[1:-1])
    
dataset_cat["default"] = pd.Series(default_l)

housing_l = []
for i in dataset_cat["housing"]:
    housing_l.append(i[1:-1])

dataset_cat["housing"] = pd.Series(housing_l)    

loan_l = []
for i in dataset_cat["loan"]:
    loan_l.append(i[1:-1])
  
dataset_cat["loan"] = pd.Series(loan_l)

contact_l = []
for i in dataset_cat["contact"]:
    contact_l.append(i[1:-1])
    
dataset_cat["contact"] = pd.Series(contact_l)

month_l = []
for i in dataset_cat["month"]:
    month_l.append(i[1:-1])
    
dataset_cat["month"] = pd.Series(month_l)

day_of_week_l = []
for i in dataset_cat["day_of_week"]:
    day_of_week_l.append(i[1:-1])
    
dataset_cat["day_of_week"] = pd.Series(day_of_week_l)

poutcome_l = []
for i in dataset_cat["poutcome"]:
    poutcome_l.append(i[1:-1])
    
dataset_cat["poutcome"] = pd.Series(poutcome_l)

y_l = []
for i in dataset["y"]:
    y_l.append(i[1:-1])
    
dataset_cat["y"] = pd.Series(y_l)

dataset_cat.isnull().sum()

########################################
# Treatment of Outliers:
#######################################

dataset_int.isnull().sum()
dataset_int.boxplot(column = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx","euribor3m", "nr.employed"])
plt.show()
    
#Since this is a classification problem, and we will be using 
    #tree based algos as well, hence we will not treat the outliers

########################################################
# Checking for value counts for categorical variables:
########################################################

dataset_cat.isnull().sum()

dataset_cat["job"].value_counts().plot(kind = "bar")
plt.show()

dataset_cat["marital"].value_counts().plot(kind = "bar")
plt.show()

dataset_cat["education"].value_counts().plot(kind = "bar")
plt.show()

dataset_cat["default"].value_counts().plot(kind = "bar")
plt.show()

dataset_cat["housing"].value_counts().plot(kind = "bar")
plt.show()

dataset_cat["loan"].value_counts().plot(kind = "bar")
plt.show()

dataset_cat["contact"].value_counts().plot(kind = "bar")
plt.show()

dataset_cat["month"].value_counts().plot(kind = "bar")
plt.show()

dataset_cat["day_of_week"].value_counts().plot(kind = "bar")
plt.show()

dataset_cat["poutcome"].value_counts().plot(kind = "bar")
plt.show()

dataset_cat["y"].value_counts().plot(kind = "bar")
plt.show()

dataset_cat["y"].value_counts(normalize = True)

# It can be seen that there is some disbalance in the target 
    # variable, but since the ratio in not around 95:5, we will 
    # not treat it in any way.

# Further, it can be seen that there are null values in for of 
    #"unknown" in the dataset in the following columns:
        # job
        # marital
        # education
        # default
        # housing
        # loan

##########################################################
# Finding relationship of variables with target variable:
##########################################################

# Hypothesis testing(Chi-square):
    # H0: Variables are independent of each other
    # H1: Variables are related to each other

for i in dataset_cat.loc[:,["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]]:
    teststat, p, df, exp = stats.chi2_contingency(pd.crosstab(dataset_cat[i], dataset_cat["y"]))
    print(i, ":", p)
    if p < 0.05:
        print("We will REJECT the null hypothesis and the two variables are related")
    else:
        print("FAIL TO REJECT the null hypothesis and the two variables are independent of each other.")
    print(" ")
    
# Basis the above Chi-square test, we can say that the following 
    # categorical columns are realted to the target column:
        
        # job
        # marital
        # education
        # default
        # contact
        # month
        # day_of_week
        # poutcome
        
# Now we will split the dataset into train and test and then we will do the 
    #treatment of the null values.
        
####################################################
# Dividing the dataset into X and y variables:
####################################################

X = pd.concat([dataset_int, dataset_cat.drop('y', axis = 1)], axis = 1)
y = dataset_cat.loc[:,['y']]
        
####################################################
# Splitting the data into train and test set:
####################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2) 

#######################################################
# Now we'll work on treatment on X_train and y_train:       
#######################################################

X_train_int = X_train.select_dtypes(include = np.number)
X_train_cat = X_train.select_dtypes(exclude = np.number)

X_train_int.isnull().sum()
X_train_cat.isnull().sum()

X_train_int.reset_index(drop = True, inplace = True)

# No treatment for int columns required, since no null values are present.

# We will treat the categorical columns in the following way:
    # job: dummy / le
    # marital: dummy
    # education: dummy
    # default: LE - "no":1, "yes":0
    # housing: LE - "no":1, "yes":0
    # loan: LE - "no":1, "yes":0
    # contact: dummy
    # month: dummy
    # day_of_week: dummy
    # poutcome: dummy / LE
    # y: LE - "yes":1, "no":0
    
# Replacing the 'unknown' with null:
X_train_cat = X_train_cat.replace('unknown', np.nan)

X_train_cat.isnull().sum() / len(X_train_cat) * 100

X_train_cat_columns = X_train_cat.columns

# Replacing the null values in the categorical column with mode(most frequent):

from sklearn.impute import SimpleImputer
sim_cat = SimpleImputer(strategy = 'most_frequent')
X_train_cat = sim_cat.fit_transform(X_train_cat)
X_train_cat = pd.DataFrame(X_train_cat, columns = X_train_cat_columns)

# Sanity check for null values:
    
X_train_cat.isnull().sum() / len(X_train_cat) * 100

    # Hence, null values have been treated.

# Dividing the X_train_cat into two parts:
    # X_train_cat_dummy :  for dummy treatment
    # X_train_cat_le : for LE treatemnt

X_train_cat_dummy = X_train_cat.loc[:,['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']]
X_train_cat_le = X_train_cat.loc[:,['default', 'housing', 'loan']]

X_train_cat_dummy.isnull().sum()
X_train_cat_le.isnull().sum()

X_train_cat_dummy = pd.get_dummies(X_train_cat_dummy)
X_train_cat_dummy.reset_index(drop = True, inplace = True)

X_train_cat_le = X_train_cat_le.replace(['no', 'yes'], [1,0])
X_train_cat_le.reset_index(drop = True, inplace = True)

# Now combining all the above treated columns to make final X_train:
    
X_train_final = pd.concat([X_train_int, X_train_cat_dummy, X_train_cat_le], axis = 1)

X_train_f = X_train_final.values

# Now considering y_train:
    
y_train.isnull().sum()
    
    # As can be seen, there are no null values present in the y_train.
    
# Label Encoding of y_train:
    
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

np.unique(y_train) # Sanity check

############################################
# Model Building:
############################################

# Note: We will be using K-Fold to check whether overfitting is being prevented or not

from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)

# 1. DECISION TREE:
    
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()

scores_dt = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_f[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    dtf.fit(X_train_idx, y_train_idx)
    scores_dt.append(dtf.score(X_test_idx, y_test_idx))

print(scores_dt)

# Output of scores_dt:
    
# [0.8892261001517451, 0.8854324734446131, 0.8849772382397572, 0.8904400606980273, 0.8880121396054628]

# 2. LOGISTIC REGRESSION:
    
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

scores_lr = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_f[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    log_reg.fit(X_train_idx, y_train_idx)
    scores_lr.append(log_reg.score(X_test_idx, y_test_idx))
    
print(scores_lr)
# output of score_lr:
    
# [0.9115326251896814, 0.9080424886191198, 0.906980273141123, 0.9122913505311078, 0.9103186646433991]

# 3.GAUSSINA NAIVE BAYES:

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

scores_nb = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_f[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    nb.fit(X_train_idx, y_train_idx)
    scores_nb.append(nb.score(X_test_idx, y_test_idx))
    
print(scores_nb)

# Output of scores_nb:

# [0.9115326251896814, 0.9080424886191198, 0.906980273141123, 0.9122913505311078, 0.9103186646433991]

# 4. RANDOM FOREST:
    
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

scores_rfc = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_f[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    rfc.fit(X_train_idx, y_train_idx)
    scores_rfc.append(rfc.score(X_test_idx, y_test_idx))
    
print(scores_rfc)

# Output of scores_rfc:
    
# [0.9153262518968134, 0.9151745068285281, 0.9071320182094081, 0.9142640364188164, 0.9101669195751139]

# 5. ADA BOOST CLASSIFIER:
    
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()

scores_ada = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_idx[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    ada.fit(X_train_idx, y_train_idx)
    scores_ada.append(ada.score(X_test_idx, y_test_idx))
    
print(scores_ada)

# Output of scores_ada:

# [0.9128983308042489, 0.8358118361153263, 0.8311077389984826, 0.8342943854324735]

# 6. GRADIENT BOOSTING CLASSIFIER:

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()

scores_gbc = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_f[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    gbc.fit(X_train_idx, y_train_idx)
    scores_gbc.append(gbc.score(X_test_idx, y_test_idx))
    
print(scores_gbc)

# Output of scores_gbc:
    
# [0.9191198786039454, 0.9174506828528073, 0.9130500758725342, 0.9163884673748103, 0.9144157814871017]


##################################################
# Conclusion of above algos:
##################################################

# From the above algos, it can be seen that the scores of Logisitic Regression,
    # Naive Bayes, Random Forest and Gradient Boosting Classifier have close
    # enough scores.
    

from sklearn.metrics import roc_auc_score, roc_curve

y_score_lr = log_reg.decision_function(X_train_f)    
print(y_score_lr)

y_score_nb = nb.predict_proba(X_train_f)
y_score_nb = y_score_nb[:,1]
print(y_score_nb)

y_score_rfc = rfc.predict_proba(X_train_f)
y_score_rfc = y_score_rfc[:,1]
print(y_score_rfc)

y_score_gbc = gbc.predict_proba(X_train_f)
y_score_gbc = y_score_gbc[:,1]
print(y_score_gbc)

print(roc_auc_score(y_train,y_score_lr))
print(roc_auc_score(y_train, y_score_nb))
print(roc_auc_score(y_train, y_score_rfc))
print(roc_auc_score(y_train, y_score_gbc))

fpr_lr, tpr_lr, th_lr = roc_curve(y_train, y_score_lr)
fpr_nb, tpr_nb, th_nb = roc_curve(y_train, y_score_nb)
fpr_rfc, tpr_rfc, th_rfc = roc_curve(y_train, y_score_rfc)
fpr_gbc, tpr_gbc, th_gbc = roc_curve(y_train, y_score_gbc)

# Plotting of ROC-AUC CURVE:
    
plt.plot(fpr_lr, tpr_lr, label = 'Logistic Regression')
plt.plot(fpr_nb, tpr_nb, label = "Gaussian Naive Bayes")
plt.plot(fpr_rfc, tpr_rfc, label = "Random Forest")
plt.plot(fpr_gbc, tpr_gbc, label = "Gradient Boosting")
plt.plot([0,1],[0,1])
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Poso=itive Rate(Sensitivity')
plt.title('ROC Curve for predicting acceptance of Term Deposit')
plt.legend()
plt.show()

# Basis the above ROC- Curve, it can be clearly seen that Random Forest has 
    # best ditinguishing ability
    
######################################################
# Making necessary changes in X_test and y_test set:
######################################################

X_test_int = X_test.select_dtypes(include = np.number)
X_test_cat = X_test.select_dtypes(exclude = np.number)

X_test_int.isnull().sum()
X_test_cat.isnull().sum()


X_test_int.reset_index(drop = True, inplace = True)

# No treatment for int columns required, since no null values are present.

# We will treat the categorical columns in the following way:
    # job: dummy / le
    # marital: dummy
    # education: dummy
    # default: LE - "no":1, "yes":0
    # housing: LE - "no":1, "yes":0
    # loan: LE - "no":1, "yes":0
    # contact: dummy
    # month: dummy
    # day_of_week: dummy
    # poutcome: dummy / LE
    # y: LE - "yes":1, "no":0
    
# Replacing the 'unknown' with null:
X_test_cat = X_test_cat.replace('unknown', np.nan)

X_test_cat.isnull().sum() / len(X_test_cat) * 100

X_test_cat_columns = X_test_cat.columns

# Replacing the null values in the categorical column with mode(most frequent):

from sklearn.impute import SimpleImputer
sim_cat = SimpleImputer(strategy = 'most_frequent')
X_test_cat = sim_cat.fit_transform(X_test_cat)
X_test_cat = pd.DataFrame(X_test_cat, columns = X_test_cat_columns)

# Sanity check for null values:
    
X_test_cat.isnull().sum() / len(X_test_cat) * 100

    # Hence, null values have been treated.

# Dividing the X_test_cat into two parts:
    # X_test_cat_dummy :  for dummy treatment
    # X_test_cat_le : for LE treatemnt

X_test_cat_dummy = X_test_cat.loc[:,['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']]
X_test_cat_le = X_test_cat.loc[:,['default', 'housing', 'loan']]

X_test_cat_dummy.isnull().sum()
X_test_cat_le.isnull().sum()

X_test_cat_dummy = pd.get_dummies(X_test_cat_dummy)
X_test_cat_dummy.reset_index(drop = True, inplace = True)

X_test_cat_le = X_test_cat_le.replace(['no', 'yes'], [1,0])
X_test_cat_le.reset_index(drop = True, inplace = True)

# Now combining all the above treated columns to make final X_train:
    
X_test_final = pd.concat([X_test_int, X_test_cat_dummy, X_test_cat_le], axis = 1)


# Now considering y_test:
    
y_test.isnull().sum()

y_test = le.transform(y_test)

######################################################
# Now checking the model using performance metrics:
######################################################

y_pred_lr = log_reg.predict(X_test_final)
y_pred_nb = nb.predict(X_test_final)
y_pred_rfc = rfc.predict(X_test_final)
y_pred_gbc = gbc.predict(X_test_final)

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix

for i in [accuracy_score, precision_score, recall_score]:
    print(i)
    print(i(y_test, y_pred_lr))
    print(i(y_test, y_pred_nb))
    print(i(y_test, y_pred_rfc))
    print(i(y_test, y_pred_gbc))
    
# Checking ROC-AUC score:
    
y_score_lr_test = log_reg.decision_function(X_test_final)
print(y_score_lr_test)

y_score_nb_test = nb.predict_proba(X_test_final)
y_score_nb_test = y_score_nb_test[:,1]
print(y_score_nb_test)

y_score_rfc_test = rfc.predict_proba(X_test_final)
y_score_rfc_test = y_score_rfc_test[:,1]
print(y_score_rfc_test)

y_score_gbc_test = gbc.predict_proba(X_test_final)
y_score_gbc_test = y_score_gbc_test[:,1]
y_score_gbc_test
print(y_score_gbc_test)

print(roc_auc_score(y_test, y_score_lr_test))
print(roc_auc_score(y_test, y_score_nb_test))
print(roc_auc_score(y_test, y_score_rfc_test))
print(roc_auc_score(y_test, y_score_gbc_test))

# Making of ROC - CURVE:
    
fpr_lr_test, tpr_lr_test, th_lr_test = roc_curve(y_test, y_score_lr_test) 
fpr_nb_test, tpr_nb_test, th_nb_test = roc_curve(y_test, y_score_nb_test)
fpr_rfc_test, tpr_rfc_test, th_rfc_test = roc_curve(y_test, y_score_rfc_test)
fpr_gbc_test, tpr_gbc_test, th_gbc_test = roc_curve(y_test, y_score_gbc_test)

plt.plot(fpr_lr_test, tpr_lr_test, label = "Logistic Regression")
plt.plot(fpr_nb_test, tpr_nb_test, label = "Gaussian Naive Bayes")
plt.plot(fpr_rfc_test, tpr_rfc_test, label = "Random Forest")
plt.plot(fpr_gbc_test, tpr_gbc_test, label = "Gradient Boosting")
plt.plot([0,1],[0,1])
plt.xlabel("False Positive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC curve showing acceptance of term deposit plan")
plt.legend()
plt.show()

# Basis the metric and roc-curve, it can be said that the performance of 
    # Random Forest Classifier and Gradient Boosting Classifier is pretty close
    # and better than all the other algos.