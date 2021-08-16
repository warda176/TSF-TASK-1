#!/usr/bin/env python
# coding: utf-8

# # WARDA RAEES

# # GRIP TASK 1

# # Prediction using Supervised ML

# Task statement - Predict the percentage of an student based on the no. of study hours.

# ### Simple Linear Regression With Python scikit learn

# Simple linear regression is used to estimate the relationship between two quantitative variables, here we have percentage of an student( dependent variable) and no. of study hours(independent variable).

# In[84]:


#importing libraries
import pandas as pd
import matplotlib.pyplot as plt


# In[85]:


#loading data file
url="http://bit.ly/w-data"
file=pd.read_csv(url)
print(file)


# ###### plotting our data points on a 2D graph to see the relationship btw both variables

# In[86]:


file.plot(x='Hours', y='Scores', color='blue', style='x')
plt.xlabel('hours studied')
plt.ylabel('score')
plt.title('Hours vs Percentage', color='Grey') 
plt.show()


# from the above plot it can be observed that there is a positive linear relationship btw both variables.

# #### First we split our data into input(X) and output(y)

# In[30]:


#preparing the data for performing regression
X=file.iloc[:,0:-1].values  #printing first column, slicing the last column
y=file.iloc[:,1].values  #printing second column

print(X)
print(y)


# Now that we have our output and input variables, the next step is to split this data into training and test sets, we use linear model of scikit to train our dataset in the following way.

# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.25)  
# here the train set is 75% and test set is 25%


# #### **Training the Algorithm**
# We have split our data into training and testing sets, and now is finally the time to train our algorithm. 

# In[32]:


from sklearn.linear_model import LinearRegression
reg= LinearRegression()
#fitting the regressor to train the data
reg.fit(X_train, y_train)


# Regression models describe the relationship between variables by fitting a line to the observed data. Linear regression models use a straight line

# In[83]:


#plotting the regression line.
#Here the equation of predicted line is in the form y=mX+c'
line=reg.coef_*X + reg.intercept_
#plotting for the test data
plt.scatter(X,y)
plt.plot(X,line, color= 'green')
plt.show()


# ### Making Predictions
# Now that we have trained our algorithm, it's time to make some predictions.

# In[46]:


print(X_test)  #testing input data (hours)


# In[81]:


y_pred = reg.predict(X_test)  #predicting the scores based on the X_test data


# ##### Comparing actual scores and predicted scores

# In[82]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
df


# ##### What will be predicted score if a student studies for 9.25 hrs/ day?

# In[62]:


hours = 9.25
own_pred = reg.predict([[hours]])
print("No of Hours studied = {}".format(hours))
print("Predicted Score = {}".format(*own_pred))


# ##### Performance of the algorithm

# In[75]:


from sklearn import metrics
print('Mean absolute error:', metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:




