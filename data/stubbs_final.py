#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Diondra Stubbs
# CSC 201 - SPRING 2021
# FINAL PROJECT: DATA ANALYSIS PROGRAM
# 26 APRIL 2021
#
# PROGRAM TITLE: 2020 AFRICA HAPPINESS FINAL DATA ANALYSIS PROGRAM
#
#
# PROGRAM DESCRIPTION: A modularized Python program that provides the user with several options to choose 
#                      from to showcase all of the data performed on the Google Forms survey and 2020 World 
#                      Happiness Report data set. This program produces relevant data computations, pivot 
#                      tables and visualizations that are relevant to the 2020 Africa Happiness topic and the 
#                      posed data-driven questions on this topic. The user is able to choose between EIGHT 
#                      options viewing the following: an overview of the topic and data set, data-driven 
#                      questions and predictions, basic data statistics, simple data visualizations, survey 
#                      analysis, data set analysis and findings and observations to the data-driven questions. 
#                      The EIGHTH option allows the user to quit the program. 
#
#
# GENERAL SOLUTION: This program breaks down the tasks required for this program into several functions including:
#
#                   findings: outputs the findings and observations to the data-driven questions using data 
#                             statistics, pivot tables and summary statistics, and visualizations as supporting 
#                             evidence
#
#                  freedom_scatter_chart: creates a scatter chart with a trendline for the freedom to make life 
#                                         choices of African countries by happiness scores
#
#                  process_df: calls the necessary functions to read and clean the data set found online for this 
#                              data set analysis
#
#                  read_as_dataframe: converts data in a CSV file to a Pandas DataFrame
#
#                  clean_dataframe: resolve any data errors that you found in your inspection
#
#                  get_continent_subset: creates filtered subset(s) on the cleaned DataFrame to help narrow it down 
#                                        and perform computations on particular information that will serve as 
#                                        supporting evidence to some of  the questions posed about this topic
# 
#                  africa_pivot: creates a pivot table containing happiness scores for each country in just Africa
#
#                  africa_pivot_line_chart: creates a line chart with a trendline for the happiness scores vs healthy 
#                                           life expectancy in Africa
#
#                  african_countries_pivot: creates a pivot table containing happiness scores for each country in 
#                                           Africa
#
#                  african_countries_pivot_bar_chart: creates a bar chart for the happiness scores by country for 
#                                                     Africa
#
#                  africa_social_pivot: creates a pivot table containing social support and happiness scores for 
#                                       each country in just Africa
#
#                  africa_social_scatter_chart: creates a scatter chart with a trendline for the social support 
#                                               of African countries by happiness score
#
#                  african_social_pivot2: creates a pivot table containing social support scores for each country 
#                                         in Africa
#
#                  african_social_bar_chart: creates a bar chart for the social support scores by country for Africa
#
#                  data_analysis: provides the user with another set of options outputting pivot tables and 
#                                 visualizations created from the cleaned DataFrame
#
#                  read_csv: opens/reads the survey responses CSV file and checks/cleans invalid data entries in 
#                            each row of the CSV file
#
#                  survey_analysis: provides the user with another set of options outputting histogram(s), data 
#                                   computations and pie chart created from the Google Forms survey responses
#
#                  check_age: checks if the age contains only digits OR that the first two characters contain only 
#                             digits (i.e., 21yrs becomes 21) AND if the age is between 0 and 150
#
#                  check_linear_scale: checks if rating contains only digits and if the rating is between the 
#                                      specified scale (i.e., 1 to 5, 1 to 10, etc.)
#
#                  check_multiple_choice: checks if the choice for the multiple choice question is not empty and 
#                                         is one of the provided choices for the question
#
#                  plt_linear_rating:  plots a histogram with the ratings for your linear scale question
#
#                  plt_counts: plots a pie chart of the counts for each choice in the multiple choice question
#
#                  compute: computes a data statistic using the numpy package (i.e, mean, median, standard deviation) 
#                           with the ratings for the linear scale question
#
#                  simple_visualizations: provides the user with another set of options outputting simple 
#                                         visualizations
#
#                  scatter_visual: creates and outputs a scatter chart based on the lists of x and y coordinates 
#                                  it receives as arguments
#
#                  line_visual: creates and outputs a line chart based on the lists of x and y coordinates it 
#                               receives as arguments
#
#                  bar_visual: creates and outputs a bar chart based on the lists of x-coordinates of each bar’s 
#                              left edge and heights of each bar along the y-axis it receives as arguments
#
#                  pie_visual: creates and outputs a pie chart based on the lists of values and slice labels it 
#                              receives as an argument
#
#                  minimum_val: finds the minimum value in a list that it receives as an argument
#
#                  maximum_val: finds the maximum value in a list
#
#                  average_val: finds the average value of a list that it receives as an argument
#
#                  basic_stats: provides the user with another set of options computing basic data statistics on 
#                               the topic
#
#                  option1: outputs happiness scores of Africa from 2015 and 2020 and if they have increased or 
#                           decreased
#
#                  option2: outputs the happiest and saddest country in Africa for 2020
#
#                  option3: outputs the happiness score of Mauritius and Finland in 2020 and compute/output the 
#                           difference
#
#                  option4: outputs the top 5 happiest and saddest countries in Africa for 2020 
#
#                  dq: outputs your data-driven questions and predictions about your topic. It is called in the main 
#                      function when the user chooses option 2 from the main menu
#
#                  overview: outputs an overview of your topic and data set. It is called in the main function when 
#                            the user chooses option 1 from the main menu.
#
#                  check_choice: asks the user for their choice and then checks that the user's input is a valid 
#                                digit between the minimum and maximum arguments passed to it
#
#                  get_choice: outputs the list of eight choices for the main menu and calls the check_choice 
#                              function passing it 1 as the minimum and 8 as the maximum to obtain a valid input 
#                              from the user
#
#                  welcome_msg: outputs the name of your topic, your data-driven questions and a welcome message 
#                               to your data analysis program.
#
#                  main: sets up the program and manages calls to the other functions that handle the eight options. 
#
#
# DATA COLLECTION: Kaggle Data Repository
#                         https://www.kaggle.com/mathurinache/world-happiness-report?select=2020.csv
#                  Google Forms Survey
#                         https://docs.google.com/forms/d/e/1FAIpQLSdu7pYt7SNpZvzXds1kjhzsSuS-X_FyPvHdx2ViR7LyaTISUQ/viewform?usp=sf_link
#                  2020 World Happiness Report 
#                         https://worldhappiness.report/ed/2020/
###########################################################################################################
# START OF IMPORTS AND SETTINGS
###########################################################################################################

# import libraries needed for this program
import csv, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# adjust display settings for DataFrame to show more rows and columns from it
pd.set_option('display.max_rows', 45)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
###########################################################################################################
# END OF IMPORTS AND SETTINGS
###########################################################################################################

###########################################################################################################
# FINDINGS AND OBSERVATIONS
########################################################################################################### 
# function name: findings
# arguments: 1 cleaned DataFrame, at least 1 list containing valid/cleaned responses for your linear rating scale question AND at least 2 integer variables containing counts for your provided choices in your multiple choice question
# return: None
# description: outputs the findings and observations to the data-driven questions using data statistics, pivot tables and summary statistics, and visualizations as supporting evidence
def findings(cleaned_df, some_list, mchoice1, mchoice2, mchoice3, mchoice4):
    
    print('\n####################################################################################################################')
    print('\t\t\t\t\tANSWERS TO DATA-DRIVEN QUESTIONS\n')
    
    # Output the findings and observations for Question 1
    print('Question #1: Which country in Africa ranked the happiest in 2020?\n')
    print('Prediction: Happiness Scores are national average responses to questions of life evaluation. Getting this data is important because it helps remind policy makers and people in power that happiness is based on social capital, not just finanacial. These data results are an essential and useful way to guide public policies and measure their effectiveness. From initial interaction with the original data set, Israel was names as a country that is in Africa. After deeper research, it was found that Israel is actually apart of Asia. The initial prediction was that Israel ranked the happiest in 2020 but the country in Africa that actually ranked the happiest in 2020 in Mauritius.\n')
    print('Observation(s): Mauritius scored as the happiest country in Africa for 2020.')
    print("")
    print('According to the 2020 World Happiness Report, Mauritius is listed as the happiest country in Africa with a happiness score of 6.1. A survey was created on this topic to determine which Africa country was the happiest based on respondents observations or opinions. The survey asked: Which African country do you think had the highest happiness rank in 2020?. The four options given were Israel, Mauritius, South Sudan and Ethiopia. The option of Israel was a mistake on my part since I did not realize that Israel is actually in Asia. Taking this into count, I visually inspected and cleaned the data set in excel by researching where each country is located by continent. Based on these findings, I cleaned the data set in excel by properly stating the continent of each country in the data set.')
    print('\nBased on the survey responses, respondents believed that Israel was the happiest country but since it is not in Africa, most respondents believed that Ethiopia was the happiesnt African country in 2020.\n')
    
    # output summary table and bar chart related to question 1
    africa_subset = get_continent_subset(cleaned_df, "Africa")
    africa_pivot_table = african_countries_pivot(africa_subset)
    print(africa_pivot_table)
    african_countries_pivot_bar_chart(africa_pivot_table)
    print("")
    
    print("Above is outputted the summary table of African countries and their happiness score. The summary table was used to create a bar visual of each African country and their happiness score.")
    print("From the visualization above and the summary table, it is shown that Mauritius has the highest happiness score for 2020. In response to this first data-driven question, the observations above show that Mauritius ranked the happiest in 2020.")
    
    
    # Output the findings and observations for Question 2
    print('\n\nQuestion #2: Does a lower or higher social support score affect the overall happiness of a country?\n')
    print('Prediction: By visually inspecting the orginal data set, the conclusion made was that higher scoring countries (in happiness) had higher social support scores. This question was formed based off that observation. The prediction made was that higher social support scores result in higher happiness scores.\n')
    print('Observation(s): Observing the original CSV data set, it was concluded that Mauritius has the highest social support score. This led to the second data-driven question. Simply viewing the data in excel was not enough to draw a conclusion or response to this question, so a scatter chart was made to observe the relationship between social support and happiness scores.')
    print("")
    print("The chart made in Excel sheets created a scatter graphic and returned an R-squared value of 0.143. R-squared is a goodness-of-fit measure for linear regression models. I decided to use some of my skills from statistics and perform a bit of correlation analysis. Since R-squared is 0.143, approximately 14% of the variability in happiness scores is being explained by social support scores. From this we can draw that the model doesn’t  really do a good job of explaining the variation of happiness scores in social support scores.\n")
    
    # output scatter chart related to question 2
    social_pivot = africa_social_pivot(africa_subset)
    africa_social_scatter_chart(social_pivot["Social Support"], social_pivot["Happiness Scores"])
    print("")
    
    print("The scatter visual showing the relationship between social support and happiness scores is outputted above. We can’t be sure if social support scores give a country a higher or lower happiness score but together, all of the variables contribute to the happiness score.")
    
    
    # Output the findings and observations for Question 3
    print('\n\nQuestion #3: Does freedom to make life choices lower or raise the happiness score of a country?\n')
    print('Prediction: A comparison was made between the highest ranking and lowest ranking country in Africa to make a prediction for this question. It was observed that South Sudan ranked the lowest and Mauritius ranked the highest in happiness. It was drawn that South Sudan had a lower score in freedom to make life choices while Mauritius scored higher. From this, the predicition that was made is that high happiness scores are a result of higher freedom to make life choices scores. The prediciton is that more freedom to make life choices makes a country happy.\n')
    print('Observation(s): While it is easy to draw that Mauritius has a high freedom to make life choices value and South Sudan has the lowest, it does not confirm whether this variable raises or lowers happiness.')
    print("")
    print("Another scatter visual was created in Excel sheets and it returns a R-squared value of 0.033. This implies that, approximately 3% of the variability in happiness scores is being explained by freedom to make life choices. This is very close to 0%, which represents a model that does not explain any of the variation in the response variable around its mean. From this we can conclude that freedom to make life choices is not a good variable to explain happiness scores. However, it is possible that social support is a better predictor of happiness scores.\n")
    
    # output the scatter chart related to question 3
    freedom_scatter_chart(africa_subset)
    print("")
    
    print("The scatter visual showing the relationship between freedom to make life choices and happiness scores is outputted above.")
    
    
    # Output the findings and observations for Question 4
    print('\n\nQuestion #4: What is the relationship between happiness scores and healthy life expectancy?\n')
    print("Prediciton: This question was formed based on the curiosity of if healthy life expectancy results from a country's happiness or if a country's happiness is evaluated in the fact that their is a healthy life expectancy. The healthy life expectancy column is based on Healthy life expectancies at birth, based on the data extracted from the World Health Organization’s (WHO) Global Health Observatory data repository. By comparing the healthy life expectancy scores between Mauritius and South Sudan, it was predicted that there is a positive relationship between happiness scores and healthy life expectancies. Higher happiness scores appear where there is higher healthy life expectancy.\n")
    print("Observation(s): This data-driven question was asked to determine whether happiness causes healthy life expectancy or vice versa. To respond to this question, a line visual was made to observe the relationship between happiness scores and healthy life expectancy.")
    
    # output the line chart related to question 4
    africa_piv = africa_pivot(africa_subset)
    africa_pivot_line_chart(africa_piv["Healthy Life Expectancy"], africa_piv["Happiness Scores"])
    print("")
    
    print("The line chart showing the relationship between happiness scores and healthy life expectancy is outputted above. From this, we can conclude that the visual is very confusing. I would consider this to be an ineffective data visualization. This outputted line visual is not able to provide us with an answer to this data-driven question.\n")
    print("I took extra time to create a scatter visualization in excel to observe this relationship deeper. The returned R-squared value was 0.091. This means that only 9% of the variability in happiness scores is explained by healthy life expectancy. This leads me to conclude that it is better to consider all of the variables (columns) instead of trying to draw which best predicts a country’s happiness score.\n")
    print("Deeper data and statistical analysis on the data set to get more accurate results:")
    print("")
    print("To further the data analysis on this topic, some extra steps were taken to perform statistical analysis. Both Python and R were used to draw final conclusions. First, the original uncleaned data set was used in R programming language to run a multiple linear regression analysis. The results are as follows:")
    print("")
    print("Together, social support, healthy life expectancy, perceptions of corruption, freedom to make life choices, and generosity explain 15% of the variability in happiness scores for African countries. These results are drawn from running a multiple linear regression on a model that only includes happiness scores, social support, healthy life expectancy, perceptions of corruption, freedom to make life choices, and generosity. The outputted adjusted R-squared value was 0.1512, therefore, giving us a 15% variability.")
    print("")
    print("I decided to take it a step further and create more models between happiness scores and just one explanatory variable (social support, healthy life expectancy, perceptions of corruption, freedom to make life choices, or generosity). From this, it was found that social support has the highest adjusted R-squared value of 0.1223. This means that 12% of the variation in Happiness scores is being explained by social support. We can assume that social support is the best predictor of the results for Africa’s happiness scores.")
    print("")
    print("To analyze further, a correlation matrix was created in Python between happiness scores, social support, healthy life expectancy, perceptions of corruption, freedom to make life choices, and generosity. Even though there weren’t any high correlations, the strongest correlation was between social support and happiness scores with a value of 0.38. This means that social support is more conditioning to the happiness score more than any other variable in the data set. Furthermore, happiness in Africa is related to people having relatives and/or friends that they can count on when they need help.")
    print('####################################################################################################################')

# function name: freedom_scatter_chart
# argument(s): 1 filtered subset of a DataFrame containg data only on Africa
# return: None
# description: creates a scatter chart with a trendline for the freedom to make life choices
#               of African countries by happiness scores
def freedom_scatter_chart(africa):
    
    # create a trendline using built-in scipy function linregress using passed x and y values
    slope, intercept, r_value, p_value, std_err = linregress(africa["Freedom to Make Life Choices"], africa["Happiness Scores"])
    
    # create a line chart using passed x and y values from a pivot table
    plt.scatter(africa["Freedom to Make Life Choices"], africa["Happiness Scores"])
   
    # plot the trendline on this scatter chart with label 
    x = africa["Freedom to Make Life Choices"]
    plt.plot(x, intercept + slope*x, 'r', label = 'fitted line')
    
    # give chart appropiate title, x and y labels, legend
    plt.title("Africa: Freedom to Make Life Choices vs. Happiness Scores")
    plt.xlabel("Freedom to Make Life Chocies")
    plt.ylabel("Happiness Scores")
    
    #show the chart
    plt.show()

###########################################################################################################
# DATA SET ANALYSIS
########################################################################################################### 
# function name: read_as_dataframe
# argument(s): 1 string representing the name of the CSV file
# return: 1 Pandas DataFrame
# description: converts data in a CSV file to a Pandas DataFrame
def read_as_dataframe(file):
    
    # read CSV file into pandas dataframe
    df = pd.read_csv(file)
    
    # return pandas dataframe containing data from csv file
    return df 


# function name: clean_dataframe
# argument(s): 1 Pandas DataFrame that has NOT been cleaned (i.e., the original DataFrame containing data errors)
# return: 1 Pandas DataFrame that has been cleaned (i.e., the cleaned DataFrame without data errors)
# description: resolve any data errors that you found in your inspection
def clean_dataframe(df):
    
    # drop any NaN (missing) values from the DataFrame
    df.dropna(inplace = True)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    
    # reset the index numbers of the DataFrame after these changes
    #       have been applied
    df.reset_index(drop = True, inplace = True)
    
    # return the cleaned df
    return df


# function name: process_df
# arguments: 1 string representing the file name of the data set you found online
# return: 1 cleaned DataFrame
# description: calls the necessary functions to read and clean the data set found online for this data set analysis
def process_df(file):
    
    print("One moment the program cleans stubbs_africahappiness.csv of data errors...")
    print("")
    print("Uncleaned DataFrame has 155 rows...")
    print("Errors contained in this data set include:")
    print("1. Remove any missing values in the DataFrame especially in Country Name, Regional Indicator, Social Support and Healthy Life Expectancy columns.")
    print("2. Drop any duplicates in the DataFrame.")
    print("3. Reset the index after making all of these changes to the DataFrame")
    print("")
    print("Cleaned DataFrame has 153 rows...")
    print("Resolved data errors include:")
    print("1. Removed any rows with NaN (missing) values in the DataFrame.")
    print("2. Dropped any duplicate rows in the DataFrame")
    print("3. Reset the index of the DataFrame")
    print("")
    print("stubbs_africahappiness.csv has been cleaned and processed")
    
    # read in CSV file as a pandas dataframe
    df = read_as_dataframe(file)
    
    # call cleaning function to clean all columns with data errors described in previous comment above
    #      reassigned df to the cleaned dataframe
    cleaned_df = clean_dataframe(df)
    
    # return the cleaned DataFrame
    return cleaned_df


# function name: get_continent_subset
# arugment(s):1 Pandas DataFrame that has been cleaned (i.e., the cleaned DataFrame without data errors)
# return: 1 subset of the Pandas DataFrame 
# description: creates filtered subset(s) on the cleaned DataFrame to help narrow it down and perform 
#              computations on particular information that will serve as supporting evidence to some of 
#              the questions posed about this topic
def get_continent_subset(df, continent):
    
    country_filter = df['Regional Indicator'] == continent
    
    africa_sub = df[country_filter]
    
    return africa_sub
    
    
# function name: africa_pivot
# argument(s): 1 filtered subset of a DataFrame containing only data on Africa
# return: 1 pivot table containing happiness scores by country for Africa
# description: creates a pivot table containing happiness scores for each country in just Africa
def africa_pivot(africa_sub):
    
    # create a pivot table calcuating the median of the happiness scores
    #        of each country in Africa in the subset containing all data 
    #        on Africa
    africa_pivot = africa_sub.groupby("Happiness Scores", as_index = False).median()
    
    # reset the index
    africa_pivot.reset_index(inplace = True)
    
    # return the pivot table
    return africa_pivot


# function name: africa_pivot_line_chart
# argument(s): 2 columns from a given pivot table as x and y values to plot
# return: None
# description: creates a line chart with a trendline for the happiness scores vs healthy life expectancy in Africa
def africa_pivot_line_chart(x, y):
    
    # create a line chart using passed x and y values from a pivot table
    plt.plot(x, y)
    
    # give chart appropiate title, x and y labels, legend
    plt.title("Africa: Healthy Life Expectancy vs. Happiness Scores")
    plt.xlabel("Healthy Life Expectancy")
    plt.ylabel("Happiness Scores")
    
    #show the chart
    plt.show()
    

# function name: african_countries_pivot
# argument(s): 1 filtered subset of a DataFram containg data only on Africa
# return: 1 pivot table containing happiness scores by country for Africa
# description: creates a pivot table containing happiness scores for each country in Africa
def african_countries_pivot(africa_sub):
    
    # create a pivot table calculating the median of happiness scores of each country
    africa_countries_pivot = africa_sub.pivot_table(values = "Happiness Scores", index = "Country Name", aggfunc = np.min, margins = False)
        
    # reset the index of pivot table
    africa_countries_pivot.reset_index(inplace = True)
    
    # return the pivot table
    return africa_countries_pivot  


# function name: african_countries_pivot_bar_chart
# argument(s): 1 pivot table with happiness scores of countries in Africa
# return: None
# description: creates a bar chart for the happiness scores by country for Africa
def african_countries_pivot_bar_chart(african_countries2_pivot):
    
    # create a bar chart using passed pivot table
    african_countries2_pivot.plot(kind = "bar")
    
    # adjust x ticks to show country names
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], ["Algeria", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", "Central African Republic", "Chad", "Comoros", "Congo (Brazzaville)", "Congo (Kinshasa)", "Egypt", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Senegal", "Sierra Leone", "South Africa", "South Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"])
    
    # give chart appropiate title, x and y labels, legend
    plt.title("African Countries Happiness Scores")
    plt.xlabel("Countries")
    plt.ylabel("Happiness Scores")
    plt.legend(bbox_to_anchor = (1.5, 1), loc = "upper right")
    
    # show the chart
    plt.show()
    

# function name: africa_social_pivot 
# argument(s): 1 filtered subset of a DataFrame containing only data on Africa
# return: 1 pivot table containing happiness scores by country for Africa
# description: creates a pivot table containing social support and happiness scores for each country in just Africa
def africa_social_pivot(africa_sub):
    
    # create a pivot table calcuating the median of the happiness scores
    #        of each country in Africa in the subset containing all data 
    #        on Africa
    africa_social_piv = africa_sub.pivot_table(values = "Social Support", index = "Happiness Scores", aggfunc = np.min, margins = False)
    
    # reset the index
    africa_social_piv.reset_index(inplace = True)
    
    # return the pivot table
    return africa_social_piv


# function name: africa_social_scatter_chart
# argument(s): 2 columns from a given pivot table as x and y values to plot
# return: None
# description: creates a scatter chart with a trendline for the social support of African countries by happiness score
def africa_social_scatter_chart(x, y):
    
    # create a trendline using built-in scipy function linregress using passed x and y values
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    
    # create a line chart using passed x and y values from a pivot table
    plt.scatter(x, y)
   
    # plot the trendline on this scatter chart with label 
    plt.plot(x, intercept + slope*x, 'r', label = 'fitted line')
    
    # give chart appropiate title, x and y labels, legend
    plt.title("Africa: Social Support vs. Happiness Scores")
    plt.xlabel("Social Support")
    plt.ylabel("Happiness Scores")
    
    #show the chart
    plt.show()


# function name: african_social_pivot2
# argument(s): 1 filtered subset of a DataFrame containg data only on Africa
# return: 1 pivot table containing happiness scores by country for Africa
# description: creates a pivot table containing social support scores for each country in Africa
def african_social_pivot2(africa_sub):
    
    # create a pivot table calculating the median of happiness scores of each country
    africa_social_pivot2 = africa_sub.pivot_table(values = "Social Support", index = "Country Name", aggfunc = np.min, margins = False)
        
    # reset the index of pivot table
    africa_social_pivot2.reset_index(inplace = True)
    
    # return the pivot table
    return africa_social_pivot2  


# function name: african_countries_pivot_bar_chart
# argument(s): 1 pivot table with happiness scores of countries in Africa
# return: None
# description: creates a bar chart for the social support scores by country for Africa
def african_social_bar_chart(african_social_pivot):
    
    # create a bar chart using passed pivot table
    african_social_pivot.plot(kind = "bar")
    
    # adjust x ticks to show country names
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], ["Algeria", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", "Central African Republic", "Chad", "Comoros", "Congo (Brazzaville)", "Congo (Kinshasa)", "Egypt", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Senegal", "Sierra Leone", "South Africa", "South Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"])
    
    # give chart appropiate title, x and y labels, legend
    plt.title("African Countries Social Support")
    plt.xlabel("Countries")
    plt.ylabel("Social Support")
    plt.legend(bbox_to_anchor = (1.5, 1), loc = "upper right")
    
    # show the chart
    plt.show()

    
# function name: data_analysis
# arguments: 1 cleaned DataFrame
# return: none
# description: provides the user with another set of options outputting pivot tables and visualizations created from your cleaned DataFrame
def data_analysis(cleaned):
    
    # output the titles for this part of the program and where the data was collected
    print('\n####################################################################################################################')
    print('\t\t\t\t\tAFRICA HAPPINESS DATA SET ANALYSIS')
    print('\t\t\tData collected from Data collected from Kaggle Data Repository\n')
    
    africa_subset = get_continent_subset(cleaned, "Africa")
    africa_pivot_table = african_countries_pivot(africa_subset)
    social_pivot = africa_social_pivot(africa_subset)
    social_pivot2 = african_social_pivot2(africa_subset)
    
    # define string variable that is empty to hold user's input
    choice = ''
    
    
    # loop until the user chooses the choice to quit the program
    while choice != 9:
        
        #output the list of options the user can choose from
        print("Choose one of the following numbered options to view data and statistics about happiness in Africa.")
        print("1. Pivot Table: Africa 2020 Happiness Scores Pivot Table")
        print("2. Pivot Table: Countries in Africa Happiness Scores")
        print("3. Pivot Table: Africa 2020 Social Support Pivot Table")
        print("4. Pivot Table: Countries in Africa Social Support Scores")
        print("5. Line Chart: Africa Healthy Life Expectancy vs. Happiness Scores")
        print("6. Bar Chart: African Countries Happiness Scores")
        print("7. Scatter Chart: Africa Social Support vs. Happiness Scores")
        print("8. Bar Chart: African Countries Social Support")
        print("9. Back to main menu\n")
    
        choice = check_choice(1,9)
        
        # check if the user chose the first option
        if choice == 1:
            print("\nAFRICA: 2020 HAPPINESS SCORES PIVOT TABLE\n")
            print(africa_subset)
            
        # check if the user chose the second option
        elif choice == 2:
            print("COUNTRIES IN AFRICA: 2020 HAPPINESS SCORES\n")
            print(africa_pivot_table)
            
        # check if the user chose the third option
        elif choice == 3:
            print("AFRICA: 2020 SOCIAL SUPPORT PIVOT TABLE\n")
            print(social_pivot)
            
            
        # check if the user chose the fourth option
        elif choice == 4:
            print("COUNTRIES IN AFRICA: 2020 SOCIAL SUPPORT SCORES\n")
            print(social_pivot2)
            
            
        # check if the user chose the fifth option
        elif choice == 5:
            
            # call the function africa_pivot_line_chart
            africa_pivot_line_chart(africa_subset["Healthy Life Expectancy"], africa_subset["Happiness Scores"])
            
        # check if the user chose the sixth option
        elif choice == 6:
            
            # call the function african_countries_pivot_bar_chart
            african_countries_pivot_bar_chart(africa_pivot_table)
            
        # check if the user chose the seventh option
        elif choice == 7:
            
            # call the function africa_social_scatter_chart
            africa_social_scatter_chart(social_pivot["Social Support"], social_pivot["Happiness Scores"])   
            
        # check if the user chose the fourth option
        elif choice == 8:
            
            # call the function african_social_bar_chart
            african_social_bar_chart(social_pivot2)
        
        # check if the user choice the ninth option    
        elif choice == 9:
            
            print("Sending you back to the main menu...")
            print('')
            
###########################################################################################################
# SURVEY ANALYSIS
###########################################################################################################  
# function name: read_csv
# arguments: 1 string representing the file name of the survey responses CSV file
#return: at least 1 list containing valid/cleaned responses for your linear rating scale question AND at least 2 integer variables containing counts for your provided choices in your multiple choice question
# description: opens/reads the survey responses CSV file and checks/cleans invalid data entries in each row of the CSV file
def read_csv(filename):
    
    # create empty list to store linear scale question in CSV file
    ratings = []
    
    # create variable(s) to store count of multiple choice otpions and set equal to 0
    mul_choice1 = 0
    mul_choice2 = 0
    mul_choice3 = 0
    mul_choice4 = 0

    # open the survey responses CSV file to read it
    with open(filename) as csv_infile:

        # reader is the Python object to read the CSV file
        reader = csv.DictReader(csv_infile)
        
        # output that the survey responses CSV file is read and cleaned
        print("Reading and cleaning survey.csv.....\n")

        # this for loop takes each row of the CSV file and calls it "row"
        for row in reader:
            
            # check if the age in the row is valid by:
            #       CALLING check_age function
            #       AND
            #       PASSING the age in the row
            if check_age(row["How old are you?"]):
                
                # check if the response for the linear scale question in the row is valid by:
                #       CALLING check_linear_scale function
                #       AND
                #       PASSING the response for the first linear scale question in the row
                if check_linear_scale(row["The more freedom there is to make life choices the more people are happy (with 1 being strongly disagree, 2 being disagree, 3 neutral, 4 agree, 5 strongly agree)."]):
                    
                    # append valid integer digit between 1 and 5 to list storing values 
                    #        for first linear scale question
                    ratings.append(int(row["The more freedom there is to make life choices the more people are happy (with 1 being strongly disagree, 2 being disagree, 3 neutral, 4 agree, 5 strongly agree)."]))
                    
                # otherwise (i.e., response for the first linear scale question in the row is invalid)
                else:
                    
                    # output error message with line number and invalid data entry and skip bad data value
                    print("Error on line", reader.line_num, row["The more freedom there is to make life choices the more people are happy (with 1 being strongly disagree, 2 being disagree, 3 neutral, 4 agree, 5 strongly agree)."], "is not a valid response. Bad data value skipped")
            
             # check if the choice for the multiple choice question in the row is valid by:
                #       CALLING check_multiple_choice function
                #       AND
                #       PASSING the choice for the multiple choice question in the row
                if check_multiple_choice(row["Which African country do you think had the highest happiness rank in 2020?"]) == True:
                    
                    # count the number of responses for each of the valid choices in this multiple choice question
                    #       and add to the 4 variables storing the counts for each of these choices
                    mul_choice1 += row["Which African country do you think had the highest happiness rank in 2020?"].count("Israel")
                    mul_choice2 += row["Which African country do you think had the highest happiness rank in 2020?"].count("Mauritius")
                    mul_choice3 += row["Which African country do you think had the highest happiness rank in 2020?"].count("South Sudan")
                    mul_choice4 += row["Which African country do you think had the highest happiness rank in 2020?"].count("Ethiopia")
                    
                    
                # otherwise (i.e., choice for the multiple choice question in the row is invalid)
                else:
                    
                    # output error message with line number and invalid data entry and skip bad data value
                    print("Error on line", reader.line_num, ":", row["Which African country do you think had the highest happiness rank in 2020?"], "Empty or invalid choice. Bad data value skipped.")
                    
            # otherwise (i.e., the age in the row is invalid)    
            else:
                
                # output error message with line number and invalid data entry and skip bad data value
                print("Error on line", reader.line_num, ":", row["How old are you?"], "-->", "Not an integer or outside of range between 0 and 150. Bad data value skipped.")
                
    
    # output that the CSV file has been cleaned
    print("\nsurvey.csv has been cleaned!\n\n\n")
    
    # return list containing valid/cleaned responses for linear scale question
    #         AND 4 integers containing counts for 4 choices in the multiple choice question
    return ratings, mul_choice1, mul_choice2, mul_choice3, mul_choice4


# function name: check_age
# argument(s): one string representing the age of the respondent
# description: checks if the age contains only digits AND checks if the age is between 0 and 150
# return: one boolean if the age is valid
#             True if the age is valid
#             False if the age is invalid
def check_age(some_age):
    
    # flag variable set to True assuming age is valid before checks
    valid = True
    
    # check if the age (passed as the argument) contains only digits
    #       OR the first two characters of the age (passed as the argument)
    #       contain only digits
    #
    # examples that PASS this check: 26years old, 200 yrs
    # examples that FAIL this check: twenty-three, too old, twenty-two
    if some_age.isdigit() or some_age[0:2].isdigit():
        
         # check if the length of age (passed as the argument) is greater than 2
        #       AND the third character of the age (passed as the argument) is
        #       a digit
        #
        # examples that PASS this check: 200 yrs
        # examples that FAIL this check: 26years old, 25
        if len(some_age) > 2 and some_age[2].isdigit():
            
            # get the first three characters of the age (passed as the argument)
            #     using string slicing, convert to an integer and store in a 
            #     NEW variable holding the cleanes value
            #
            # example CLEANED with this statement: 200 yrs becomes 200
            age = int(age[0:3])

        # otherwise (i.e., the length of age is NOT greater than 2 AND the third
        #            character of the age is NOT a digit)
        #
        # examples that PASS this check: 26years old, 25, ... etc.
        else:
            # get the first two characters of the age (passed as the argument)
            #     using string slicing, convert to an integer and store in a
            #     NEW variable holding the cleaned value
            #
            # example CLEANED with this statement: 26years old becomes 26
            age = int(some_age[0:2])
            
            
        # check if the age (NEW variable created in previous check) is outside the 
        #       range of 1 to 150 for human lifespan
        #
        # examples that PASS this check: 200
        # examples that FAIL this check: 26, 25, ... etc. 
        if age < 0 or age > 150:
            
            # update the flag variable to be set to False since the cleaned age
            #     (NEW variable created in previous check) is invalid and outside
            #     of range
            valid = False
    # otherwise (i.e., age (passed as the argument) does NOT contain only digits)
    #
    # examples that PASS this check: twenty-three, too old, twenty-two
    else: 
        # update the flag variable to be set to False since the age
        #     (passed as the argument) is invalid and not an integer
        valid = False
    
    # return the flag variable that is either True or False depending on if the
    #    age of the respondent is valid after all above checks
    return valid


# function name: check_linear_scale
# argument(s): one string representing the rating given for a linear sclae question
# description: checks if rating contains only digits and if the rating is between
#              the scale of 1 to 5
# return: one boolean (i.e., either True or False) if the rating is valid
#             True if the rating is valid
#             False if the rating is invalid
def check_linear_scale(linear_question):
    
    # flag variable set to true assuming ratin is valid before checks
    valid = True
    
     # check if the rating (passed as the argument) only contains digits
     #  convert to an integer and store in a NEW variable holding the integer value 
    if linear_question.isdigit():
        
        linear = int(linear_question)
        
        # check if the rating (NEW variable created in previous line) is between
        #       1 and 5 for this linear rating scale question
        if linear < 1 or linear > 5:
            
            # update the flag variable to be set to False since the rating
            #     (NEW variable created in previous line) is invalid and outside
            #     the range of 1 and 5
            valid = False
            
     # otherwise (i.e, the rating (passed as the argument) does NOT only contain digits)
    else:
        
        # update the flag variable to be set to False since the rating
        #     (passed as the argument) is invalid and not an integer
        valid = False
    
    # return the flag variable that is either True or False depending on if the
    #    rating for the linear scale question is valid after all above checks
    return valid      
    
# function name: check_multiple_choice
# argument(s): one string representing the choice given for a multiple choice question
# description: checks if the choice for the multiple choice question is not empty and
#              is one of the provided choices for the question
# return: one boolean (i.e., either True or False) if the choice is valid
#             True if the choice is valid
#             False if the choice is invalid
def check_multiple_choice(multiple_choice):
    
    # flag variable set to True assuming choice is valid before checks
    valid = True
    
    # check is the choice (passed as the argument) for the multiple choice questions is NOT empty
    if multiple_choice != "" or multiple_choice != None: 
        
        # check if the choice (passed as the argument) for the multiple choice questions is NOT 
        #       equal to any of the provided choices for the multiple choice question
        if multiple_choice != "Israel" and multiple_choice != "Mauritius" and multiple_choice != "South Sudan" and multiple_choice != "Ethiopia": # ???
            
            # update the flag variable to be set to False since the choice 
            #        (passed as the argument) is invalid and not one of the provided
            #        choices for the multiple choice question
            valid = False
            
     # otherwise (i.e, the choice (passed as the argument) is empty)
    else:
        
        # update the flag variable to be set to False since the choice
        #        (passed as the argument) is invalid and cannot be empty
        valid = False
        
    # return the flag variable that is either True or False depending on if the
    #    choice for the multiple choice question is valid after all above checks
    return valid 
# function name: plt_linear_rating
# arguments(s): one list containing integer ratings to the linear scale questions
# description: plots a histogram of the ratings for the linear scale question
# return: none
def plt_linear_rating(some_list):
    
    # plot a histogram using
    plt.hist(some_list, bins = 5)
    
    # give appropiate title and y label
    plt.title("Freedom to make life choices makes people happier")
    plt.ylabel("Rating (1 = Strong Disagree to 5 = Strongly Agree)")
    
    # give appropiate x and y ticks
    plt.xticks([1,2,3,4,5], ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
    
    
    # display the histogram graph
    plt.show()

    
# function name: plt_counts
# argument(s): 4 integer variables storing the counts for each choice in the 
#              multiple choice question
# description: plots a pie chart of the counts for each choice in the multiple choice question
# return: none     
def plt_counts(ct_opt1, ct_opt2, ct_opt3, ct_opt4):
    
    # create a list storing the values of all 4 variables with the counts for each choice 
    #        in the multiple choice question
    counts = [ct_opt1, ct_opt2, ct_opt3, ct_opt4]    
    
    # create a list of the string labels containing all the listed choice from the multiple
    #        choice question that is being plotted in this pie chart
    slice_labels = ['Israel', 'Mauritius', 'South Sudan', 'Ethiopia']
    
    # create a pie chart using the two lists created in the previous lines for
    #        the counts and labels
    plt.pie(counts, labels = slice_labels, autopct = "%.2f")
    
    # give appropriate title
    plt.title("Which African country had the highest happiness rank in 2020?")
    
    # plot the legend in the upper right corner
    plt.legend(bbox_to_anchor=(2,0.8), loc="upper right")
    
    # display the pie chart
    plt.show()
    
    
# function name: compute
# argument(s): 1 list containing ratings for linear scale question
# description: computes a data statistic using the numpy module (i.e, mean, median, standard deviation) with the ratings for linear scale question
# return: 3 float containing the computed data statistic on the list of ratings passed to it 
def compute(l_ratings):
    
    # create a variable storing the mean/average of ratings from the linear scale question
    average_ratings = np.mean(l_ratings)
    
    # create a variable storing the median of ratings from the linear scale question
    median_ratings = np.median(l_ratings)
    
    # create a variable storing the standard deviation of ratings from the linear scale question
    std_ratings = np.std(l_ratings)
    
    # return 3 float variables containg the computed data statistcs (mean, median, standard deviation)
    return average_ratings, median_ratings, std_ratings


# function name: survey_analysis
# arguments: at least 1 list containing valid/cleaned responses for your linear rating scale question AND at least 2 integer variables containing counts for your provided choices in your multiple choice question
# return: none
# description: provides the user with another set of options outputting histogram(s), data computations and pie chart(s) created from your Google Forms survey responses
# output the titles for this part of the program and where the data was collected
def survey_analysis(some_list, ct_opt1, ct_opt2, ct_opt3, ct_opt4):
    
    
    # output the titles for this part of the program and where the data was collected
    print('\n####################################################################################################################')
    print('\t\t\t\t\tAFRICA HAPPINES SURVEY ANALYSIS')
    print('\t\t\t\t    Data collected from Google Forms Survey\n')
    
    
    # define string variable that is empty to hold user's input
    choice = ''
    
    
    # loop until the user chooses the choice to quit the program
    while choice != 4:
        
        #output the list of options the user can choose from
        print("Choose one of the following numbered options to view statistics and visualizations about the happiness in Africa survey.")
        print("1. Histogram: Freedom to make life choices makes people happier")
        print("2. Computed Statistics: Mean, Median and Standard deviation on response from linear scale survey question")
        print("3. Pie Chart: Breakdown of Perspectives on the happiest country in Africa")
        print("4. Back to main menu\n")
    
        choice = check_choice(1,4)
        
        # check if the user chose the first option
        if choice == 1:
            
            # call the function option1
            plt_linear_rating(some_list)
            
        # check if the user chose the second option
        elif choice == 2:
            
            # call the function option2
            average, median, std = compute(some_list)
            print("The computed average rating from the linear scale question is", format(average, '.02f'))
            print("The computed median rating from the linear scale question is", median)
            print("The computed standard deviation of ratings from the linear scale question is", format(std, '.02f'),"\n\n\n")
            
        # check if the user chose the third option
        elif choice == 3:
            
            # call the function option3
            plt_counts(ct_opt1, ct_opt2, ct_opt3, ct_opt4)
            
        # check if the user chose the fourth option
        elif choice == 4:
            
            print("Sending you back to the main menu...")
            print('')
            
            
###########################################################################################################
# SIMPLE VISUALIZATIONS
###########################################################################################################
# function name: scatter_visual
# arguments: 2 integer lists where one contains the x-coordinates and the other contains the y-coordinates
# return: none
# description: creates and outputs a scatter chart based on the lists of x and y coordinates it received as arguments

def scatter_visual(x_coords, y_coords):
    
    # plot the given lists of x and y coordinates as a scatter chart
    x_coords = [0, 1, 2, 3, 4, 5]
    y_coords = [0.98, 0.76, 1.21, 1.39, 1.40, 0.91]
    
    plt.scatter(x_coords, y_coords)
    
    # give appropiate title, x-axis and y-axis labels, and grid layout
    plt.title("Mauritius Social Support Overtime")
    plt.xlabel("Year")
    plt.ylabel("Social Support")
    
    # customize the x tick marks along the x-axis and y-axis
    plt.xticks([0, 1, 2, 3, 4, 5], ['2015', '2016', '2017', '2018', '2019', '2020'])
    
    # show the scatter chart
    plt.show()
    

# function name: line_visual
# arguments: 2 integer lists where one contains the x-coordinates and the other contains the y-coordinates
# return: none
# description:creates and outputs a line chart based in the lists of x and y coordinates it received as arguments

def line_visual(x_coords, y_coords):
    
    # plot the given lists of x and y coordinates as a line chart using marker
    x_coords = [0, 1, 2, 3, 4, 5]
    y_coords = [4.30, 4.27, 4.24, 4.33, 4.31, 4.43]
    
    plt.plot(x_coords, y_coords)
    
    # give appropiate title, x-axis and y-axis labels, and grid layout
    plt.title("Africa Happiness Scores from 2015 to 2020")
    plt.xlabel("Year")
    plt.ylabel("Happiness Scores")
    
    # customize the x tick marks along the x-axis and y-axis
    plt.xticks([0, 1, 2, 3, 4, 5], ['2015', '2016', '2017', '2018', '2019', '2020'])
    
    # show the line chart 
    plt.show()
    

# function name: bar_visual
# arguments: 2 integer lists where one contains the x-coordinates of each bar's left edge and heights of each bar along y-axis it receives as arguments
# return: none
# description: creates and outputs a bar chart based on the lists of x-coordinates of each bar’s left edge and heights of each bar along y-axis it receives as arguments

def bar_visual(left_edges, heights):
    
    # plot the years and happiness scores as a bar chart
    left_edges = [0, 1, 2, 3, 4, 5]
    heights = [5.48, 5.65, 5.63, 5.89, 5.89, 6.10]
    
    plt.bar(left_edges, heights, color={'r','m','g','k','b','c'})
    
    # give appropriate title and x-axis and y-axis labels
    plt.title("Mauritius Happiness Scores from 2015 to 2020")
    plt.xlabel("Year")
    plt.ylabel("Happiness Scores")
    
    # customize the x tick marks along the x-axis and y-axis
    plt.xticks([0, 1, 2, 3, 4, 5], ['2015', '2016', '2017', '2018', '2019', '2020'])
    
    # show the bar chart
    plt.show()
    

# function name: pie_visual
# arguments: 2 lists where one contains integer values for the slice sizes and the other contains 
#             the labels of the slices
# return: none
# description: creates and outputs a pie chart based on the lists of values and slice labels it 
#                receives as an argument

def pie_visual(happiest_country, slice_labels):
    
    # plot the list of percentages and labels as a pie chart
    happiest_country = [12, 12, 24, 52]
    
    slice_labels = ['Mauritius', 'South Sudan', 'Ethiopia', 'Israel']
    
    # plot the legend in the upper right corner to show the labels/colors of each slice in the pie chart
    plt.pie(happiest_country, labels = slice_labels, colors={'k', 'm', 'r', 'c'})
    
    # give appropriate title 
    plt.title('Which African Country is the Happiest?')
    
    # show the pie chart
    plt.show()
    
    
# function name: minimum_val
# arguments - 1 list containing integer values
# return - 1 integer that is the minimum value in the list that it receives as an argument
# description: : finds the minimum value in a list that it receives as an argument

def minimum_val(some_list):
    
    minimum = some_list[0]
    
    for i in some_list:
        
        if i < minimum:
            minimum = i
            
    return minimum


# function name: maximum_val
# arguments - 1 list containing integer values
# return - 1 integer that is the maximum value in the list that it receives as an argument
# description: : finds the maximum value in a list that it receives as an argument

def maximum_val(some_list):
    
    maximum = some_list[0]
    
    for i in some_list:
        
        if i > maximum:
            maximum = i
    
    return maximum


# function name: average_val
# arguments - 1 list containing integer values
# return - 1 integer (or float) that is the average of the list
# description: : finds the average value of a list that it receives as an argument

def average_val(some_list):
    
    total = 0
    
    for value in some_list:
        total += value
        
    average = total / len(some_list)
    
    return average


# function name: simple_visualizations
# arguments: none
# return: none
# description: provides the user with another set of options outputting simple visualizations on your topic
def simple_visualizations():
    
    # output the titles for this part of the program and where the data was collected
    print('\n####################################################################################################################')
    print('\t\t\t\t\tAFRICA HAPPINESS DATA VISUALIZATIONS OF HAPPINESS SCORES')
    print('\t\t\tData collected from Data collected from 2015 and 2020 World Happiness Report\n')
    
    
    # define string variable that is empty to hold user's input
    choice = ''
    
    
    # loop until the user chooses the choice to quit the program
    while choice != 5:
        
        #output the list of options the user can choose from
        print("\nChoose one of the following numbered options to view data and statistics about happiness in Africa.")
        print("1. Scatter Chart: Social Support of Mauritius Over Time")
        print("2. Line Chart: African Happiness Scores from 2015 to 2020")
        print("3. Bar Chart: Mauritius Happiness Score between 2015 and 2020")
        print("4. Pie Chart: Breakdown of Perspective on Happiest Country in Africa")
        print("5. Back to main menu")
    
        choice = check_choice(1,5)
        
        # check if the user chose the first option
        if choice == 1:
            
            # call the function scatter_visual
            x_coords = [0, 1, 2, 3, 4, 5]
            y_coords = [0.98, 0.76, 1.21, 1.39, 1.40, 0.91]
            scatter_visual(x_coords, y_coords)
            
            social_support = [0.98, 0.76, 1.21, 1.39, 1.40, 0.91]
    
            # calculate and output the minimum, maximum and average of social support
            min1 = minimum_val(social_support)
            print("Minimum Social Support of Mauritius from 2015 to 2020:", min1)
    
            max1 = maximum_val(social_support)
            print("Maximum Social Support of Mauritius from 2015 to 2020:", max1)
    
            avg1 = average_val(social_support)
            print("Average Social Support Score of Mauritius from 2015 to 2020:", avg1)
            
        # check if the user chose the second option
        elif choice == 2:
            
            # call the function line_visual
            x_coords = [0, 1, 2, 3, 4, 5]
            y_coords = [4.30, 4.27, 4.24, 4.33, 4.31, 4.43]
            line_visual(x_coords, y_coords)
            
            african_happiness = [4.30, 4.27, 4.24, 4.33, 4.31, 4.43]
    
            # calculate and output the minimum, maximum and average of happiness scores of  Africa between 2015 and 2020
            min2 = minimum_val(african_happiness)
            print("Minimum African Happiness Score between 2015 and 2020:", min2)
    
            max2 = maximum_val(african_happiness)
            print("Maximum African Happiness Score between 2015 and 2020:", max2)
    
            avg2 = average_val(african_happiness)
            print("Average African Happiness Score between 2015 and 2020:", avg2)
            
        # check if the user chose the third option
        elif choice == 3:
            
            # call the function bar_visual
            left_edges = [0, 1, 2, 3, 4, 5]
            heights = [5.48, 5.65, 5.63, 5.89, 5.89, 6.10]
            bar_visual(left_edges, heights)
            
            mauritius_happiness = [5.48, 5.65, 5.63, 5.89, 5.89, 6.10]
    
            # calculate and output the minimum, maximum and average of happiness scores 
            min3 = minimum_val(mauritius_happiness)
            print("Minimum Mauritius Happiness Score between 2015 and 2020:", min3)
    
            max3 = maximum_val(mauritius_happiness)
            print("Maximum Mauritius Happiness Score between 2015 and 2020:", max3)
    
            avg3 = average_val(mauritius_happiness)
            print("Average Happiness Score of Mauritius between 2015 and 2020:", avg3)
            
        # check if the user chose the fourth option
        elif choice == 4:
            
            # call the function pie_visual
            # create a list of percentages for the user responses to statements provided in survey question about the happiest country in africa
            happiest_country = [12, 12, 24, 52]
    
            # create labels for respective slices in the pie chart
            slice_labels = ['Mauritius', 'South Sudan', 'Ethiopia', 'Israel']
            pie_visual(happiest_country, slice_labels)
            
            happiest_country = [12, 12, 24, 52]
    
            # calculate and output the minimum, maximum and average of user responses
            min4 = minimum_val(happiest_country)
            print("Minimum Response:", min4)
    
            max4 = maximum_val(happiest_country)
            print("Maximum Response:", max4)
    
            avg4 = average_val(happiest_country)
            print("Average Response:", avg4)
            
        # check if the user choice the fifth option    
        elif choice == 5:
            
            print("Sending you back to the main menu...")
            print('')
            
###########################################################################################################
# BASIC DATA STATISITCS
###########################################################################################################            
# function name: option4
# arguments: none
# return: none
# description: outputs the top 5 happiest and saddest countries in Africa for 2020                      
def option4():
    # output the top 5 happiest & saddest African countries for 2020
    print("The Top 5 Happiest Countries in Africa for 2020:")
    print("Mauritius")
    print("Libya")
    print("Ivory Coast")
    print("Benin")
    print("Congo(Brazzaville)")
    print("")
    print("The Top 5 Saddest Countries in Africa for 2020:")
    print("Tanzania")
    print("Central Africa Republic")
    print("Rwanda")
    print("Zimbabwe")
    print("South Sudan")
    
    print("")
    print("")

# function name: option3
# arguments: none
# return: none
# description: outputs the happiness score of Mauritius and Finland in 2020 and compute/output the difference
def option3():
    
    # define float variables storing the happiness scores of Mauritius and Finland
    mauritius = 6.10
    finland = 7.81

    # compute the difference between the happiness scores of Finland and Mauritius
    difference = (finland - mauritius)

    #output the happiness score of Mauritius and Finland in 2020 and compute/output the difference
    print("2020 Mauritius Happiness Score:", mauritius)
    print("2020 Finland Happiness Score:", finland)
    print("In 2020, Finland ranked happier than Mauritius with a difference of", difference)
    
    print("")
    print("")
    
# function name: option2
# arguments: none
# return: none
# description: outputs the happiest and saddest country in Africa for 2020
def option2():
    
    # define float variables storing the happiness scores of Mauritius and South Sudan from 2020
    mauritius = 6.10
    south_sudan = 2.82

    # output the happiest and saddest country in Africa for 2020
    print("The Happiest African Country in 2020 was Mauritius with a score of", mauritius)
    print("The Saddest African Country in 2020 was South Sudan with a score of", south_sudan)
    
    print("")
    print("")
    
# function name: option1
# arguments: none
# return: none
# description: outputs happiness scores of Africa from 2015 and 2020 and if they have increased or decreased
def option1():
    
    # define float variables storing Africa's happiness scores for 2015 and 2020
    africa_happiness2015 = 4.30
    africa_happiness2020 = 4.43

    # output happiness scores of Africa from 2015 and 2020 and if they have increased or decreased
    print("Overall happiness score of Africa in 2015:", africa_happiness2015)
    print("Overall happiness score of Africa in 2020:", africa_happiness2020) 
    print("The overall happiness score of Africa has increased since 2015.")
    
    print("")
    print("")

###########################################################################################################
# BASIC DATA STATISTICS
###########################################################################################################   

# function name: basic_stats
# arguments: none
# return: none
# description: provides the user with another set of options computing basic data statistics on your topic
def basic_stats():
    
    # output the titles for this part of the program and where the data was collected
    print('\n####################################################################################################################')
    print('\t\t\t\t\tAFRICA HAPPINESS REPORT BASIC DATA STATISTICS')
    print('\t\t\tData collected from Data collected from 2015 and 2020 World Happiness Report\n')
    
    
    # define string variable that is empty to hold user's input
    choice = ''
    
    
    # loop until the user chooses the choice to quit the program
    while choice != 5:
        
        #output the list of options the user can choose from
        print("Choose one of the following numbered options to view data and statistics about happiness in Africa.")
        print("1. Overall Happiness of Africa: 2015 vs 2020")
        print("2. Happiest & Saddest country in Africa for 2020")
        print("3. Happiness score of Mauritius vs Finland for 2020")
        print("4. Top 5 Happiest & Saddest countries in Africa for 2020")
        print("5. Back to main menu\n")
    
        choice = check_choice(1,5)
        
        # check if the user chose the first option
        if choice == 1:
            
            # call the function option1
            option1()
            
        # check if the user chose the second option
        elif choice == 2:
            
            # call the function option2
            option2()
            
        # check if the user chose the third option
        elif choice == 3:
            
            # call the function option3
            option3()
            
        # check if the user chose the fourth option
        elif choice == 4:
            
            # call the function option4
            option4()
            
        # check if the user choice the fifth option    
        elif choice == 5:
            
            print("Sending you back to the main menu...")
            print('')

###########################################################################################################
# DATA DRIVEN QUESTIONS & PREDICTIONS
###########################################################################################################  

# function name: dq
# arguments: none
# return: none
# description: outputs data-driven questions and predictions about your topic and data set
def dq():
    
    print('\n####################################################################################################################')
    print('\t\t\t\t\tDATA-DRIVEN QUESTIONS AND PREDICITIONS\n')
    print('Question #1: Which country in Africa ranked the happiest in 2020?\n')
    print('Prediction: Happiness Scores are national average responses to questions of life evaluation. Getting this data is important because it helps remind policy makers and people in power that happiness is based on social capital, not just finanacial. These data results are an essential and useful way to guide public policies and measure their effectiveness. From initial interaction with the original data set, Israel was names as a country that is in Africa. After deeper research, it was found that Israel is actually apart of Asia. The initial prediction was that Israel ranked the happiest in 2020 but the country in Africa that actually ranked the happiest in 2020 in Mauritius.\n')
    print('')
    print('Question #2: Does a lower or higher social support score affect the overall happiness of a country?\n')
    print('Prediction: By visually inspecting the orginal data set, the conclusion made was that higher scoring countries (in happiness) had higher social support scores. This question was formed based off that observation. The prediction made was that higher social support scores result in higher happiness scores.\n')
    print('')
    print('Question #3: Does freedom to make life choices lower or raise the happiness score of a country?\n')
    print('Prediction: A comparison was made between the highest ranking and lowest ranking country in Africa to make a prediction for this question. It was observed that South Sudan ranked the lowest and Mauritius ranked the highest in happiness. It was drawn that South Sudan had a lower score in freedom to make life choices while Mauritius scored higher. From this, the predicition that was made is that high happiness scores are a result of higher freedom to make life choices scores. The prediciton is that more freedom to make life choices makes a country happy.\n')
    print('')
    print('Question #4: What is the relationship between happiness scores and healthy life expectancy?\n')
    print("Prediciton: This question was formed based on the curiosity of if healthy life expectancy results from a country's happiness or if a country's happiness is evaluated in the fact that their is a healthy life expectancy. The healthy life expectancy column is based on Healthy life expectancies at birth, based on the data extracted from the World Health Organization’s (WHO) Global Health Observatory data repository. By comparing the healthy life expectancy scores between Mauritius and South Sudan, it was predicted that there is a positive relationship between happiness scores and healthy life expectancies. Higher happiness scores appear where there is higher healthy life expectancy.\n")
    print('\n####################################################################################################################')

    
###########################################################################################################
# OVERVIEW
###########################################################################################################  

# function name: overview
# arguments: none
# return: none
# description: outputs an overview of your topic and data set
def overview():
    print('\n####################################################################################################################')
    print('\t\t\t\t\t2020 AFRICA HAPPINESS REPORT\n')
    print('The data set used in this program is from the 2020 World Happiness Report. This annual report is generated from data resulting from the Gallup World Poll. It is a annual landmark survey of the state of global happiness. The filtered data set used in this program contains data only on countries in Africa. This is so that the data-driven questions for this topic may be answered.')
    print('')
    print('This data set has 8 columns. The columns are listed as Country Name, Regional Indicator (Continent), Happiness Score, Social Support, Healthy Life Expectancy, Freedom to Make Life Choices, Generosity and Perceptions of Corruption.')
    print('\n####################################################################################################################\n')   


###########################################################################################################
# START OF GETTING AND CHECKING USER INPUT CHOICE(S)
########################################################################################################### 

# function name: check_choice
# arguments: 2 integer values representing the minimum and maximum numbered options in the menu
# return: the valid input (i.e., a positive digit between the passed minimum and maximum arguments)
# description: asks the user for their choice and then checks that the user's input is a valid digit 
#              between the minimum and maximum arguments passed to it 
#              it keeps asking the for input until the user enters valid input (i.e., a positive digit 
#              between between the passed minimum and maximum arguments)
def check_choice(minimum, maximum):
    # define string variable that is empty to hold user's input
    some_input = ''
    
    # loop until the user's input is no longer empty
    while some_input == '':
        # ask the user to input their choice and store in a variable
        some_input = input('Choice: ')
        print('')
        
        # check if the user's input does not contain only digits
        if some_input.isdigit() == False:
            # output an error message stating that the user did not enter input with only digits
            print('\nYou did not enter input with only digits! Try again.\n')
        
        # otherwise (i.e., the user's input contains only digits)
        else:
            # convert the user's input to an integer
            # "1" --> 1
            some_input = int(some_input)

            # check if the user's input is less than the minimum or greater than the maximum
            if some_input < minimum or some_input > maximum:
                # output an error message stating that the user did not enter input between 1 and 5
                print('\nYou did not enter a valid choice. Choose any option from ' + str(minimum) + ' to ' + str(maximum) + '.\n')
    
    # return the user's input 
    return some_input


# function name: get_choice
# arguments: none
# return: the valid input (i.e., a positive digit between 1 and 5)
# description: outputs the list of five choices for the main menu and calls the check_choice function 
#              passing it 1 as the minimum and 5 as the maximum to obtain a valid input from the user 
def get_choice():
    # output the list of options the user can choose from
    # choice 1 --> Overview of the African Happiness Report Data Set
    # choice 2 --> Data-Driven Questions and Predictions
    # choice 3 --> Basic Statistics on African Happiness
    # choice 4 --> Simple Data Visualizations on African Happiness
    # choice 5 --> Survey Analysis
    # choice 6 --> Data Set Analysis
    # choice 7 --> Findings and Observations
    # choice 8 --> Quit
    print('Choose one of the options below to view the data analysis for this data set and its data-driven question.\n')    
    
    print('1. Overview of the 2020 African Happiness Report Data Set')
    print('2. Data-Driven Questions and Predictions')
    print('3. Basic Statistics on African Happiness')
    print('4. Simple Data Visualizations on African Happiness')
    print('5. Survey Analysis')
    print('6. Data Set Analysis')
    print('7. Findings and Observations')
    print('8. Quit.\n')
    
    
    # call the check_choice function passing it 1 as the minimum and 8 as the maximum
    #      store the returned value in a variable
    choice = check_choice(1, 8)
     
        
    # return the user's input that was stored in the variable above
    return choice

###########################################################################################################
# END OF GETTING AND CHECKING USER INPUT CHOICE(S)
########################################################################################################### 


###########################################################################################################
# START OF SETUP --> WELCOME MESSAGE AND MAIN FUNCTION
###########################################################################################################

# function name: welcome_msg
# arguments: none
# return: none
# description: outputs the title of the program and where the data was collected from (i.e., website, data set)
def welcome_msg():
    dq = 'What is the general state of happiness in Africa?'
    
    print('\t\t\t\t\t2020 AFRICA HAPPINESS DATA SET: ')
    print('\t\t\t\t' + dq.upper())
    print('\n')
    print('\t\t\tWelcome to the 2020 African Happiness data set analysis program!\n')


# function name: main
# arguments: none
# return: none
# description: setups the program and manages calls to other functions defined to handle the eight options
#              it repeats the eight options until the user chooses the option to quit
def main():
    # call welcome_msg function to output the title of the program and where the data was collected from
    welcome_msg()
    
    
    # flag variable keeping track of if the survey data has already been processed/cleaned
    #      assume at the beginning of program that survey data has NOT been processed/cleaned
    #      update this flag variable when the 5th or 7th option have been chosen for the first time
    survey_processed = False
    
    
    # flag variable keeping track of if the DataFrame containing the data from your data set you found online 
    #      has already been processed/cleaned
    #      assume at the beginning of program that DataFrame has NOT been processed/cleaned
    #      update this flag variable when the 6th or 7th option have been chosen for the first time
    data_processed = False
    
    
    # define string variable that is empty to hold user's input
    choice = ''
    
    
    # loop until the user chooses the choice to quit the program
    while choice != 8:
        
        # call the get_choice function to get the choice from the user and store in variable
        choice = get_choice()
        
        # check if the user chose the first option
        if choice == 1:
            
            # call overview function to output an overview of your topic and data set
            overview()
        
        
        # check if the user chose the second option
        elif choice == 2:
            
            # call dq function to output data-driven questions and predictions on topic/data set
            dq()
            
            
        # check if the user chose the third option
        elif choice == 3:
            
            # call basic stats function to provide the user with sub-menu on basic data stats on topic
            basic_stats()
            
        # check if the user chose the fourth option
        elif choice == 4:
            
            # call simple_visualizations function to provide the user with sub-menu on data visualizations
            simple_visualizations()
            
        # check in the user chose fifth or sixth or seventh optopn
        elif choice == 5 or choice == 6 or choice == 7:
            
            # check if the user chose the fifth option
            if choice == 5:
                
                # check if the survey data has not already been processed/cleaned
                if survey_processed == False:
                
                    # create variable storing the name of your survey data CSV file
                    file = "survey.csv"
                    
                    # call the read_csv function
                    some_list, mchoice1, mchoice2, mchoice3, mchoice4 = read_csv(file)
                    
                    # update the flag variable keeping track of if survey data has been processed/cleaned
                    survey_processed = True
                    
                # call the survey analysis function with sub-menu on survey data stats and visualizations
                survey_analysis(some_list, mchoice1, mchoice2, mchoice3, mchoice4)
                
            # check if the user chose the sixth option
            if choice == 6:
                
                # check if the DataFrame containing your data set found online has not already been processed/cleaned
                if data_processed == False:
                    
                    # create variable storing the name of your data set you found online CSV file
                    file2 = "stubbs_africahappiness.csv"
                    
                    # call the cleaning function that cleans the DataFrame
                    cleaned_df = process_df(file2)
                    
                    # update the flag variable keeping track of if DataFrame has been processed/cleaned
                    data_processed = True
                    
                # call the data analysis function with sub-menu on DataFraem pivot tables and visualizations
                data_analysis(cleaned_df)
            
            
            # check if the user chose the seventh option
            if choice ==7:
                
                # check if the survey data has not already been processed/cleaned
                if survey_processed == False:
                
                    # create variable storing the name of your survey data CSV file
                    file = "survey.csv"
                    
                    # call the read_csv function
                    some_list, mchoice1, mchoice2, mchoice3, mchoice4 = read_csv(file)
                    
                    # update the flag variable keeping track of if survey data has been processed/cleaned
                    survey_processed = True
                    
                if data_processed == False:
                    
                    # create variable storing the name of your data set you found online CSV file
                    file2 = "stubbs_africahappiness.csv"
                    
                    # call the cleaning function that cleans the DataFrame
                    cleaned_df = process_df(file2)
                    
                    # update the flag variable keeping track of if DataFrame has been processed/cleaned
                    data_processed = True   
                    
                # call the findings function outputting findings and observations on topic
                #      pass this function:
                #      1 cleaned DataFrame
                #      at least 1 list containing valid/cleaned responses for your linear rating scale question
                #      at least 2 integer variables containing counts for choices in your multiple choice question
                findings(cleaned_df, some_list, mchoice1, mchoice2, mchoice3, mchoice4)
                
        
        
        # check if the user chose the eighth option
        elif choice == 8:
            
            # output program has ended
            print('The program has ended.')  
        
###########################################################################################################
# END OF SETUP --> WELCOME MESSAGE AND MAIN FUNCTION
###########################################################################################################



# call the main function to run your program  
main()


# In[ ]:




