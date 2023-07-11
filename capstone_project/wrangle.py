# standard imports
import pandas as pd
import numpy as np


# visualized your data
import matplotlib.pyplot as plt
import seaborn as sns

# my imports
# from env import get_db_url
# import os
# import env


'''
*------------------*
|                  |
|     ACQUIRE      |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
def check_file_exists(fn, query, url):
    """
    This function will:
    - check if file exists in my local directory, if not, pull from sql db
    - read the given `query`
    - return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df
    
# ----------------------------------------------------------------------------------
# def get_logs_data():
#     """
#     This function will:
#         - from the connection made to the `curriculum_logs` DB
#             - using the `get_db_url` from my wrangle module.
#     """
#     # How to import a database from MySQL
#     # VARIABLES
#     url = get_db_url('curriculum_logs')

#     query = """
#     SELECT * 
#     FROM curriculum_logs.logs as l 
#     JOIN curriculum_logs.cohorts as c ON c.id = l.cohort_id;
#     """

#     filename = 'logs.csv'
    
#     # CREATE DATAFRAME
#     df = check_file_exists(filename, query, url)
    
#     return df
# ----------------------------------------------------------------------------------

# def acquire_logs(user=env.user, password=env.password, host=env.host):
#     '''
#     This function gathers curriculum_logs data from the 
#     SQL codeup database and returns the information in a 
#     pandas dataframe
#     '''
#     url = get_db_url('curriculum_logs')
#     query = '''
#     select * from cohorts;
#     '''
#     df = pd.read_sql(query, url)
#     df.to_csv('cohorts.csv')
# ----------------------------------------------------------------------------------

def get_data():
    """
    This function will:
    - from the connection made to the `curriculum_logs` DB
        - using the `get_db_url` from my wrangle module.
    """
    df = pd.read_csv('anonymized-curriculum-access.txt', sep=" ")
    df_2 = pd.read_csv('cohorts.csv')
    df_2 = df_2.drop(columns ='Unnamed: 0')
    df.loc[len(df.index)] = ['2018-01-26', '09:55:03', '/', 1, 8 , '97.105.19.61']
    df.columns = ['date', 'time', 'path', 'user_id', 'cohort_id', 'ip']

    df = df.merge(df_2, left_on='cohort_id', right_on='id', how='left')

    df = df.drop(columns = 'id')
    df = df.drop(columns = 'deleted_at')
    df = df.drop(columns = 'slack')
    df['date'] = pd.to_datetime( df['date'])
    df['start_date'] = pd.to_datetime( df['start_date'])
    df['end_date'] = pd.to_datetime( df['end_date'])
    df['created_at'] = pd.to_datetime( df['created_at'])
    df['updated_at'] = pd.to_datetime( df['updated_at'])
    df.path = df.path.dropna()
    df.program_id = df.program_id.replace({1: 'full_stack_java_php'})
    df.program_id = df.program_id.replace({2: 'full_stack_java_java'})
    df.program_id = df.program_id.replace({3: 'datascience'})
    df.program_id = df.program_id.replace({4: 'front_end_web_dev'})
    return df 

# ----------------------------------------------------------------------------------
'''
*------------------*
|                  |
|     SUMMARY      |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
# a function that show a summary of the dataset
def data_summary(df):
    """
    This function that show a summary of the dataset 
    """
    # Print the shape of the DataFrame
    print(f'data shape: {df.shape}')
    # set all the columns names to a lowercase
    df.columns = df.columns.str.lower()
    # Create a summary DataFrame
    summary = pd.DataFrame(df.dtypes, columns=['data type'])
    # Calculate the number of missing values
    summary['#missing'] = df.isnull().sum().values 
    # Calculate the percentage of missing values
    summary['%missing'] = df.isnull().sum().values / len(df)* 100
    # Calculate the number of unique values
    summary['#unique'] = df.nunique().values
    # Create a descriptive DataFrame
    desc = pd.DataFrame(df.describe(include='all').transpose())
    # Add the minimum, maximum, and first three values to the summary DataFrame
    summary['count'] = desc['count'].values
    summary['mean'] = desc['mean'].values
    summary['std'] = desc['std'].values
    summary['min'] = desc['min'].values
    summary['25%'] = desc['25%'].values
    summary['50%'] = desc['50%'].values
    summary['75%'] = desc['75%'].values
    summary['max'] = desc['max'].values
    summary['first_value'] = df.loc[0].values
    summary['second_value'] = df.loc[1].values
    summary['third_value'] = df.loc[2].values
    
    # Return the summary DataFrame
    return summary

# ----------------------------------------------------------------------------------
'''
*------------------*
|                  |
|     PREPARE      |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
def remove_columns(df, col_to_remove):
    """
    This function will:
    - take in a df and list of columns (you need to create a list of columns that you would like to drop under the name 'cols_to_remove')
    - drop the listed columns
    - return the new df
    """
    df = df.drop(columns=col_to_remove)
    
    return df

# ----------------------------------------------------------------------------------
def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - calculates the minimum number of non-missing values required for each column/row to be retained
    - drops columns/rows with a high proportion of missing values.
    - returns the new df
    """
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    
    return df

# ----------------------------------------------------------------------------------
def data_prep(df, col_to_remove, prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - list of columns
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - removes unwanted columns
    - remove rows and columns that contain a high proportion of missing values
    - returns cleaned df
    """
    df = remove_columns(df, col_to_remove)
    df = handle_missing_values(df, prop_required_columns, prop_required_rows)
    
    # converts int to datetime
    df.date = pd.to_datetime(df.date)
    
    # rename columns
    df.columns
    df = df.rename(columns={'name':'cohort_name'})   
    
    # rename the numbers for names
    df.program_id = df.program_id.replace({1: 'full_stack_php', 2: 'full_stack_java', 3: 'data_science', 4: 'front_end_web_dev'})
    return df


# ----------------------------------------------------------------------------------
'''
*------------------*
|                  |
|    Question 1    |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
# Define a function to find the most accessed lesson per program
def most_accessed_lesson(df):
    """
    This function takes in a DataFrame and returns a DataFrame with the most accessed lesson per program.
    """
    df = get_data()
    program_1 = df[df.program_id == 'full_stack_java_php']
    program_2 = df[df.program_id == 'full_stack_java_java']
    program_3 = df[df.program_id == 'datascience']
    program_4 = df[df.program_id == 'front_end_web_dev']

    the_list = [program_1, program_2, program_3, program_4]

    
    list_of_most_viewed_full_stack_java_php = []
    list_of_most_viewed_full_stack_java_java = []
    list_of_most_viewed_datascience = []
    list_of_most_viewed_front_end_web_dev = []


    for j in the_list:
        for i in list(set(j.cohort_id)):
            
            answer = j.groupby(by=["cohort_id", 'path']).count()
            answer = answer.reset_index()

            
            if j.program_id.unique()[0] == 'full_stack_java_php':
                answer = answer[answer.path.str.len() > 1]
                answer = answer[answer.path.str.contains('/')]
                answer = answer[answer.path.str.contains('.json') == False]
                answer = answer[answer.path != 'content/php_ii/command-line']
                answer = answer[answer.path != 'content/php_i']
                answer = answer[answer.path != 'html-css/elements']

                the_df = answer[['path', 'cohort_id', 'user_id']][answer.cohort_id == i].sort_values(by ='user_id').tail(1)
                if len(the_df.path.unique()) == 1:

                    list_of_most_viewed_full_stack_java_php.append(the_df.path.iloc[0])

            if j.program_id.unique()[0] == 'full_stack_java_java':
                answer = answer[answer.path.str.len() > 1]
                answer = answer[answer.path.str.contains('/')]
                answer = answer[answer.path.str.contains('.json') == False]
                answer = answer[answer.path != 'content/php_ii/command-line']
                answer = answer[answer.path != 'content/php_i']
                answer = answer[answer.path != 'html-css/elements']

                the_df = answer[['path', 'cohort_id', 'user_id']][answer.cohort_id == i].sort_values(by ='user_id').tail(1)
                if len(the_df.path.unique()) == 1:

                    list_of_most_viewed_full_stack_java_java.append(the_df.path.iloc[0])


            if j.program_id.unique()[0] == 'datascience':
                answer = answer[answer.path.str.len() > 1]
                answer = answer[answer.path.str.contains('/')]
                answer = answer[answer.path.str.contains('.json') == False]
                answer = answer[answer.path != 'content/php_ii/command-line']
                answer = answer[answer.path != 'content/php_i']
                answer = answer[answer.path != 'html-css/elements']

                the_df = answer[['path', 'cohort_id', 'user_id']][answer.cohort_id == i].sort_values(by ='user_id').tail(1)
                if len(the_df.path.unique()) == 1:

                    list_of_most_viewed_datascience.append(the_df.path.iloc[0])

            if j.program_id.unique()[0] == 'front_end_web_dev':
                answer = answer[answer.path.str.len() > 1]
                answer = answer[answer.path.str.contains('/')]
                answer = answer[answer.path.str.contains('.json') == False]
                answer = answer[answer.path != 'content/php_ii/command-line']
                answer = answer[answer.path != 'content/php_i']
                answer = answer[answer.path != 'html-css/elements']

                the_df = answer[['path', 'cohort_id', 'user_id']][answer.cohort_id == i].sort_values(by ='user_id').tail(1)
                if len(the_df.path.unique()) == 1:

                    list_of_most_viewed_front_end_web_dev.append(the_df.path.iloc[0])
                    
                    
    the_dict = {}
    program_names = ['full_stack_java_php','full_stack_java_java','datascience','front_end_web_dev']
    the_dict_of_answers = {}
    list_to_be_added = []
    top_df= pd.DataFrame()
    for iteration, list_of_most_viewed in enumerate([list_of_most_viewed_full_stack_java_php,
                                list_of_most_viewed_full_stack_java_java,
                                list_of_most_viewed_datascience,
                                list_of_most_viewed_front_end_web_dev
                                ]):
        for i in list_of_most_viewed:
            if i in the_dict:
                the_dict[i] += 1
            else:
                the_dict[i] = 1

        for key, values in the_dict.items():
            if the_dict[key] == max(the_dict.values()):
                list_to_be_added.append(key)

        the_dict_of_answers[program_names[iteration]] = list_to_be_added
        
        list_to_be_added = []

        the_dict = {}
    for key, values in the_dict_of_answers.items():
        the_df = pd.DataFrame({'program': str(key), "page":str(values)},  index=[0])
        
        top_df = pd.concat([top_df, the_df])

    return top_df
# ----------------------------------------------------------------------------------


'''
*------------------*
|                  |
|    Question 2    |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
# Define a function to find the lesson that a cohort referred to significantly more than other cohorts
def most_referred_lesson(df):
    """
    This function takes in a DataFrame and returns a DataFrame with the lesson that a cohort referred to significantly more than other cohorts.
    """
    df = get_data()
    program_1 = df[df.program_id == 'full_stack_java_php']
    program_2 = df[df.program_id == 'full_stack_java_java']
    program_3 = df[df.program_id == 'datascience']
    program_4 = df[df.program_id == 'front_end_web_dev']

    the_list = [program_1, program_2, program_3, program_4]

    
    top_df_full_stack_java_php = pd.DataFrame()
    top_df_full_stack_java_java = pd.DataFrame()
    top_df_datascience = pd.DataFrame()
    top_df_front_end_web_dev = pd.DataFrame()


    for j in the_list:
        for i in list(set(j.cohort_id)):
            
            answer = j.groupby(by=["cohort_id", 'path']).count()
            answer = answer.reset_index()
            

            
            if j.program_id.unique()[0] == 'full_stack_java_php':
                answer = answer[answer.path.str.len() > 1]
                answer = answer[answer.path.str.contains('/')]
                answer = answer[answer.path.str.contains('.json') == False]
                answer = answer[answer.path != 'content/php_ii/command-line']
                answer = answer[answer.path != 'content/php_i']
                answer = answer[answer.path != 'html-css/elements']
                
                the_df = answer[['path', 'cohort_id', 'user_id']][answer.cohort_id == i].sort_values(by ='user_id', ascending= False)
                
                if the_df.shape[0] > 0:
                    the_df[the_df.path == 'spring/fundamentals/repositories']
                    top_df_full_stack_java_php = pd.concat([top_df_full_stack_java_php, the_df[the_df.path == 'spring/fundamentals/repositories']]).sort_values(by='user_id', ascending= False)


            if j.program_id.unique()[0] == 'full_stack_java_java':
                answer = answer[answer.path.str.len() > 1]
                answer = answer[answer.path.str.contains('/')]
                answer = answer[answer.path.str.contains('.json') == False]
                answer = answer[answer.path != 'content/php_ii/command-line']
                answer = answer[answer.path != 'content/php_i']
                answer = answer[answer.path != 'html-css/elements']
                
                the_df = answer[['path', 'cohort_id', 'user_id']][answer.cohort_id == i].sort_values(by ='user_id', ascending= False)
                
                if the_df.shape[0] > 0:
                    the_df[the_df.path == 'jquery/ajax/weather-map']
                    top_df_full_stack_java_java = pd.concat([top_df_full_stack_java_java, the_df[the_df.path == 'jquery/ajax/weather-map']]).sort_values(by='user_id', ascending= False)


            if j.program_id.unique()[0] == 'datascience':
                answer = answer[answer.path.str.len() > 1]
                answer = answer[answer.path.str.contains('/')]
                answer = answer[answer.path.str.contains('.json') == False]
                answer = answer[answer.path != 'content/php_ii/command-line']
                answer = answer[answer.path != 'content/php_i']
                answer = answer[answer.path != 'html-css/elements']
                
                the_df = answer[['path', 'cohort_id', 'user_id']][answer.cohort_id == i].sort_values(by ='user_id', ascending= False)
                
                if the_df.shape[0] > 0:
                    the_df[the_df.path == 'classification/overview']
                    top_df_datascience = pd.concat([top_df_datascience, the_df[the_df.path == 'classification/overview']]).sort_values(by='user_id', ascending= False)


    top_df_full_stack_java_php['cohort'] = 'Lassen'
    top_df_full_stack_java_php['program'] = 'full_stack_java_php'

    top_df_full_stack_java_java['cohort'] = 'Staff'
    top_df_full_stack_java_java['program'] = 'full_stack_java_java'

    top_df_datascience['cohort'] = 'Darden'
    top_df_datascience['program'] = 'datascience'

    df_1 = top_df_full_stack_java_php
    df_2 = top_df_full_stack_java_java
    df_3 = top_df_datascience

    return pd.concat([df_1.head(1),df_2.head(1),df_3.head(1),]) 

# ----------------------------------------------------------------------------------


'''
*------------------*
|                  |
|    Question 3    |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------   
# Define a function to find students who, when active, hardly access the curriculum
def less_active_students(df):
    """
    This function takes in a DataFrame and returns a DataFrame with students who, when active, hardly access the curriculum.
    """
    # Group by 'user_id', count the number of occurrences, and reset the index
    user_activity = df.groupby('user_id').size().reset_index(name='count')

    # Find students who accessed the curriculum less than a certain threshold
    # Here, the threshold is set to 10, but it can be adjusted as needed
    less_active = user_activity[user_activity['count'] < 10]

    # Merge with the original dataframe to bring the 'date', 'cohort_name', 'program_id', 'ip' columns
    less_active = less_active.merge(df[['user_id', 'date', 'cohort_name', 'program_id', 'ip']], on='user_id', how='left')

    less_active = less_active.sort_values(by='count', ascending=True)

    return less_active

# ----------------------------------------------------------------------------------

'''
*------------------*
|                  |
|    Question 4    |
|                  |
*------------------*
'''
# ---------------------------------------------------------------------------------- 
# Define a function to find suspicious activity
def find_suspicious_activity(df):
    """
    This function takes in a DataFrame and returns a DataFrame with suspicious activity, such as users/machines/etc accessing the curriculum who shouldn’t be.
    """
    # Group by 'user_id', count the number of occurrences, and reset the index
    user_activity = df.groupby('user_id').size().reset_index(name='count')

    # Find users who accessed the curriculum more than a certain threshold
    # Here, the threshold is set to 10000, but it can be adjusted as needed
    suspicious_activity = user_activity[user_activity['count'] > 10000]

    return suspicious_activity

# ---------------------------------------------------------------------------------- 
# -----------------------------The start of Anomalies-------------------------------
def one_user_df_prep(df, user):
    """
    Prepares a DataFrame for a specific user by filtering the data, converting the 
    'date' column to datetime, and setting it as the index. The DataFrame is then 
    sorted by the index, and the 'endpoint' column is resampled by day and counted.

    Parameters:
        df (pd.DataFrame): The original DataFrame, which should include a 'user_id' 
                           and a 'date' column.
        user (int or str): The user ID to filter the DataFrame by.

    Returns:
        pages_one_user (pd.Series): A Series where the index is the date and the 
                                    value is the count of 'endpoint' for that day.
    """
    df = df[df.user_id == user].copy()
    df.date = pd.to_datetime(df.date)
    df = df.set_index(df.date)
    df = df.sort_index()
    pages_one_user = df['path'].resample('d').count()
    return pages_one_user

# ----------------------------------------------------------------------------------
def compute_pct_b(pages_one_user, span, k, user):
    """
    Computes the Exponential Moving Average (EMA), upper band, lower band, and 
    percentage bandwidth (%b) for a given user's page visits over a specified span 
    of time. The EMA, upper band, and lower band are calculated using a specified 
    number of standard deviations.

    Parameters:
        pages_one_user (pd.Series): A Series where the index is the date and the 
                                    value is the count of 'endpoint' for that day.
        span (int): The span of the window for the EMA calculation, representing 
                    the number of time periods (e.g., 7 for a week, 30 for a month).
        k (int): The number of standard deviations to use when calculating the 
                 upper and lower bands.
        user (int or str): The user ID to be added to the resulting DataFrame.

    Returns:
        my_df (pd.DataFrame): A DataFrame containing the original page visit data, 
                              the EMA (midband), the upper and lower bands (ub and lb), 
                              the %b value (pct_b), and the user ID.
    """
    midband = pages_one_user.ewm(span=span).mean()
    stdev = pages_one_user.ewm(span=span).std()
    ub = midband + stdev*k
    lb = midband - stdev*k
    
    my_df = pd.concat([pages_one_user, midband, ub, lb], axis=1)
    my_df.columns = ['pages_one_user', 'midband', 'ub', 'lb']
    
    my_df['pct_b'] = (my_df['pages_one_user'] - my_df['lb'])/(my_df['ub'] - my_df['lb'])
    my_df['user_id'] = user
    return my_df

# ----------------------------------------------------------------------------------
def plot_bands(my_df, user):
    """
    Plots the number of pages visited by a user, the Exponential Moving Average (EMA or midband), 
    and the upper and lower bands over time. 

    Parameters:
        my_df (pd.DataFrame): A DataFrame containing the original page visit data, 
                              the EMA (midband), the upper and lower bounds (ub and lb), 
                              the %b value (pct_b), and the user ID.
        user (int or str): The user ID to be used in the plot's label.

    Returns:
        None. Displays a plot with the number of pages, EMA/midband, upper band, and lower band 
        over time for the specified user.
    """
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(my_df.index, my_df.pages_one_user, label='Number of Pages, User: '+str(user))
    ax.plot(my_df.index, my_df.midband, label = 'EMA/midband')
    ax.plot(my_df.index, my_df.ub, label = 'Upper Band')
    ax.plot(my_df.index, my_df.lb, label = 'Lower Band')
    ax.legend(loc='best')
    ax.set_ylabel('Number of Pages')
    plt.show()

# ----------------------------------------------------------------------------------
def find_anomalies(df, user, span, k, plot=False):
    """
    Finds anomalies in the number of pages visited by a user over a specified span 
    of time. An anomaly is defined as a value that is above the upper band, which 
    is calculated using the Exponential Moving Average (EMA or midband) and a 
    specified number of standard deviations.

    Parameters:
        df (pd.DataFrame): The original DataFrame, which should include a 'user_id' 
                           and a 'date' column.
        user (int or str): The user ID to filter the DataFrame by.
        span (int): The span of the window for the EMA calculation, representing 
                    the number of time periods (e.g., 7 for a week, 30 for a month).
        k (int): The number of standard deviations to use when calculating the 
                      upper and lower bounds.
        plot (bool, optional): Whether to display a plot of the number of pages, 
                               EMA/midband, upper band, and lower band over time 
                               for the specified user. Defaults to False.

    Returns:
        my_df (pd.DataFrame): A DataFrame containing the original page visit data, 
                              the EMA (midband), the upper and lower bounds (ub and lb), 
                              the %b value (pct_b), and the user ID. Only rows where 
                              pct_b > 1 (indicating an anomaly) are included. If no 
                              anomalies are found, the DataFrame will be empty.
    """

    pages_one_user = one_user_df_prep(df, user)
    
    my_df = compute_pct_b(pages_one_user, span, k, user)
    
    if plot:
        plot_bands(my_df, user)
    
    return my_df[my_df.pct_b>1]

# ----------------------------------------------------------------------------------
def find_all_anomalies(df, span, k):
    """
    Finds anomalies for all users in the provided DataFrame over a specified span 
    of time. An anomaly is defined as a value that is above the upper band, which 
    is calculated using the Exponential Moving Average (EMA or midband) and a 
    specified number of standard deviations.

    Parameters:
        df (pd.DataFrame): The original DataFrame, which should include a 'user_id' 
                           and a 'date' column.
        span (int): The span of the window for the EMA calculation, representing 
                    the number of time periods (e.g., 7 for a week, 30 for a month).
        k (int): The number of standard deviations to use when calculating the 
                 upper and lower bounds.

    Returns:
        anomalies (pd.DataFrame): A DataFrame containing the anomalies for all users. 
                                   Each row includes the original page visit data, 
                                   the EMA (midband), the upper and lower bounds (ub and lb), 
                                   the %b value (pct_b), and the user ID. Only rows where 
                                   pct_b > 1 (indicating an anomaly) are included. If no 
                                   anomalies are found for a user, no rows for that user 
                                   will be included in the DataFrame.
    """
    anomalies = pd.DataFrame()

    for u in df.user_id.unique():
        one_user = find_anomalies(df, u, span, k)
        anomalies = pd.concat([anomalies, one_user])

    return anomalies

# ----------------------------------------------------------------------------------
# Define a function to calculate count
def count(df, column):
    """
    This function counts the values inside the columns
    """
    return df[column].value_counts()

# ----------------------------------------------------------------------------------
# Define a function to calculate frequency
def frequency(df, column):
    """
    This function takes in a DataFrame and a column.
    Returns a DataFrame with the frequency of each item in the column.
    """
    return df[column].value_counts(normalize=True)*100

# ----------------------------------------------------------------------------------
# Define a function to visualize count
def visualize_count(df, column):
    """
    This function takes in a DataFrame and a column.
    Returns a horizontal bar plot with the frequency of each item in the column.
    """
    df[column].value_counts().sort_values().plot(kind='barh')
# -----------------------------End of Anomalies-------------------------------------


'''
*------------------*
|                  |
|    Question 5    |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
# Define a function to find evidence of students and alumni accessing both curriculums (web dev to ds, ds to web dev)
def find_cross_access_after_2019(df):
    """
    This function takes in a DataFrame and returns a DataFrame with evidence of students and alumni accessing both curriculums (web dev to ds, ds to web dev).
    """
    # Filter the DataFrame to include only records from 2019 onwards
    df_2019_onwards = df[df['date'] >= '2019-01-01']

    # Group by 'user_id' and 'program_id', count the number of occurrences, and reset the index
    user_program_activity = df_2019_onwards.groupby(['user_id', 'program_id']).size().reset_index(name='count')

    # Find users who accessed more than one program
    cross_access = user_program_activity[user_program_activity['user_id'].duplicated(keep=False)]

    return cross_access

# ----------------------------------------------------------------------------------
# Define a function to find evidence of students and alumni accessing both curriculums (web dev to ds, ds to web dev)
def find_cross_access_before_2019(df):
    """
    This function takes in a DataFrame and returns a DataFrame with evidence of students and alumni accessing both curriculums (web dev to ds, ds to web dev).
    """
    # Filter the DataFrame to include only records from 2019 onwards
    df_2019_onwards = df[df['date'] <= '2019-01-01']

    # Group by 'user_id' and 'program_id', count the number of occurrences, and reset the index
    user_program_activity = df_2019_onwards.groupby(['user_id', 'program_id']).size().reset_index(name='count')

    # Find users who accessed more than one program
    cross_access = user_program_activity[user_program_activity['user_id'].duplicated(keep=False)]

    return cross_access
# ---------------------------------------------------------------------------------- 


'''
*------------------*
|                  |
|    Question 6    |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
# question 6 was not answered 

# ----------------------------------------------------------------------------------


'''
*------------------*
|                  |
|    Question 7    |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
# Define a function to find the least accessed lessons
def find_least_accessed_lessons(df):
    """
    This function takes in a DataFrame and returns a DataFrame with the least accessed lessons.
    """
    df = get_data()
    program_1 = df[df.program_id == 'full_stack_java_php']
    program_2 = df[df.program_id == 'full_stack_java_java']
    program_3 = df[df.program_id == 'datascience']
    program_4 = df[df.program_id == 'front_end_web_dev']

    program_names = ['full_stack_java_php','full_stack_java_java','datascience','front_end_web_dev']

    the_list = [program_1, program_2, program_3, program_4]

    for iteration , j in enumerate(the_list):
        
        if program_names[iteration] == 'full_stack_java_php':
            df_1 = j.groupby(by='path').count()
            df_1 = df_1.reset_index()
            df_1 = df_1[df_1.path.str.len() > 1]
            df_1 = df_1[df_1.path.str.contains('/')]
            df_1 = df_1[df_1.path.str.contains('.json') == False]
            df_1 = df_1[df_1.path != 'content/php_ii/command-line']
            df_1 = df_1[df_1.path != 'content/php_i']
            df_1 = df_1[df_1.path != 'html-css/elements']
            df_1 = df_1.sort_values(by='user_id').head(1)
            df_1['program'] = 'full_stack_java_php'
        if program_names[iteration] == 'full_stack_java_php':
            df_2 = j.groupby(by='path').count()
            df_2 = df_2.reset_index()
            df_2 = df_2[df_2.path.str.len() > 1]
            df_2 = df_2[df_2.path.str.contains('/')]
            df_2 = df_2[df_2.path.str.contains('.json') == False]
            df_2 = df_2[df_2.path != 'content/php_ii/command-line']
            df_2 = df_2[df_2.path != 'content/php_i']
            df_2 = df_2[df_2.path != 'html-css/elements']
            df_2 = df_2.sort_values(by='user_id').head(1)
            df_2['program'] = 'full_stack_java_php'
        if program_names[iteration] == 'datascience':
            df_3 = j.groupby(by='path').count()
            df_3 = df_3.reset_index()
            df_3 = df_3[df_3.path.str.len() > 1]
            df_3 = df_3[df_3.path.str.contains('/')]
            df_3 = df_3[df_3.path.str.contains('.json') == False]
            df_3 = df_3[df_3.path != 'content/php_ii/command-line']
            df_3 = df_3[df_3.path != 'content/php_i']
            df_3 = df_3[df_3.path != 'html-css/elements']
            df_3 = df_3.sort_values(by='user_id').head(1)
            df_3['program'] = 'datascience'
        if program_names[iteration] == 'front_end_web_dev':
            df_4 = j.groupby(by='path').count()
            df_4 = df_4.reset_index()
            df_4 = df_4[df_4.path.str.len() > 1]
            df_4 = df_4[df_4.path.str.contains('/')]
            df_4 = df_4[df_4.path.str.contains('.json') == False]
            df_4 = df_4[df_4.path != 'content/php_ii/command-line']
            df_4 = df_4[df_4.path != 'content/php_i']
            df_4 = df_4[df_4.path != 'html-css/elements']
            df_4 = df_4.sort_values(by='user_id').head(1)
            df_4['program'] = 'front_end_web_dev'

    return pd.concat([df_1, df_2, df_3, df_4])[['path','program','user_id']].rename(columns ={'user_id':'count'})

# ----------------------------------------------------------------------------------


'''
*------------------*
|                  |
|    Question 8    |
|                  |
*------------------*
'''
# ----------------------------------------------------------------------------------
# This questions was not answered
