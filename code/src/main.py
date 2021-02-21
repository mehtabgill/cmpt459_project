import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from helper1 import *

def plot_bargraph(title, x_label, y_label, x_attribute, y_attribute):
    plt.subplots(figsize=(19, 10))
    plt.title(title)
    graph = sns.barplot(x=x_attribute, y=y_attribute)
    graph.set(xlabel=x_label, ylabel=y_label)
    plt.savefig('../plots/' + title + '.png')

def plot_countplot(df, title, x_label, y_label, x_attribute, hue=None, hue_order=None, class_order=None, width=19, height=10):
    plt.subplots(figsize=(width, height))
    plt.title(title)
    graph = ''
    if hue != None:
        graph = sns.countplot(data=df, x=x_attribute, hue=hue, hue_order = hue_order, order=class_order)
    else:
        graph = sns.countplot(data=df, x=x_attribute)
    graph.set(xlabel=x_label, ylabel=y_label)
    plt.savefig('../plots/' + title + '.png')

def plot_scatterplot(df, title, x_label, y_label, column_x, column_y, ):
    plt.subplots(figsize=(19, 10))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x=df[column_x], y=df[column_y])
    plt.savefig('../plots/' + title + '.png')

# Helper functions
# get data frame by name, [train, test, location]
def get_data_frame(name='train'):
    if name == 'train':
       return pd.read_csv('../data/cases_train.csv')
    elif name == 'location':
        return pd.read_csv('../data/location.csv')

# prints missing values and returns list of columns and columns's missing value
def print_num_of_missing_vals(df):
    col_names = []
    col_na = []
    num_of_rows = len(df.index)
    for column in df.columns:
        col_names.append(column)
        col_na.append((len(df[df[column].isnull()])/num_of_rows)*100)
        print(column, " ", len(df[df[column].isnull()]))
    return col_names, col_na

# 1.1 Data Analysis cases_train.csv 
def perform_data_analysis_train():
    df = get_data_frame()
    print("---- Dataset -> cases_train.csv --------------------")
    col_names, col_na = print_num_of_missing_vals(df)

    plot_bargraph('Missing Values (cases_train)', 'Attributes', 'Total percentage of values missing', col_names, col_na)
    
    # plot countries v/s outcome for top 5 countries
    top_5_countries = df['country'].value_counts().nlargest(5).index
    country_df = df[df['country'].isin(top_5_countries)]

    plot_countplot(country_df, 'Top 5 Countries_vs_Outcome', 'Countries', 'Outcome', x_attribute='country', hue='outcome')
    plot_countplot(country_df, 'Top 5 Countries Frequency wise', 'Countries', 'Frequency', x_attribute='country')

    # Plot Sex
    plot_countplot(country_df, 'Sex vs Outcome (cases_train)', 'Sex', 'Outcome', x_attribute='sex', hue='outcome')

    # combination of longitude and latitude
    plot_scatterplot(df=df, title='Longitude and Latitude (cases_train)', x_label='Longitude', y_label='Latitude', column_x='longitude', column_y='latitude')

    # Top 5 countries's top 3 provinces
    top_provs = []
    for c in top_5_countries:
        temp_df = country_df[country_df['country'] == c]
        top_states = temp_df['province'].value_counts().nlargest(5).index
        top_provs.extend(top_states)
    top_provs_df = country_df[country_df['province'].isin(top_provs)]
    plot_countplot(top_provs_df,title='Top 3 Provinces in Top 5 countries (cases_train)', x_label='Country and Provinces', y_label='Count', x_attribute='country', hue='province', hue_order=top_provs, class_order= top_5_countries)

    # plot age frequency
    isDigit_age_df = df[df['age'].notna()]
    isDigit_age_df = isDigit_age_df.loc[isDigit_age_df['age'].str.isdigit()]
    isDigit_age_df = isDigit_age_df.sort_values(by='age')
    plot_countplot(df=isDigit_age_df, title='Age Frequency (cases_train)', x_label='Age', y_label='Frequency', x_attribute='age', width=25, class_order=isDigit_age_df['age'])

    # plot month frequency
    df['date_confirmation'] = pd.to_datetime(df['date_confirmation'], errors='coerce')
    df = df[df['date_confirmation'].notna()]
    df_f = df.loc[df['date_confirmation'].dt.year.between(2020, 2020)]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'dec']
    month_counts = []
    for month in range(1, 13):
        count = len(df.loc[df['date_confirmation'].dt.month.between(month, month)])
        month_counts.append(count)
    plot_bargraph(title='Month Frequency (cases_train)', x_label='Month', y_label='Frequency', x_attribute=month_names, y_attribute=month_counts)

# 1.1 Data Analysis location.csv
def perform_data_analysis_location():
    df = get_data_frame('location')
    print("---- Dataset -> location.csv --------------------")
    col_names, col_na = print_num_of_missing_vals(df)
    
    # Attribute missing values
    plot_bargraph('Missing Values (location)', 'Attributes', 'Total percentage of values missing', col_names, col_na)

    # combination of longitude and latitude
    plot_scatterplot(df=df, title='Longitude and Latitude (location)', x_label='Longitude', y_label='Latitude', column_x='Long_', column_y='Lat')

    # Top 10 countries with max confirmed cases
    top_10_countries_confirmed = df.groupby(['Country_Region'])['Confirmed'].sum().sort_values(ascending=False).nlargest(10)
    plot_bargraph(title='Top 10 Confirmed cases countries (location)', x_label='Country', y_label='# of Confirmed cases', x_attribute=top_10_countries_confirmed.index, y_attribute=top_10_countries_confirmed.values)
    # Top 10 countries with max Deaths
    top_10_countries_confirmed = df.groupby(['Country_Region'])['Deaths'].sum().sort_values(ascending=False).nlargest(10)
    plot_bargraph(title='Top 10 Deaths cases countries (location)', x_label='Country', y_label='# of Deaths', x_attribute=top_10_countries_confirmed.index, y_attribute=top_10_countries_confirmed.values)
    # Top 10 countries with max Recovered cases
    top_10_countries_confirmed = df.groupby(['Country_Region'])['Recovered'].sum().sort_values(ascending=False).nlargest(10)
    plot_bargraph(title='Top 10 Recovered cases countries (location)', x_label='Country', y_label='# of Recovered cases', x_attribute=top_10_countries_confirmed.index, y_attribute=top_10_countries_confirmed.values)
    # Top 10 countries with max Active cases
    top_10_countries_confirmed = df.groupby(['Country_Region'])['Active'].sum().sort_values(ascending=False).nlargest(10)
    plot_bargraph(title='Top 10 Active cases countries (location)', x_label='Country', y_label='# of Active cases', x_attribute=top_10_countries_confirmed.index, y_attribute=top_10_countries_confirmed.values)

    # Top 10 frequent countries
    top_10_countries = df['Country_Region'].value_counts().nlargest(10)
    plot_bargraph(title='Top 10 Most Frequent countries (location)', x_label='Countries', y_label='Frequency', x_attribute=top_10_countries.index, y_attribute=top_10_countries.values)

    # Top 10 frequent countries top 3 frequent provinces
    top_countries_df = df[df['Country_Region'].isin(top_10_countries.index)]
    top_provs = []
    for c in top_10_countries.index:
        temp = top_countries_df[top_countries_df['Country_Region'] == c]
        provs = temp['Province_State'].value_counts().nlargest(3).index
        top_provs.extend(provs)
    top_provs_df = df[df["Province_State"].isin(top_provs)]
    plot_countplot(df=top_provs_df, title='Top Countries"s Top 3 provinces (location)', x_label='Countries', y_label='Frequency', x_attribute='Country_Region', hue='Province_State', hue_order=top_provs, class_order=top_10_countries.index)

    # top incidence rate regions
    top_incidence = df.sort_values(by='Incidence_Rate', ascending=False).head(5)
    plot_bargraph(title='Top 5 Incidence Rate regions (location)', x_label='Region', y_label='Incidence rate', x_attribute=top_incidence['Combined_Key'], y_attribute=[float(i) for i in top_incidence['Incidence_Rate']])

    # top case fatility rate regions
    top_incidence = df.sort_values(by='Case-Fatality_Ratio', ascending=False).head(5)
    plot_bargraph(title='Top 5 Case Fatility Rate regions (location)', x_label='Region', y_label='Case Fatality Ratio rate', x_attribute=top_incidence['Combined_Key'], y_attribute=[float(i) for i in top_incidence['Case-Fatality_Ratio']])


# 1.3 - Plot box plots and get outliers using IQR
def outlier_detection_elimination():
    print('-------- Performing Outlier Detection and Elimination------')
    df = get_data_frame()

    print("-------- For 'Age' --------")
    isDigit_age_df = df[df['age'].notna()]
    isDigit_age_df = isDigit_age_df.loc[isDigit_age_df['age'].str.isdigit()]
    isDigit_age_df['age'] = isDigit_age_df['age'].astype(float)
    sns.boxplot(x=isDigit_age_df['age'])
    plt.savefig('../plots/outliers/' + 'Cases Train- ' + 'Age' + '.png')
    # TODO: Decide the quantile 
    q1, q3 = np.percentile(isDigit_age_df['age'], [25, 75])
    print('Quantile1 and Quantile3 -> ', q1, q3)
    iqr = q3-q1
    print('IQR -> ', iqr)
    lower_bound = q1 - (1.5*iqr)
    upper_bound = q3 + (1.5*iqr)
    print('Lower and Upper bound -> ', lower_bound, upper_bound)
    # No values in dataset that are less than 0
    isDigit_age_df = isDigit_age_df.loc[(isDigit_age_df['age'] < 0.0) | (isDigit_age_df['age'] > upper_bound)]
    isDigit_age_df.to_csv('../data/outliers/Age.csv')
    print('------ Outliers saved to ---->  ./code/data/outliers/')

    numeric_cols = ['latitude', 'longitude']
    for column in numeric_cols:
        print("-------- For " + column + "--------")
        temp_df = df[df[column].notna()]
        sns.boxplot(x=temp_df[column])
        plt.savefig('../plots/outliers/' + 'Cases Train- ' + column + '.png')
        q1, q3 = np.percentile(temp_df[column], [25, 75])
        print('Quantile1 and Quantile3 -> ', q1, q3)
        iqr = q3 - q1
        print('IQR -> ', iqr)
        lower_bound = q1 - (1.5*iqr)
        upper_bound = q3 + (1.5*iqr)
        print('Lower and Upper bound -> ', lower_bound, upper_bound)
        temp_df = temp_df.loc[(temp_df[column] < 0.0) | (temp_df[column] > upper_bound)]
        temp_df.to_csv('../data/outliers/'+column+'.csv')
        print('------ Outliers saved to ---->  ./code/data/outliers/')

def preprocess():

    # readdata
    train_data = pd.read_csv('../data/cases_train.csv')
    location_data = pd.read_csv('../data/location.csv')
    test_data = pd.read_csv('../data/cases_test.csv')


    # =========== 1.2 Data Cleanning =======
    # rename the location data columns
    location_data.rename({'Country_Region': 'country', 'Province_State': 'province'}, axis=1, inplace=True)

    # refactor age columns
    train_data['age'] = train_data['age'].apply(lambda x: transform_age(x) if np.all(pd.notnull(x)) else x)
    test_data['age'] = test_data['age'].apply(lambda x: transform_age(x) if np.all(pd.notnull(x)) else x)
    # drop some columns
    DROP_COLUMNS = ['additional_information', 'source']
    train_data.drop(DROP_COLUMNS, axis=1, inplace=True)
    test_data.drop(DROP_COLUMNS, axis=1, inplace=True)

    AVERAGE_COLUMNS = ['age']
    for column in AVERAGE_COLUMNS:
        mean_val_train = train_data[column].mean()
        mean_val_test = test_data[column].mean()
        train_data[column].fillna(mean_val_train, inplace=True)
        test_data[column].fillna(mean_val_test, inplace=True)

    # fill sex columns using a random value
    train_data['sex'] = train_data['sex'].apply(lambda x: x if np.all(pd.notnull(x)) else generate_sex())
    test_data['sex'] = test_data['sex'].apply(lambda x: x if np.all(pd.notnull(x)) else generate_sex())

    # transform `date_confirmation` to month representation
    train_data['date_confirmation'] = train_data['date_confirmation'].apply(
        lambda x: transform_datetime(x) if np.all(pd.notnull(x)) else x)
    train_data['date_confirmation'] = pd.DatetimeIndex(train_data['date_confirmation']).month

    test_data['date_confirmation'] = test_data['date_confirmation'].apply(
        lambda x: transform_datetime(x) if np.all(pd.notnull(x)) else x)
    test_data['date_confirmation'] = pd.DatetimeIndex(test_data['date_confirmation']).month

    # drop row with NA value in certain columns
    DROPNA_COLUMNS = ['date_confirmation', 'country']
    train_data.dropna(subset=DROPNA_COLUMNS, inplace=True)
    test_data.dropna(subset=DROPNA_COLUMNS, inplace=True)

    # finally cast type to int
    train_data = train_data.astype({"age": int, "date_confirmation": int})
    test_data = test_data.astype({"age": int, "date_confirmation": int})

    # set the province

    train_data['province'] = train_data.apply(
        lambda row: get_province(row) if pd.isnull(row['province']) else row['province'], axis=1)
    train_data.isnull().sum().sort_values(ascending=False)
    # see all the missing values
    print(train_data.isnull().sum().sort_values(ascending = False))


    # ================== 1.4 Transform Location dataset ======================
    # aggregate the location dataset
    AGG_MAP = {'Confirmed': 'sum',
               'Deaths': 'sum',
               'Recovered': 'sum',
               'Active': 'sum',
               'Incidence_Rate': 'mean',
               'Case-Fatality_Ratio': 'mean'
               }
    country_province_location_data = location_data.groupby(['province', 'country']).agg(AGG_MAP).reset_index()

    train_data['Combined_Key'] = train_data.apply(lambda row: generate_combined_key(row), axis=1)
    test_data['Combined_Key'] = test_data.apply(lambda row: generate_combined_key(row), axis=1)
    country_province_location_data['Combined_Key'] = country_province_location_data.apply(
        lambda row: generate_combined_key(row), axis=1)

    LOCATION_DROP_COLUMNS = ["Lat", "Long_", "Last_Update"]



    # =========================== 1.5 Joing cases and location dataset =====================

    # join the two  dataset for train data
    after_join1_train = pd.merge(train_data.drop(["province", "country"], axis=1),
                                 country_province_location_data[country_province_location_data['province'].notnull()],
                                 how='right', on=['Combined_Key'])
    # use the rows without a province in location dataset
    after_join2_train = pd.merge(train_data.drop(["province", "Combined_Key"], axis=1),
                                 location_data[location_data['province'].isnull()].drop(LOCATION_DROP_COLUMNS, axis=1),
                                 how='right', on=['country'])

    after_join_train = pd.concat([after_join1_train, after_join2_train])
    after_join_train.drop(["province"], axis=1, inplace=True)
    after_join_train.dropna(subset=["outcome"], inplace=True)

    after_join1_test = pd.merge(test_data.drop(["province", "country"], axis=1),
                                country_province_location_data[country_province_location_data['province'].notnull()],
                                how='right', on=['Combined_Key'])
    after_join2_test = pd.merge(test_data.drop(["province", "Combined_Key"], axis=1),
                                location_data[location_data['province'].isnull()].drop(LOCATION_DROP_COLUMNS, axis=1),
                                how='right', on=['country'])

    after_join_test = pd.concat([after_join1_test, after_join2_test])
    after_join_test.drop(["province"], axis=1, inplace=True)
    after_join_train.dropna(subset=["outcome"], inplace=True)


    # Write result to csv file
    after_join_train.to_csv(path_or_buf='../data/clean_cases_train.csv', index=False)
    after_join_test.to_csv(path_or_buf='../data/clean_cases_test.csv', index=False)
    location_data.to_csv(path_or_buf='../data/aggregated_location.csv', index=False)


if __name__ == '__main__':

    perform_data_analysis_train()
    perform_data_analysis_location()

    outlier_detection_elimination()

