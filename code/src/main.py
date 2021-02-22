import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from helper1 import *

# 1.1 Data Analysis cases_train.csv 
def perform_data_analysis_train():
    df = get_data_frame()
    print("---- Dataset -> cases_train.csv --------------------")
    col_names, col_na = print_num_of_missing_vals(df)

    print('---- Numerical attributes Statistics -------')
    numerical_cols = ['latitude', 'longitude']
    for col in numerical_cols:
        print('For -> ', col)
        print(df[col].describe())
        print()

    print('---------- Plotting graphs--------')
    plot_bargraph('Missing Values (cases_train)', 'Attributes', 'Total percentage of values missing', col_names, col_na)
    
    # plot countries v/s outcome for top 5 countries
    top_5_countries = df['country'].value_counts().nlargest(5).index
    country_df = df[df['country'].isin(top_5_countries)]

    plot_countplot(country_df, 'Top 5 Countries_vs_Outcome (cases_train)', 'Countries', 'Outcome', x_attribute='country', hue='outcome')
    plot_countplot(country_df, 'Top 5 Countries Frequency wise (cases_train)', 'Countries', 'Frequency', x_attribute='country')

    # Plot Sex
    plot_countplot(country_df, 'Sex Frequency (cases_train)', 'Sex', 'Frequency', x_attribute='sex')
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

    numerical_cols = ['Lat', 'Long_', 'Confirmed', 'Deaths','Recovered','Active','Incidence_Rate','Case-Fatality_Ratio']
    print('---- Numerical attributes Statistics -------')
    for col in numerical_cols:
        print('For -> ', col)
        print(df[col].describe())
        print()
    
    print('---------- Plotting graphs--------')
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

# =========== 1.2 Data Cleanning and Imputing values =======
def preprocess():
    # read data
    train_data = get_data_frame('train')
    location_data = get_data_frame('location')
    test_data = get_data_frame('test')

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
    train_country_prov_pairs= l = set( list(zip(train_data['province'], train_data['country']))  )
    # train_country_prov_pairs = {('Gujarat', 'India')}
    test_country_prov_pairs= l = set( list(zip(test_data['province'], test_data['country']))  )
    train_mean_age = train_data['age'].mean()
    test_mean_age = test_data['age'].mean()

    for pair in train_country_prov_pairs:
        temp_df = train_data.loc[(train_data['province'] == pair[0]) & (train_data['country'] == pair[1])]
        mean_age = temp_df['age'].mean()
        if math.isnan(mean_age) :
            country = train_data.loc[train_data['country'] == pair[1]]
            mean_age = country['age'].mean()
            if math.isnan(mean_age):
                mean_age = train_mean_age
        
        train_data.loc[ (train_data['province'] == pair[0]) & (train_data['country'] == pair[1]) & (train_data['age'].eq('') | train_data['age'].isnull()), 'age'] = mean_age

    for pair in test_country_prov_pairs:
        temp_df = test_data.loc[(test_data['province'] == pair[0]) & (test_data['country'] == pair[1])]
        mean_age = temp_df['age'].mean()
        if math.isnan(mean_age) :
            country = test_data.loc[test_data['country'] == pair[1]]
            mean_age = country['age'].mean()
            if math.isnan(mean_age):
                mean_age = test_mean_age
        test_data.loc[ (test_data['province'] == pair[0]) & (test_data['country'] == pair[1]) & (test_data['age'].eq('') | test_data['age'].isnull()), 'age'] = mean_age

    for column in AVERAGE_COLUMNS:
        mean_val_train = train_data[column].mean()
        mean_val_test = test_data[column].mean()
        train_data[column].fillna(mean_val_train, inplace=True)
        test_data[column].fillna(mean_val_test, inplace=True)

    #fill sex columns using a random value
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
    print(train_data)

    # write cleaned data sets
    train_data.to_csv(path_or_buf='../data/clean_cases_train.csv', index=False)
    test_data.to_csv(path_or_buf='../data/clean_cases_test.csv', index=False)
    
    #set the province
    # train_data['province'] = train_data.apply(
    #     lambda row: get_province(row) if pd.isnull(row['province']) else row['province'], axis=1)
    # train_data.isnull().sum().sort_values(ascending=False)
    
    # # see all the missing values
    # print(train_data.isnull().sum().sort_values(ascending = False))


# 1.3 - Plot box plots and get outliers using IQR
def outlier_detection_elimination():
    print('-------- Performing Outlier Detection and Elimination------')
    df = get_data_frame('clean train')
    print(df['age'].describe())
    numeric_cols = ['latitude', 'longitude', 'age']
    for column in numeric_cols:
        print("-------- For " + column + "--------")
        temp_df = df
        plt.subplots(figsize=(19, 10))
        plt.title( column + ' Outliers')
        graph = sns.boxplot(x=temp_df[column])
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

# =========================== 1.4 Location transformation =====================
def transformation():
    location_data = get_data_frame('location')
    location_data.rename({'Country_Region': 'country', 'Province_State': 'province'}, axis=1, inplace=True)
    AGG_MAP = {'Confirmed': 'sum',
               'Deaths': 'sum',
               'Recovered': 'sum',
               'Active': 'sum',
               'Incidence_Rate': 'mean',
               'Case-Fatality_Ratio': 'mean'
               }
    country_province_location_data = location_data.groupby(['province', 'country']).agg(AGG_MAP).reset_index()
    country_province_location_data['Combined_Key'] = country_province_location_data.apply(
        lambda row: generate_combined_key(row), axis=1)
    country_province_location_data.to_csv(path_or_buf='../data/aggregated_location.csv', index=False)

# =========================== 1.5 Joing cases and location dataset =====================
def joining():
    LOCATION_DROP_COLUMNS = ["Lat", "Long_", "Last_Update"]
    train_data = get_data_frame('clean train')
    test_data = get_data_frame('clean test')
    country_province_location_data = get_data_frame('agg location')
    location_data = get_data_frame('location')
    location_data.rename({'Country_Region': 'country', 'Province_State': 'province'}, axis=1, inplace=True)

    train_data['Combined_Key'] = train_data.apply(lambda row: generate_combined_key(row), axis=1)
    test_data['Combined_Key'] = test_data.apply(lambda row: generate_combined_key(row), axis=1)
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
    after_join_train.to_csv(path_or_buf='../data/joined_cases_train.csv', index=False)
    after_join_test.to_csv(path_or_buf='../data/joined_cases_test.csv', index=False)
    location_data.to_csv(path_or_buf='../data/aggregated_location.csv', index=False)
    print('joined data sets written to ./data/')

if __name__ == '__main__':

    print('======== Performing 1.1 ================')
    perform_data_analysis_train()
    perform_data_analysis_location()

    print('======== Performing 1.2 ================')
    preprocess()

    print('======== Performing 1.3 ================')
    outlier_detection_elimination()

    print('======== Performing 1.4 ================')
    transformation()

    print('======== Performing 1.5 ================')
    joining()

