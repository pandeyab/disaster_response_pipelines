import sys
import pandas as pd
from sqlalchemy import create_engine

#data loading function
def load_data(messages_filepath, categories_filepath):
    '''
    loading function for message and categories data
    Parameters:   messages_filepath, categories_filepath
    Returns: A pandas DataFrame containing both files
    '''
    

    msg = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #splitting categories
    catg = categories['categories'].str.split(';',expand=True)

    row = catg.iloc[[1]].values[0]
    catg.columns = [ x.split("-")[0] for x in row]

    print("after splitting, categories are: ",catg)

    #convert to numerical
    for clm in catg:

        catg[clm] = catg[clm].map(lambda x: 1 if int(x.split("-")[1]) > 0 else 0 )	

    print("after numerical changes categories are: ", catg)
    joined = pd.concat([msg,catg],join="inner", axis=1)

    return joined


#data cleaning function
def clean_data(df):
    '''
    dropping categories column from the dataframe
    Parameters: pandas DataFrame
    Returns: clean pandas DataFrame (after dropping duplicates)
    '''
    df = df.drop_duplicates()
    return df


#data saving function
def save_data(df, database_filename):
    '''
    saving the data into sql database
    Parameters:   df: The pandas DataFrame which contains the disaster messages and the categories that were cleaned
            database_filename: sql database name and path
    Returns:  saving db
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('labeled_messages', engine, if_exists='replace', index=False)
 

#main function
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

#calling function
if __name__ == '__main__':
    main()