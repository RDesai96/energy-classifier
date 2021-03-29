import csv
import os
import pandas as pd

def clean_solar(filepath = 'data/cleandata/SolarCleaned.csv', delimiter = ','):

    """Cleans the scraped solar dataset obtained by solar_spider.py and returns a csv file
    (to the location specified by filepath, default path is data/cleandata/SolarCleaned.csv)
    and a pandas.df. A custom delimiter can be specified by the delimiter argument if desired"""

    raw_file  = 'data/rawdata/SolarRawRetrieve.csv'
    temp_file = 'data/cleandata/SolarCleaning.csv'
    if os.path.exists(temp_file):
        os.remove(temp_file)

    with open(raw_file, 'r') as f:
        for line in f:
            with open(temp_file, 'a') as C:
                C.writelines(line)

    with open(temp_file) as D:
        reader = csv.reader(D, delimiter='\t')
        data_tuples = [ ( row[0] + ',' + row[1], row[-2], row[-4], row[-3]) for row in reader]

    df = pd.DataFrame(data_tuples, columns = ['Location', 'Capacity', 'Dev1', 'Dev2'])
    os.remove(temp_file)

    unwanted_strings1 = ['\xa0', ',Operating' , ',Private', ',Under Development', ',Public',
                        ',Under Development','Public (State Owned)', ',Public (City Owned)' ,
                        ',Under Construction', ',Photovoltaic (PV)','(City Owned)','(State Owned)']
    pattern1 = '|'.join(unwanted_strings1)
    df['Location'] = df['Location'].str.replace(pattern1, '', regex=True)



    df.to_csv(filepath, sep = delimiter)
    return df

df = clean_solar()
