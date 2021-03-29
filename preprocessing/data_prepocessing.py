from preprocessing.data_cleaner import clean_solar

df = clean_solar()

df.iloc[79,0]  = 'Kern County,California'
df.iloc[[191,192,194],0] = 'Indianapolis,Indiana'
df.iloc[233,0] = 'Riverside,California'
df.iloc[344,0] = 'Madera County,California'
df.iloc[345,0] = 'Fort Mohave,Arizona'
df.iloc[346,0] = 'Panoche Valley,California'
df.iloc[346,1] = '130 MW'
df.iloc[347,0] = 'Imperial County,California'
df.iloc[348,0] = 'Boulder City,Nevada'
df.iloc[350,0] = 'Imperial County, California'
df.drop(df.infdex[[88,89,131,132,157,230,245,328,335]], inplace=True)
