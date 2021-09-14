import texthero as hero
import pandas as pd

df = pd.read_csv('sample.csv')

df['clean_text'] = df['text'].pipe(hero.clean)

print(df['clean_text'])

