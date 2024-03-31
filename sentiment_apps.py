import finnhub
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
import warnings
import datetime
import streamlit as st
import plotly.figure_factory as ff


st.title("Sentiment Analaysis of Stocks based on News:zap:")

def get_data_API():
    # Setup client
    finnhub_client = finnhub.Client(api_key='cnv8u01r01qub9j0j5i0cnv8u01r01qub9j0j5ig')

    res = finnhub_client.company_news('AAPL', _from="2024-01-01", to="2024-03-31")
    res1 = finnhub_client.company_news('GOOG', _from="2024-01-01", to="2024-03-31")
    res2 = finnhub_client.company_news('META', _from="2024-01-01", to="2024-03-31")
    z1=pd.DataFrame(res1)
    z= pd.DataFrame(res)
    z2= pd.DataFrame(res2)

    z= z._append(z1,ignore_index=True)
    z= z._append(z2,ignore_index=True)
    z['datetime'] = pd.to_datetime(z['datetime'],unit='s')
    z['datetime'] = z['datetime'].apply(lambda x: x.date())
    return z

warnings.filterwarnings("ignore")
stopwords_list = stopwords.words('english')
exclude = string.punctuation

def remove_punc(text):
    return text.translate(str.maketrans('','',exclude))

def remove_stopwords(text):
    clean_text = []  
    data = [word for word in text.split() if word not in stopwords_list]
    return " ".join(data)

def text_data_cleaning(df):
    z['summary_new'] = z['summary'].copy()
    z['summary'] = z['summary'].str.lower()
    z['summary'] = z['summary'].apply(remove_punc)  
    z['summary'] = z['summary'].apply(remove_stopwords)
    z['headline_new'] = z['headline'].copy()
    z['headline'] = z['headline'].str.lower()
    z['headline'] = z['headline'].apply(remove_punc)  
    z['headline'] = z['headline'].apply(remove_stopwords)
    return df

z= get_data_API()
z = text_data_cleaning(z)

z['compound']=z['summary'].apply(lambda x:TextBlob(x).sentiment.polarity)
z['head_compound']=z['headline'].apply(lambda x:TextBlob(x).sentiment.polarity)

plt.figure(figsize=(15,15))
mean_df = z.groupby(['related','datetime']).mean(numeric_only=True)
#mean_df = z.groupby(['related']).mean(numeric_only=True)
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis = 'columns').transpose()

#mean_df.plot(kind = 'bar')
#plt.show()
st.bar_chart(mean_df)
#st.altair_chart(mean_df)
#st.plotly_chart(mean_df,use_container_width=True)
