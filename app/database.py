import OpenBlender
import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
import json
from sqlalchemy import BigInteger, Column, Float, MetaData, Table, Integer, Date, String, Text, ForeignKey, create_engine
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Boolean

Base = declarative_base()
meta = MetaData()
engine = create_engine("sqlite:///database.db", echo=False)

class Bitok(Base):
    __tablename__ = 'Bitcoin_Data'
    index = Column(BigInteger(), index=True)
    volume = Column(Float(), primary_key=True)
    market_cap = Column(Float())
    high = Column(Float())
    low = Column(Float())
    close = Column(Float())
    open = Column(Float())
    timestamp = Column(Integer, ForeignKey('News_Data.timestamp'))


class News(Base):
    __tablename__ = 'News_Data'
    index = Column(BigInteger(), index=True)
    headline = Column(Text())
    publisher = Column(Text())
    title = Column(Text(), primary_key=True)
    url = Column(Text())
    timestamp = Column(Float(), primary_key=True)


def pullObservationsToDF(parameters):
    action = 'API_getObservationsFromDataset'
    df = pd.read_json(json.dumps(OpenBlender.call(action, parameters)['sample']), convert_dates=False,
                      convert_axes=False).sort_values('timestamp', ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df

def create_bitcoin(dataset):

    dataset.to_sql('Bitcoin_Data', con=engine)

def create_news(dataset):
    dataset.to_sql('News_Data', con=engine)

# цены биткоин
parameters_prices = {
    'token': '6375135895162959cf33dea6ncr1sn1ZuDno6ZjhRtnVK3nsznDuAv',
    'id_dataset': '5d4c3b789516290b02fe3e24',
}
# новости
parameters_news = {
    'token': '6375135895162959cf33dea6ncr1sn1ZuDno6ZjhRtnVK3nsznDuAv',
    'id_dataset': '5defce899516296bfe37c366',
}

#ETH

#df = pd.read_csv('ETH-USD.csv', sep=',', quotechar='|')

#print(df.head())

#df.to_sql("ETH-USD", con = engine)

#Bitcoin

#df2 = pd.read_csv('BTC-USD.csv', sep=',', quotechar='|')

#print(df2.head())

#df2.to_sql("BTC-USD", con = engine)


df2 = pd.read_csv('VectorisedNews.csv', sep=',', quotechar='|')

df2 = df2.loc[:,:'precovid']

print(df2.head())

df2.to_sql("VectorisedNews", con = engine)

#df_bit = pullObservationsToDF(parameters_prices)
#df_news = pullObservationsToDF(parameters_news)

#create_news(df_news)
#create_bitcoin(df_bit)



#Base.metadata.

Base.metadata.create_all(engine)