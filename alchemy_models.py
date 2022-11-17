# coding: utf-8
from sqlalchemy import BigInteger, Column, Float, MetaData, Table, Integer, Date, String, Text, ForeignKey, create_engine
from sqlalchemy.orm import mapper, relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = MetaData()

engine = create_engine("sqlite:///C:\\Users\\temic\\PycharmProjects\\database\\identifier.sqlite", echo=False)


t_Bitcoin_Data = Table(
    'Bitcoin_Data', metadata,
    Column('index', BigInteger, index=True, ),
    Column('volume', Float),
    Column('market_cap', Float),
    Column('timestamp', Float,primary_key=True),
    Column('high', Float),
    Column('low', Float,primary_key=True),
    Column('close', Float),
    Column('open', Float)
)


t_News_Data = Table(
    'News_Data', metadata,
    Column('index', BigInteger, index=True),
    Column('headline', Text),
    Column('publisher', Text),
    Column('title', Text, primary_key=True),
    Column('url', Text),
    Column('timestamp', Float)
)


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

