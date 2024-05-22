#!/usr/bin/python
import models
from models.base_model import BaseModel, Base
from sqlalchemy import String, Float, ForeignKey, JSON
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import relationship


class Users(BaseModel, Base):
    __tablename__ = 'users'
    username = Column(String(250), unique=True, nullable=False)
    password = Column(String(250), nullable=False)
    
    def __init__(self, *args, **kwargs):
        """initializes criteria"""
        super().__init__(*args, **kwargs)