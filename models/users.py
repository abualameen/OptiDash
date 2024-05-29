#!/usr/bin/python
""" this is the users module """
import models
from models.base_model import BaseModel, Base
from sqlalchemy import String, Float, ForeignKey, JSON
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import relationship


class Users(BaseModel, Base):
    """ this is the userS Class """
    __tablename__ = 'users'
    username = Column(String(250), unique=True, nullable=False)
    email = Column(String(250), nullable=False, unique=True)
    password = Column(String(250), nullable=False)
    problems = relationship('Problems', backref='user', lazy=True)
    optimization_results = relationship(
        'OptimizationResult', backref='user',
        lazy=True)
    optimization_parameters = relationship(
        'OptimizationParameters', backref='user_param',
        lazy=True, overlaps="user_param,user")

    def __init__(self, *args, **kwargs):
        """initializes criteria"""
        super().__init__(*args, **kwargs)

    def get_id(self):
        """ gets id"""
        return str(self.id)

    @property
    def is_active(self):
        """ checks is active"""
        return True

    @property
    def is_authenticated(self):
        """ checks auth """
        return True

    @property
    def is_anonymous(self):
        """ checks ano"""
        return False
