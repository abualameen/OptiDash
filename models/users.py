#!/usr/bin/python
import models
from models.base_model import BaseModel, Base
from sqlalchemy import String, Float, ForeignKey, JSON
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import relationship


class Users(BaseModel, Base):
    __tablename__ = 'users'
    username = Column(String(250), unique=True, nullable=False)
    email = Column(String(250), nullable=False, unique=True)
    password = Column(String(250), nullable=False)
    problems = relationship('Problems', backref='user', lazy=True)
    optimization_results = relationship('OptimizationResult', backref='user', lazy=True)
    
    
    def __init__(self, *args, **kwargs):
        """initializes criteria"""
        super().__init__(*args, **kwargs)

    def get_id(self):
        return str(self.id)

    @property
    def is_active(self):
        return True

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False
