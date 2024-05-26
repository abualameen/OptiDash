#!/usr/bin/python
import models
from models.base_model import BaseModel, Base
from sqlalchemy import String, Float, ForeignKey, JSON
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import relationship




class Problems(BaseModel, Base):
    __tablename__ = 'problems'
    # id = Column(Integer, primary_key=True)
    users_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    objective_functions = Column(JSON, nullable=False)
    decision_variables = Column(JSON, nullable=False)
    optimization_results = relationship('OptimizationResult', backref='problems', lazy=True)
    
    
    def __init__(self, *args, **kwargs):
        """initializes Problems"""
        super().__init__(*args, **kwargs)