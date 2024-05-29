#!/usr/bin/python
""" this is the problem module"""
import models
from models.base_model import BaseModel, Base
from sqlalchemy import String, Float, ForeignKey, JSON
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import relationship


class Problems(BaseModel, Base):
    """ the is the Problem class"""
    __tablename__ = 'problems'
    # id = Column(Integer, primary_key=True)
    users_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    objective_functions = Column(JSON, nullable=False)
    decision_variables = Column(JSON, nullable=False)
    optimization_results = relationship(
        'OptimizationResult', backref='problems',
        lazy=True)
    optimization_parameters = relationship(
        'OptimizationParameters',
        backref='problems', lazy=True,
        overlaps="opt_params,optimization_parameters")


    def __init__(self, *args, **kwargs):
        """initializes Problems"""
        super().__init__(*args, **kwargs)
