#!/usr/bin/python
""" this is the paremeters module"""
import models
from models.base_model import BaseModel, Base
from sqlalchemy import String, Float, ForeignKey, JSON
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import relationship


class OptimizationParameters(BaseModel, Base):
    """This is the OptimizationParameter class"""
    __tablename__ = 'optimization_parameters'
    problem_id = Column(Integer, ForeignKey('problems.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    pop_size = Column(Integer, nullable=False)
    iteration_no = Column(Integer, nullable=False)
    crossover_rate = Column(Integer, nullable=False)
    crossover_coef = Column(Integer, nullable=False)
    mutation_rate = Column(Integer, nullable=False)
    mutation_coef = Column(Integer, nullable=False)
    problem = relationship(
        'Problems', backref='opti_params',
        overlaps="optimization_parameters,problems")
    user = relationship(
        'Users', backref='opti_params',
        overlaps="optimization_parameters,user_param")

    def __init__(self, *args, **kwargs):
        """initializes OptimizationParameter"""
        super().__init__(*args, **kwargs)
