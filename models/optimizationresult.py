#!/usr/bin/python
import models
from models.base_model import BaseModel, Base
from sqlalchemy import String, Float, ForeignKey, JSON
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import relationship


class OptimizationResult(BaseModel, Base):
    __tablename__ = 'optimizationresult'
    # id = Column(Integer, primary_key=True)
    problem_id = Column(Integer, ForeignKey('problems.id'), nullable=False)
    users_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    opti_front_obj1 = Column(JSON, nullable=False)
    opti_front_obj2 = Column(JSON, nullable=False)
    opti_front_obj3 = Column(JSON, nullable=True)
    opti_para = Column(JSON, nullable=False)

    def __init__(self, *args, **kwargs):
        """initializes optimazationresult"""
        super().__init__(*args, **kwargs)
