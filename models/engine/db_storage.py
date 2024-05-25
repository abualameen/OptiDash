#!/usr/bin/python3
"""
Contains the class DBStorage
"""
import models
from models.users import Users
from models.problems import Problems
from models.optimizationresult import OptimizationResult
from models.base_model import BaseModel, Base
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from os import getenv


classes = {"Users": Users, "OptimizationResult": OptimizationResult, "Problems": Problems}


class DBStorage:
    __engine = None
    __session = None

    def __init__(self):
        OD_MYSQL_USER = getenv('OD_MYSQL_USER')
        OD_MYSQL_PWD = getenv('OD_MYSQL_PWD')
        OD_MYSQL_HOST = getenv('OD_MYSQL_HOST')
        OD_MYSQL_DB = getenv('OD_MYSQL_DB')
        self.__engine = create_engine('mysql+mysqldb://{}:{}@{}/{}'.
                                      format(OD_MYSQL_USER,
                                             OD_MYSQL_PWD,
                                             OD_MYSQL_HOST,
                                             OD_MYSQL_DB))

    def all(self, cls=None):
        """query on the current database session"""
        new_dict = {}
        for clss in classes:
            if cls is None or cls is classes[clss] or cls is clss:
                objs = self.__session.query(classes[clss]).all()
                for obj in objs:
                    key = obj.__class__.__name__ + '.' + str(obj.id)
                    new_dict[key] = obj
        return (new_dict)

    def new(self, obj):
        """add the object to the current database session"""
        self.__session.add(obj)

    def save(self):
        """commit all changes of the current database session"""
        self.__session.commit()

    def delete(self, obj=None):
        """delete from the current database session obj if not None"""
        if obj is not None:
            self.__session.delete(obj)

    def reload(self):
        """reloads data from the database"""
        Base.metadata.create_all(self.__engine)
        sess_factory = sessionmaker(bind=self.__engine, expire_on_commit=False)
        Session = scoped_session(sess_factory)
        self.__session = Session

    def close(self):
        """call remove() method on the private session attribute"""
        self.__session.remove()

    def get(self, cls, id):
        """
        Returns the object based on the class name and its ID, or
        None if not found
        """
        id = int(id)
        if cls not in classes.values():
            return None
        all_cls = models.storage.all(cls)
        for value in all_cls.values():
            if (value.id == id):
                return value
        return None


    def get_by_username(self, cls, username):
        """
        Returns the user object based on the class name and its username, or
        None if not found
        """
        if cls not in classes.values():
            return None
        all_cls = models.storage.all(cls)
        for value in all_cls.values():
            if (value.username == username):
                return value
        return None

    def get_problems_by_user_id(self, users_id):
        """Returns all problems associated with a specific user_id"""
        all_problems = self.all(Problems)
        user_problems = [problem for problem in all_problems.values() if problem.users_id == int(users_id)]
        return user_problems

    def get_optimization_results_by_problem_id(self, problem_id):
        """Returns all optimization results associated with a specific problem_id"""
        all_results = self.all(OptimizationResult)
        problem_results = [result for result in all_results.values() if result.problem_id == int(problem_id)]
        return problem_results


    # def get_by_users_id(self, cls, users_id):
    #     """
    #     Returns the user object based on the class name and its username, or
    #     None if not found
    #     """
    #     if cls not in classes.values():
    #         return None
    #     all_cls = models.storage.all(cls)
    #     for value in all_cls.values():
    #         if (value.users_id == users_id):
    #             return value
    #     return None


    # def get_prob_user(self, cls, users_id):
    #     """Returns objects based on class, user_id, and id"""
    #     if cls not in classes.values():
    #         return None
    #     all_cls = models.storage.all(cls)
    #     for value in all_cls.values():
    #         if value.users_id == users_id:
    #             return value
    #     return None
        
    
    # def get_problems_by_user_id(self, users_id):
    #     """Returns all problems associated with a specific user_id"""
    #     all_problems = self.all(Problems)
    #     user_problems = [problem for problem in all_problems.values() if problem.users_id == users_id]
    #     return user_problems

    def count(self, cls=None):
        """
        count the number of objects in storage
        """
        all_class = classes.values()

        if not cls:
            count = 0
            for clas in all_class:
                count += len(models.storage.all(clas).values())
        else:
            count = len(models.storage.all(cls).values())

        return count