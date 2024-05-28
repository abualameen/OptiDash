#!/usr/bin/python3

"""
initializing the storage model

"""

from models.engine.db_storage import DBStorage
storage = DBStorage()
storage.reload()
