#!/usr/bin/python3
""" module for index view """
from models.optimizationresult import OptimizationResult
from models.users import Users
from models.optimizationparameters import OptimizationParameters
from models.problems import Problems
from api.v1.views import app_views
from flask import jsonify
from models import storage


@app_views.route('/status', methods=['GET'], strict_slashes=False)
def status():
    """ Status of API """
    return jsonify({"status": "OK"})


@app_views.route('/eachobj', methods=['GET'], strict_slashes=False)
def number_objects():
    """ Retrieves the number of each objects by type """
    classes = [Users, OptimizationResult, OptimizationParameters, Problems]
    names = ["users", "optimizationresult", "optimizationparameters",
             "problems"]

    num_objs = {}
    for i in range(len(classes)):
        num_objs[names[i]] = storage.count(classes[i])
    return jsonify(num_objs)
