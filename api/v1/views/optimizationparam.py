#!/usr/bin/python3
"""This module handles all default RESTFul
API actions for Users objects.
"""
from flask import jsonify, abort
from models import storage
from models.optimizationparameters import OptimizationParameters
from api.v1.views import app_views


@app_views.route(
        '/optimizationparameters',
        methods=['GET'], strict_slashes=False)
def get_all_optimizationparam():
    """ Retrieves the list of all
        Optimization PARAM objects.
    """
    optimizationparams = [
        optimizationparameters.to_dict()
        for optimizationparameters in
        storage.all(OptimizationParameters).values()]
    return jsonify(optimizationparams)


@app_views.route(
        '/optimizationparameters/<optimizationpar_id>',
        methods=['GET'], strict_slashes=False)
def get_optimizationparam_by_id(optimizationpar_id):
    """Retrieves a optimization result object."""
    optimizationparameters = storage.get(
        OptimizationParameters,
        optimizationpar_id)
    if optimizationparameters is None:
        abort(404)
    return jsonify(optimizationparameters.to_dict())


@app_views.route(
        '/problems/<int:problem_id>/optimization_params',
        methods=['GET'],
        strict_slashes=False)
def get_optimization_param_by_problem_id(problem_id):
    """Retrieves all optimization results
       associated with a specific problem
    """
    params = storage.get_optimization_param_by_problem_id(
        problem_id)
    print('param', params)
    if not params:
        abort(404, description="Param not found or no optimization"
              "param associated with this problem.")
    return jsonify([result.to_dict() for result in params])
