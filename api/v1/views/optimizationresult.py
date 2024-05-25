#!/usr/bin/python3
"""This module handles all default RESTFul API actions for Users objects."""
from flask import jsonify, abort
from models import storage
from models.optimizationresult import OptimizationResult
from api.v1.views import app_views



@app_views.route('/optimizationresults', methods=['GET'], strict_slashes=False)
def get_all_optimizationresult():
    """Retrieves the list of all Optimization Results objects."""
    optimizationresults = [optimizationresult.to_dict() for optimizationresult in storage.all(OptimizationResult).values()]
    return jsonify(optimizationresults)


@app_views.route('/optimizationresults/<optimizationresult_id>', methods=['GET'], strict_slashes=False)
def get_optimizationresult_by_id(optimizationresult_id):
    """Retrieves a optimization result object."""
    optimizationresult = storage.get(OptimizationResult, optimizationresult_id)
    if optimizationresult is None:
        abort(404)
    return jsonify(optimizationresult.to_dict())


@app_views.route('/problems/<int:problem_id>/optimization_results', methods=['GET'], strict_slashes=False)
def get_optimization_results_by_problem_id(problem_id):
    """Retrieves all optimization results associated with a specific problem"""
    problem_results = storage.get_optimization_results_by_problem_id(problem_id)
    if not problem_results:
        abort(404, description="Problem not found or no optimization results associated with this problem.")
    return jsonify([result.to_dict() for result in problem_results])







