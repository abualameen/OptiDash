#!/usr/bin/python3
"""This module handles all default RESTFul API actions for Problems objects."""
from flask import jsonify, abort
from models import storage
from models.problems import Problems
from api.v1.views import app_views



@app_views.route('/problems', methods=['GET'], strict_slashes=False)
def get_all_problems():
    """Retrieves the list of all Problems objects."""
    problems = [problem.to_dict() for problem in storage.all(Problems).values()]
    return jsonify(problems)


@app_views.route('/problems/<problem_id>', methods=['GET'], strict_slashes=True)
def get_problem_by_id(problem_id):
    """Retrieves a problems object."""
    problem = storage.get(Problems, problem_id)
    if problem is None:
        abort(404)
    return jsonify(problem.to_dict())

@app_views.route('/users/<int:users_id>/problems', methods=['GET'], strict_slashes=False)
def get_problems_by_user_id(users_id):
    """Retrieves all problems associated with a specific user"""
    user_problems = storage.get_problems_by_user_id(users_id)
    if not user_problems:
        abort(404, description="User not found or no problems associated with this user.")
    return jsonify([problem.to_dict() for problem in user_problems])









# @app_views.route('/problems/<problem_id>/<users_id>', methods=['GET'], strict_slashes=True)
# def get_problem_by_users_id(users_id):
#     """Retrieves a PROBLEMS object."""
#     problem = storage.get_by_users_id(Problems, users_id)
#     print('dfd', problem)
#     if problem is None:
#         abort(404)
#     return jsonify(problem.to_dict())

# @app_views.route('/problems/user/<users_id>', methods=['GET'], strict_slashes=False)
# def get_problems_by_user_id(users_id):
#     """Retrieves all problems associated with a specific user"""
#     user_problems = storage.get_problems_by_user_id(users_id)
#     if not user_problems:
#         abort(404)
#     return jsonify([problem.to_dict() for problem in user_problems])