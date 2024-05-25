#!/usr/bin/python3
"""This module handles all default RESTFul API actions for Users objects."""
from flask import jsonify, abort
from models import storage
from models.users import Users
from api.v1.views import app_views



@app_views.route('/users', methods=['GET'], strict_slashes=False)
def get_all_users():
    """Retrieves the list of all Problems objects."""
    users = [user.to_dict() for user in storage.all(Users).values()]
    return jsonify(users)


@app_views.route('/users/<user_id>', methods=['GET'], strict_slashes=True)
def get_user_by_id(user_id):
    """Retrieves a user object."""
    user = storage.get(Users, user_id)
    if user is None:
        abort(404)
    return jsonify(user.to_dict())
