#!/usr/bin/python3
"""
the API Blueprint
"""
from flask import Blueprint
from api.v1.views.problems import *
from api.v1.views.users import *
from api.v1.views.optimizationresult import *

app_views = Blueprint('app_views', __name__, url_prefix='/api/v1')
