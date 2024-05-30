#!/usr/bin/python3
"""
the API Blueprint
"""
from flask import Blueprint


app_views = Blueprint('app_views', __name__, url_prefix='/api/v1')

from api.v1.views.problems import *
from api.v1.views.users import *
from api.v1.views.optimizationresult import *
from api.v1.views.optimizationparam import *
from api.v1.views.index import *
