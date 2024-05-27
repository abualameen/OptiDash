#!/usr/bin/python
import models
from models.base_model import BaseModel, Base
from sqlalchemy import String, Float, ForeignKey, JSON
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import relationship
import unittest
from datetime import datetime
import inspect
import pep8
from models.optimizationresult import OptimizationResult



class TestOptimizationResult(unittest.TestCase):
    """Test the OptimizationResult class"""
    def test_is_subclass(self):
        """Test that OptimizationResult is a subclass of BaseModel"""
        opti_result = OptimizationResult()
        self.assertIsInstance(opti_result, BaseModel)
        self.assertTrue(hasattr(opti_result, "id"))
        self.assertTrue(hasattr(opti_result, "created_at"))
        self.assertTrue(hasattr(opti_result, "updated_at"))

    def test_problem_id_attr(self):
        """Test that OptimizationResult has attribute problem_id, and it's None"""
        opti_result = OptimizationResult()
        self.assertTrue(hasattr(opti_result, "problem_id"))
        self.assertIsNone(opti_result.problem_id)

    def test_users_id_attr(self):
        """Test that OptimizationResult has attribute users_id, and it's None"""
        opti_result = OptimizationResult()
        self.assertTrue(hasattr(opti_result, "users_id"))
        self.assertIsNone(opti_result.users_id)

    def test_opti_front_obj1_attr(self):
        """Test that OptimizationResult has attribute opti_front_obj1, and it's an empty list"""
        opti_result = OptimizationResult()
        self.assertTrue(hasattr(opti_result, "opti_front_obj1"))
        self.assertEqual(opti_result.opti_front_obj1, [])

    def test_opti_front_obj2_attr(self):
        """Test that OptimizationResult has attribute opti_front_obj2, and it's an empty list"""
        opti_result = OptimizationResult()
        self.assertTrue(hasattr(opti_result, "opti_front_obj2"))
        self.assertEqual(opti_result.opti_front_obj2, [])

    def test_opti_front_obj3_attr(self):
        """Test that OptimizationResult has attribute opti_front_obj3, and it's None"""
        opti_result = OptimizationResult()
        self.assertTrue(hasattr(opti_result, "opti_front_obj3"))
        self.assertIsNone(opti_result.opti_front_obj3)

    def test_opti_para_attr(self):
        """Test that OptimizationResult has attribute opti_para, and it's an empty list"""
        opti_result = OptimizationResult()
        self.assertTrue(hasattr(opti_result, "opti_para"))
        self.assertEqual(opti_result.opti_para, [])

    def test_to_dict_creates_dict(self):
        """Test to_dict method creates a dictionary with proper attrs"""
        o = OptimizationResult()
        new_d = o.to_dict()
        self.assertEqual(type(new_d), dict)
        self.assertFalse("_sa_instance_state" in new_d)
        for attr in o.__dict__:
            if attr is not "_sa_instance_state":
                self.assertTrue(attr in new_d)
        self.assertTrue("__class__" in new_d)

    def test_to_dict_values(self):
        """Test that values in dict returned from to_dict are correct"""
        t_format = "%Y-%m-%dT%H:%M:%S.%f"
        o = OptimizationResult()
        new_d = o.to_dict()
        self.assertEqual(new_d["__class__"], "OptimizationResult")
        self.assertEqual(type(new_d["created_at"]), str)
        self.assertEqual(type(new_d["updated_at"]), str)
        self.assertEqual(new_d["created_at"], o.created_at.strftime(t_format))
        self.assertEqual(new_d["updated_at"], o.updated_at.strftime(t_format))

    def test_str(self):
        """Test that the str method has the correct output"""
        opti_result = OptimizationResult()
        string = "[OptimizationResult] ({}) {}".format(opti_result.id, opti_result.__dict__)
        self.assertEqual(string, str(opti_result))



class TestOptimizationResultDocs(unittest.TestCase):
    """Tests to check the documentation and style of OptimizationResult class"""
    @classmethod
    def setUpClass(cls):
        """Set up for the doc tests"""
        cls.opti_result_f = inspect.getmembers(OptimizationResult, inspect.isfunction)

    def test_pep8_conformance_optimizationresult(self):
        """Test that models/optimizationresult.py conforms to PEP8."""
        pep8s = pep8.StyleGuide(quiet=True)
        result = pep8s.check_files(['models/optimizationresult.py'])
        self.assertEqual(result.total_errors, 0,
                         "Found code style errors (and warnings).")

    def test_pep8_conformance_test_optimizationresult(self):
        """Test that tests/test_models/test_optimizationresult.py conforms to PEP8."""
        pep8s = pep8.StyleGuide(quiet=True)
        result = pep8s.check_files(['tests/test_models/test_optimizationresult.py'])
        self.assertEqual(result.total_errors, 0,
                         "Found code style errors (and warnings).")

    def test_optimizationresult_module_docstring(self):
        """Test for the optimizationresult.py module docstring"""
        self.assertIsNot(OptimizationResult.__doc__, None,
                         "optimizationresult.py needs a docstring")
        self.assertTrue(len(OptimizationResult.__doc__) >= 1,
                        "optimizationresult.py needs a docstring")

    def test_optimizationresult_class_docstring(self):
        """Test for the OptimizationResult class docstring"""
        self.assertIsNot(OptimizationResult.__doc__, None,
                         "OptimizationResult class needs a docstring")
        self.assertTrue(len(OptimizationResult.__doc__) >= 1,
                        "OptimizationResult class needs a docstring")

    def test_optimizationresult_func_docstrings(self):
        """Test for the presence of docstrings in OptimizationResult methods"""
        for func in self.opti_result_f:
            self.assertIsNot(func[1].__doc__, None,
                             "{:s} method needs a docstring".format(func[0]))
            self.assertTrue(len(func[1].__doc__) >= 1,
                            "{:s} method needs a docstring".format(func[0]))
