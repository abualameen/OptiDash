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
from models.problems import Problems


class TestProblemsDocs(unittest.TestCase):
    """Tests to check the documentation and style of Problems class"""
    @classmethod
    def setUpClass(cls):
        """Set up for the doc tests"""
        cls.problems_f = inspect.getmembers(Problems, inspect.isfunction)

    def test_pep8_conformance_problems(self):
        """Test that models/problems.py conforms to PEP8."""
        pep8s = pep8.StyleGuide(quiet=True)
        result = pep8s.check_files(['models/problems.py'])
        self.assertEqual(result.total_errors, 0,
                         "Found code style errors (and warnings).")

    def test_pep8_conformance_test_problems(self):
        """Test that tests/test_models/
        test_problems.py conforms to PEP8.
        """
        pep8s = pep8.StyleGuide(quiet=True)
        result = pep8s.check_files(['tests/test_models/test_problems.py'])
        self.assertEqual(result.total_errors, 0,
                         "Found code style errors (and warnings).")

    def test_problems_module_docstring(self):
        """Test for the problems.py module docstring"""
        self.assertIsNot(Problems.__doc__, None,
                         "problems.py needs a docstring")
        self.assertTrue(len(Problems.__doc__) >= 1,
                        "problems.py needs a docstring")

    def test_problems_class_docstring(self):
        """Test for the Problems class docstring"""
        self.assertIsNot(Problems.__doc__, None,
                         "Problems class needs a docstring")
        self.assertTrue(len(Problems.__doc__) >= 1,
                        "Problems class needs a docstring")

    def test_problems_func_docstrings(self):
        """Test for the presence of docstrings in Problems methods"""
        for func in self.problems_f:
            self.assertIsNot(func[1].__doc__, None,
                             "{:s} method needs a docstring".format(func[0]))
            self.assertTrue(len(func[1].__doc__) >= 1,
                            "{:s} method needs a docstring".format(func[0]))


class TestProblems(unittest.TestCase):
    """Test the Problems class"""
    def test_is_subclass(self):
        """Test that Problems is a subclass of BaseModel"""
        problem = Problems()
        self.assertIsInstance(problem, BaseModel)
        self.assertTrue(hasattr(problem, "id"))
        self.assertTrue(hasattr(problem, "created_at"))
        self.assertTrue(hasattr(problem, "updated_at"))

    def test_users_id_attr(self):
        """
        Test that Problems has attribute
        users_id, and it's an empty string
        """
        problem = Problems()
        self.assertTrue(hasattr(problem, "users_id"))
        self.assertEqual(problem.users_id, None)

    def test_objective_functions_attr(self):
        """
        Test that Problems has attribute
        objective_functions, and it's an empty string
        """
        problem = Problems()
        self.assertTrue(hasattr(problem, "objective_functions"))
        self.assertEqual(problem.objective_functions, None)

    def test_decision_variables_attr(self):
        """ Test that Problems has attribute
            decision_variables, and it's an empty string
        """
        problem = Problems()
        self.assertTrue(hasattr(problem, "decision_variables"))
        self.assertEqual(problem.decision_variables, None)

    def test_optimization_results_attr(self):
        """
        Test that Problems has attribute
        optimization_results, and it's an empty string
        """
        problem = Problems()
        self.assertTrue(hasattr(problem, "optimization_results"))
        self.assertEqual(problem.optimization_results, None)

    def test_to_dict_creates_dict(self):
        """ Test to_dict method creates a
            dictionary with proper attrs
        """
        problem = Problems()
        new_d = problem.to_dict()
        self.assertEqual(type(new_d), dict)
        self.assertFalse("_sa_instance_state" in new_d)
        for attr in problem.__dict__:
            if attr is not "_sa_instance_state":
                self.assertTrue(attr in new_d)
        self.assertTrue("__class__" in new_d)

    def test_to_dict_values(self):
        """Test that values in dict returned from to_dict are correct"""
        t_format = "%Y-%m-%dT%H:%M:%S.%f"
        problem = Problems()
        new_d = problem.to_dict()
        self.assertEqual(new_d["__class__"], "Problems")
        self.assertEqual(type(new_d["created_at"]), str)
        self.assertEqual(type(new_d["updated_at"]), str)
        self.assertEqual(
            new_d["created_at"],
            problem.created_at.strftime(t_format))
        self.assertEqual(
            new_d["updated_at"],
            problem.updated_at.strftime(t_format))

    def test_str(self):
        """Test that the str method has the correct output"""
        problem = Problems()
        string = "[Problems] ({}) {}".format(problem.id, problem.__dict__)
        self.assertEqual(string, str(problem))


if __name__ == '__main__':
    unittest.main()
