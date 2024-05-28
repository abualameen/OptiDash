#!/usr/bin/python3
"""
Contains the TestUsersDocs classes
"""
import inspect
import models
from models import users
from models.base_model import BaseModel, Base
import pep8
import unittest
Users = users.Users


class TestUsersDocs(unittest.TestCase):
    """Tests to check the documentation and style of Users class"""
    @classmethod
    def setUpClass(cls):
        """Set up for the doc tests"""
        cls.users_f = inspect.getmembers(Users, inspect.isfunction)

    def test_pep8_conformance_users(self):
        """Test that models/users.py conforms to PEP8."""
        pep8s = pep8.StyleGuide(quiet=True)
        result = pep8s.check_files(['models/users.py'])
        self.assertEqual(result.total_errors, 0,
                         "Found code style errors (and warnings).")

    def test_pep8_conformance_test_users(self):
        """Test that tests/test_models/test_users.py conforms to PEP8."""
        pep8s = pep8.StyleGuide(quiet=True)
        result = pep8s.check_files(['tests/test_models/test_users.py'])
        self.assertEqual(result.total_errors, 0,
                         "Found code style errors (and warnings).")

    def test_users_module_docstring(self):
        """Test for the users.py module docstring"""
        self.assertIsNot(users.__doc__, None,
                         "users.py needs a docstring")
        self.assertTrue(len(users.__doc__) >= 1,
                        "users.py needs a docstring")

    def test_users_class_docstring(self):
        """Test for the Users class docstring"""
        self.assertIsNot(Users.__doc__, None,
                         "Users class needs a docstring")
        self.assertTrue(len(Users.__doc__) >= 1,
                        "Users class needs a docstring")

    def test_users_func_docstrings(self):
        """Test for the presence of docstrings in Users methods"""
        for func in self.users_f:
            self.assertIsNot(func[1].__doc__, None,
                             "{:s} method needs a docstring".format(func[0]))
            self.assertTrue(len(func[1].__doc__) >= 1,
                            "{:s} method needs a docstring".format(func[0]))


class TestUsers(unittest.TestCase):
    """Test the Users class"""
    def test_is_subclass(self):
        """Test that Users is a subclass of BaseModel"""
        user = Users()
        self.assertIsInstance(user, BaseModel)
        self.assertTrue(hasattr(user, "id"))
        self.assertTrue(hasattr(user, "created_at"))
        self.assertTrue(hasattr(user, "updated_at"))

    def test_username_attr(self):
        """Test that Users has attribute username, and it's an empty string"""
        user = Users()
        self.assertTrue(hasattr(user, "username"))
        if models.storage_t == 'db':
            self.assertEqual(user.username, None)
        else:
            self.assertEqual(user.username, "")

    def test_email_attr(self):
        """Test that Users has attribute email, and it's an empty string"""
        user = Users()
        self.assertTrue(hasattr(user, "email"))
        if models.storage_t == 'db':
            self.assertEqual(user.email, None)
        else:
            self.assertEqual(user.email, "")

    def test_password_attr(self):
        """Test that Users has attribute password, and it's an empty string"""
        user = Users()
        self.assertTrue(hasattr(user, "password"))
        if models.storage_t == 'db':
            self.assertEqual(user.password, None)
        else:
            self.assertEqual(user.password, "")

    def test_problems_attr(self):
        """Test that Users has attribute problems, and it's an empty list"""
        user = Users()
        self.assertTrue(hasattr(user, "problems"))
        self.assertEqual(user.problems, [])

    def test_optimization_results_attr(self):
        """Test that Users has attribute"
           optimization_results, and it's an empty list
        """
        user = Users()
        self.assertTrue(hasattr(user, "optimization_results"))
        self.assertEqual(user.optimization_results, [])

    def test_to_dict_creates_dict(self):
        """Test to_dict method creates a dictionary with proper attrs"""
        u = Users()
        new_d = u.to_dict()
        self.assertEqual(type(new_d), dict)
        self.assertFalse("_sa_instance_state" in new_d)
        for attr in u.__dict__:
            if attr is not "_sa_instance_state":
                self.assertTrue(attr in new_d)
        self.assertTrue("__class__" in new_d)

    def test_to_dict_values(self):
        """Test that values in dict returned from to_dict are correct"""
        t_format = "%Y-%m-%dT%H:%M:%S.%f"
        u = Users()
        new_d = u.to_dict()
        self.assertEqual(new_d["__class__"], "Users")
        self.assertEqual(type(new_d["created_at"]), str)
        self.assertEqual(type(new_d["updated_at"]), str)
        self.assertEqual(new_d["created_at"], u.created_at.strftime(t_format))
        self.assertEqual(new_d["updated_at"], u.updated_at.strftime(t_format))

    def test_str(self):
        """Test that the str method has the correct output"""
        user = Users()
        string = "[Users] ({}) {}".format(user.id, user.__dict__)
        self.assertEqual(string, str(user))
