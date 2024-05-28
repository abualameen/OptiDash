import unittest
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.security import generate_password_hash
from web_dynamic.OptiDash import app, login
from models.users import Users  # Import your Users model
from models import storage
from werkzeug.security import generate_password_hash, check_password_hash
from unittest.mock import patch, MagicMock
from flask import Flask, request, url_for, redirect, flash, render_template
from flask_login import current_user
from web_dynamic.OptiDash import app, storage
from models.users import Users
from flask_login import current_user
from flask_login import login_user, UserMixin, logout_user


class TestFlaskRoutes(unittest.TestCase):
    """ the is the test TestFlask """
    def setUp(self):
        """ Set up the Flask app for testing """
        app.testing = True
        self.app = app.test_client()

    def test_home_route_get(self):
        """ Test the GET request to the home route """
        response = self.app.get('/home')
        # Assuming home route returns HTTP status code 200
        self.assertEqual(response.status_code, 302)

    def test_home_route_post(self):
        """ Test the POST request to the home route """
        response = self.app.post(
            '/home', json={'tableData1': [],
                           'tableData': [], 'tableData2': []})
        # Assuming home route returns HTTP status code 200
        self.assertEqual(response.status_code, 302)

    def test_about_us_route(self):
        """ Test the GET request to the about us route """
        response = self.app.get('/about_us')
        # Assuming about us route returns HTTP status code 200
        self.assertEqual(response.status_code, 200)

    def test_contact_us_route(self):
        """ Test the GET request to the contact us route """
        response = self.app.get('/contact_us')
        # Assuming contact us route returns HTTP status code 200
        self.assertEqual(response.status_code, 200)

    def test_index_route_post(self):
        # Define test data
        test_username = "test_user"
        test_password = "test_password"
        test_email = "test@example.com"

        # Make a POST request to the index route
        response = self.app.post('/', data=dict(
            username=test_username,
            password=test_password,
            email=test_email
        ), follow_redirects=False)
        new_user = storage.get_by_username(Users, test_username)
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(new_user)
        self.assertEqual(new_user.username, test_username)
        self.assertEqual(new_user.email, test_email)
        self.assertTrue(
            check_password_hash(new_user.password,
                                test_password))


class TestLoginRoute(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch('web_dynamic.OptiDash.storage')
    def test_login_successful_redirect(self, mock_storage):
        # Mock form data
        form_data = {'username': 'test_user',
                     'password': 'test_password'}
        # Create a real user instance
        mock_user = Users(
            username='test_user',
            email='test_user@example.com',
            password='hashed_password')
        mock_storage.get_by_username.return_value = mock_user

        # Mock password hash check
        with patch('web_dynamic.OptiDash.check_password_hash',
                   return_value=True):
            response = self.app.post('/login', data=form_data)

        # Check that the user is redirected to the home page
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, '/home')

    @patch('web_dynamic.OptiDash.storage')
    def test_login_invalid_credentials(self, mock_storage):
        # Mock form data
        form_data = {'username': 'test_user',
                     'password': 'wrong_password'}
        # Create a real user instance
        mock_user = Users(
            username='test_user',
            email='test_user@example.com',
            password='hashed_password')
        mock_storage.get_by_username.return_value = mock_user

        # Mock password hash check
        with patch(
             'web_dynamic.OptiDash.check_password_hash',
             return_value=False):
            response = self.app.post('/login', data=form_data)

        # Check that the user is redirected to the login
        # page with an error message
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid username or password', response.data)

    def test_login_get_request(self):
        response = self.app.get('/login')
        # Check that the login page is rendered for GET request
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<form method="POST" action="/login">',
                      response.data)


class MockUser(UserMixin):
    def __init__(self, id):
        self.id = int(id)


class TestLogoutRoute(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch(
            'web_dynamic.OptiDash.logout_user',
            side_effect=logout_user)
    def test_logout(self, mock_logout_user):
        with app.app_context():
            with self.app as c:
                with c.session_transaction() as sess:
                    # Mock login_user to simulate a logged-in user
                    # Use an integer ID instead of 'test_user_id'
                    test_user = MockUser(3)
                    sess['_user_id'] = test_user.id
                    sess['_fresh'] = True
                # Ensure the current user is logged in
                with c.get('/home'):
                    self.assertTrue(current_user.is_authenticated)
                # Call the logout route
                response = self.app.get('/logout')
                # Check that logout_user was called
                mock_logout_user.assert_called_once()
                # Check that the response is a redirect to the login page
                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.location, '/login')


if __name__ == '__main__':
    unittest.main()
