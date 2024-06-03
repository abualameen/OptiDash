# OptiDash
![OptiDash](https://github.com/abualameen/OptiDash/assets/75878845/b0dd9246-2684-4594-a679-f98a0c290072)

OptiDash is an online multi-objective optimization problem solver designed for engineers, researchers, educators in the field of optimization, students and any one who wants to make informed decision. It is based on the Non Dominated Sorted Genetic Algorithm II,
which is able to solve two and three objective optimization problems in any field. The Algorithm was developed by Professor Debs and his team of researchers.
The output of OptiDash is the optimal pareto front data, the optimal solution data and the dynamic plotting of the pareto optimal front as the data is being generated from the backend. process.
With OptiDash obtaining the pareto optimal solutions is made easy, thereby enhancing productivity and accuracy.


## Inspiration

The inspiration behind developing OptiDash stemmed from my experience as a mechanical engineer specializing in solving engineering optimization problems. Often, in the field we encounter the challenge solving multi-objective optimization problem of which most is solve in codes, which is not very fimilair and easy to use by other non-programmers or research team mate that are not able to code or programme.

Recognizing the need for a more efficient and user-friendly solution, I embarked on developing OptiDash. This project streamlines the decision-making process by allowing users to input their NSGA II parameters, objective functions and search space of each decision variable and swiftly obtain the Optimal Pareto Front of the problem. By automating this process, OptiDash enhances productivity and accuracy, empowering engineers, researchers, and students alike to make informed decisions effectively.

## Features

- **Problem Matrix Generation**: Easily create Problem matrices dynamically by specifying the number of decision variable in the problem.
- **Optimal Pareto **: Obtains the Optimal Pareto front of any problem 
- **Selection of any combination of objective**: Users are able to select any type of problem objective.
- **Optimal Solution**: Obtains the Optimal solution.
- **Interactive User Interface**: Intuitive and user-friendly interface for inputting data and viewing results and real time dynamic generation of the Pareto Front displayed to users.
- **Neatly Presented Optimal Pareto Front and Optimal solution in a table**: The Optimal Pareto Front and Optimal solution in a table are neatly presented to users in table form  .
- **API Integration**: Robust API for interacting with the application's database, providing efficient data retrieval and management.

## How it Works

1. **Create Problem Matrix**: Specify the number of decision variable in your problem to generate the Problem matrix.
2. **Input Objective functions and Search space**: Enter Objective function and decision variable search space
3. **Input NSGA II Parameters**: Enter the population size, number of generation, crossover rate, cross over coeficient, mutation rate, mutation coeficient.
4. **Click the submit button**: By clicking the submit button the problem input are sent to the backend for processsing, and almost immediately the dynamic searching and ploting of the optimal pareto front in the frontend begins.
5. **Dynamic Display of the Pareto Optimal Front**: The Pareto Optimal front is displayed for users view
6. **Tabular Result Output**: At the end of ploting the optimal pareto front, the result able is display to users utilization of the data within

## Usage

To use OptiDash, follow these steps:

1. Visit the [OptiDash website](https://13.50.232.164).
2. Specify the number of decision variables in your problem
3. Enter all input fields accordingly
4. Click on the "Submit" button.
5. The dynamic realtime ploting of the pareto front, tabular(optimal solutions and pareto optimal fitness values) would be outputed for users utilization.

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript, Plotly
- **Backend**: Python, Flask, Flask Socket.io
- **Database**: MySQL (for storing user data)
- **Deployment**: Nginx. Gunicorn, Supervisor

## API

In addition to the user interface, OptiDash features a robust API that interacts seamlessly with the app's database. This API serves as the backbone of the application, enabling efficient retrieval of data from the result table, criteria table, and alternative table.

### Endpoints

1. **Index Endpoints**:
   - `/api/v1/status`: Checks for API status.
   - `/api/v1/eachobj`: Gets the total number of objects in each class.

2. **Optimization Parameter Endpoints**:
   - `/api/v1/optimizationparameters`: Retrieve all optimization paremeter in it table.
   - `/api/v1/optimizationparameters/<optimizationpar_id>`: Access specific optimization parameters data by ID.
   - `/api/v1/problems/<int:problem_id>/optimization_params`: Retrieves all optimization results associated with a specific problem

3. **Optimization Results Endpoints**:
   - `/api/v1/optimizationresults`: Retrieves the list of all Optimization Results objects.
   - `/api/v1/optimizationresults/<optimizationresult_id>`: Retrieves a optimization result object based on ID
   - `/problems/<int:problem_id>/optimization_results`: Retrieves all optimization results associated with a specific problem
4. **Problems Endpoints**:
   - `/api/v1/problems/`: Retrieve all problem data.
   - `/api/v1/problems/<problem_id>`: Access specific problem data by ID.
   - `/api/v1/users/<int:users_id>/problems`: Retrieves all problems associated with a specific user

5. **Users Endpoints**:
   - `/api/v1/users/`: Retrieve all problem data.
   - `/api/v1/users/<user_id>'`: Access specific user data by ID.

These endpoints facilitate efficient data retrieval, allowing users to access comprehensive datasets or specific records as needed. Whether querying users, optimization paremeter, optimization results, or problem,the API endpoints provide a streamlined approach to data access, enhancing the functionality and usability of OptiDash.

## Contributing

Contributions to OptiDash are welcome! If you have any ideas for improvements, new features, or bug fixes, feel free to open an issue or submit a pull request.

## Flow Chart

![OptiDash flow chart](https://github.com/abualameen/ChoiceCrafter/assets/75878845/ef89830c-a1f6-40fa-8376-25debf688cf0)

## License

OptiDash is licensed under the [MIT License](LICENSE).
