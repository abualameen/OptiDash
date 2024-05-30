#!/usr/bin/python3
"""
this the choice crafter flask app
"""


from models import storage
from models.users import Users
from models.problems import Problems
from models.optimizationresult import OptimizationResult
from models.optimizationparameters import OptimizationParameters
from flask import url_for, redirect, flash
from flask_login import LoginManager, UserMixin, login_user
from flask_login import logout_user, login_required, current_user
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import time
import copy
import math
import random
import numpy as np
from cmath import inf, nan
from werkzeug.security import generate_password_hash, check_password_hash
import json
from threading import Lock
import uuid

# Global dictionary to store task results
task_results = {}
task_results_lock = Lock()
app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")
app.config["SECRET_KEY"] = "abulyaqs@gmail.com"

login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def loader_user(id):
    return storage.get(Users, int(id))


@login_manager.unauthorized_handler
def unauthorized():
    flash('Login is required to access this page', 'warning')
    return redirect(url_for('login'))


@app.route('/', methods=['GET', 'POST'], strict_slashes=False)
def index():
    return render_template('index.html')


@app.route('/reg', methods=['GET', 'POST'], strict_slashes=False)
def reg():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        email = request.form.get("email")

        # existing_user = Users.query.filter_by(username=username).first()
        existing_user = storage.get_by_username(Users, username)
        if existing_user:
            flash(
                'Username already taken. Please choose a different username.',
                'danger'
                )
            return render_template('index.html')
        new_user = Users(
            username=username,
            email=email, password=generate_password_hash(password))
        storage.new(new_user)
        storage.save()
        return redirect(url_for("login"))
    return render_template('reg.html')


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = username = request.form.get("username")
        password = password = request.form.get("password")
        user = storage.get_by_username(Users, username)
        if user is not None and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        elif user is None:
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.teardown_appcontext
def close_db(error):
    """ Remove the current SQLAlchemy Session """
    storage.close()


@app.route('/home', methods=['GET', 'POST'], strict_slashes=False)
@login_required
def home():
    """ this is a index function for the index route """
    if request.method == "POST":
        try:
            data = request.get_json()
            import re

            def tokenize(expression):
                """ this tokenize """
                pattern = re.compile(r'(\b\w+\b|[+\-*/()**])')
                tokens = pattern.findall(expression)
                return tokens

            def flatten_expressions(objective_function_list):
                """ this flattens the """
                flattened = []
                for expr in objective_function_list:
                    flattened.extend(tokenize(expr))
                return flattened

            # Example input
            objective_function_list = data.get('tableData1')
            decision_variables = [
                sublist[0] for sublist in data.get('tableData')]
            # Flatten the expressions
            flattened = flatten_expressions(objective_function_list)
            for element in decision_variables:
                if element not in flattened:
                    flash(
                        "Decision variables error in the objective function",
                        "danger"
                        )
                    return render_template('home.html')
            table_data = data.get('tableData')
            table_data1 = data.get('tableData1')
            table_data2 = data.get('tableData2')
            table_data3 = data.get('tableData3')

            def con_str_int1(table_dat):
                """Convert strings in the list
                to integers or floats.
                """
                table_dataa1 = []
                for item in table_dat:
                    try:
                        number = float(item)
                        if number.is_integer():
                            number = int(number)
                        table_dataa1.append(number)
                    except ValueError:
                        table_dataa1.append(item)
                return table_dataa1
            table_data3 = con_str_int1(table_data3)
            print('table_dat3', table_data3)
            task_id = str(uuid.uuid4())
            if len(table_data1) == 2:
                results = socketio.start_background_task(
                    target=nsga2, table_data=table_data,
                    table_data1=table_data1, table_data2=table_data2,
                    table_data3=table_data3, task_id=task_id)
            else:
                results = socketio.start_background_task(
                    target=nsgaa2, table_data=table_data,
                    table_data1=table_data1, table_data2=table_data2,
                    table_data3=table_data3, task_id=task_id)
            # Wait for the task to complete
            while True:
                with task_results_lock:
                    if task_id in task_results:
                        result_data = task_results.pop(task_id)
                        break
            new_problem = Problems(
                users_id=current_user.id,
                objective_functions=table_data1,
                decision_variables=table_data
            )
            storage.new(new_problem)
            storage.save()
            problem_id = new_problem.id
            new_opti_para = OptimizationParameters(
                problem_id=problem_id,
                user_id=current_user.id,
                pop_size=table_data3[0],
                iteration_no=table_data3[1],
                crossover_rate=table_data3[2],
                crossover_coef=table_data3[3],
                mutation_rate=table_data3[4],
                mutation_coef=table_data3[5]
            )
            storage.new(new_opti_para)
            storage.save()
            if len(table_data1) == 2:
                opti_front_obj1 = result_data['opti_front_obj1']
                opti_front_obj2 = result_data['opti_front_obj2']
                opti_para = result_data['opti_para']
                new_result = OptimizationResult(
                    problem_id=problem_id,
                    users_id=current_user.id,
                    opti_front_obj1=opti_front_obj1,
                    opti_front_obj2=opti_front_obj2,
                    opti_para=opti_para
                )
                storage.new(new_result)
                storage.save()
                return jsonify({
                    'opti_front_obj1': opti_front_obj1,
                    'opti_front_obj2': opti_front_obj2,
                    'opti_para': opti_para,
                    'message': 'Data recieved succesfully'
                })
            else:
                opti_front_obj1 = result_data['opti_front_obj1']
                opti_front_obj2 = result_data['opti_front_obj2']
                opti_front_obj3 = result_data['opti_front_obj3']
                opti_para = result_data['opti_para']
                new_result = OptimizationResult(
                    problem_id=problem_id,
                    users_id=current_user.id,
                    opti_front_obj1=opti_front_obj1,
                    opti_front_obj2=opti_front_obj2,
                    opti_front_obj3=opti_front_obj3,
                    opti_para=opti_para
                )
                storage.new(new_result)
                storage.save()
                return jsonify({
                    'opti_front_obj1': opti_front_obj1,
                    'opti_front_obj2': opti_front_obj2,
                    'opti_front_obj3': opti_front_obj3,
                    'opti_para': opti_para,
                    'message': 'Data recieved succesfully'
                })
        except Exception as e:
            return jsonify({"error": "Internal server error"}), 500
    return render_template('home.html')


@app.route('/about_us', strict_slashes=False)
def about_us():
    """ the about us route """
    return render_template('about_us.html')


@app.route('/contact_us', strict_slashes=False)
def contact_us():
    """ the contact us route """
    return render_template('contact_us.html')


def nsga2(
        table_data, table_data1, table_data2,
        table_data3, task_id):
    """
    implementation of NSGAII for two objectives
    """
    time.sleep(1)

    def con_str_int(table_dat):
        """ convert str2int """
        table_dataa = []
        # Converting the search space from string to ing
        for sublist in table_dat:
            row = []
            for item in sublist:
                if item.isdigit():
                    # Convert the string to an integer
                    item = int(item)
                row.append(item)
            table_dataa.append(row)
        return table_dataa
    table_new = con_str_int(table_data)
    # table_new1 = con_str_int1(table_data3)
    dv = []  # retrieving only the decision variables
    for k in range(len(table_new)):
        dv.append(table_new[k][0])
    pop_size = table_data3[0]
    bounds = [[row[1], row[2]] for row in table_new]
    nv = len(bounds)
    iteration = table_data3[1]
    crossover_rate = table_data3[2]
    ita_c = table_data3[3]
    mutation_rate = table_data3[4]
    ita_m = table_data3[5]
    nv = 30
    ghk = []
    newpopfit = []
    pop_posi = []

    def posi(bounds, pop_size):
        for i in range(pop_size and len(bounds)):
            x = (
                bounds[i][0] + np.random.rand(pop_size) *
                (bounds[i][1]-bounds[i][0])
                ).tolist()
            pop_posi.append(x)
        return pop_posi

    pops = posi(bounds, pop_size)
    popss = list(zip(*pops))
    pop = [list(ele) for ele in popss]
    objective_function = create_objective_function(
                            dv, table_data1[0])
    objective_function1 = create_objective_function(
                            dv, table_data1[1])
    # STAGE 1 (FAST NON DOMINATED SORTING)
    # CROWDINGIN DISTANCE
    ita_count = 0
    for zz in range(iteration):
        fitnessvalue = [
            objective_function(h) for h in pop]
        fitnessvalue1 = [
            objective_function1(h) for h in pop]
        fff = [fitnessvalue, fitnessvalue1]

        # DOMINANT DEPTH METHOD
        def fronting(pop_size, fitnessvalue, fitnessvalue1):
            """ this function computes
            fronts of the solutions
            """
            Fs = []
            for v in range(pop_size):
                Fs.append([])
            # Sps is the vector containing all solutions being
            # dominated by a particular solution
            Sps = []
            for g in range(pop_size):
                Sps.append([])
            # nxs is the vector containing the numbers of
            # solution dominating a particular solution
            nxs = []
            for i in range(pop_size):
                nx = 0
                for j in range(pop_size):
                    if i != j:
                        if (
                            dominates1(
                                [fitnessvalue[i], fitnessvalue[j]],
                                [fitnessvalue1[i], fitnessvalue1[j]],
                                table_data2)):
                            Sps[i].append(j)
                        elif (non_dominates1(
                               [fitnessvalue[i], fitnessvalue[j]],
                               [fitnessvalue1[i], fitnessvalue1[j]],
                               table_data2)):
                            continue
                        else:
                            nx = nx + 1
                # nxs-  number of solution that dominates fitenessvalue[i]
                nxs.append(nx)
                # if nx equal to zero meaning the solution is not dominated
                # by any solution and they belong to the first front or front 1
                if nx == 0:
                    P_rank = 0      # is  the front one ranking
                    # Fs is the matrix containing
                    # the fronts of all solutin
                    Fs[P_rank].append([i])
            # STAGE 2 (FAST NON DOMINATED SORTING)
            b = 0
            bb = 0
            q = []
            while Fs[b] != []:
                Q = []
                for w in range(len(Fs[b])):
                    # extracting the solutions from the first front
                    ee = Fs[b][w][bb]
                    # extracting the solutions of the Sps
                    # of the first front solutions
                    q.append(Sps[ee])
                # FLATENED q
                q = [item for sublist in q for item in sublist]
                for g in range(len(q)):
                    ee1 = q[g]  # extracting the solutions from list q
                    # the solutions of list q has values in nxs,
                    # so extracting it and substracting
                    # 1 as per the algorigthm
                    nxs[ee1] = (nxs[ee1]) - 1
                    if nxs[ee1] == 0:
                        q_rank = b + 1
                        Q.append([ee1])
                b = b + 1
                q = []
                Fs[b] = Q
            return Fs
        fronts = fronting(pop_size, fitnessvalue, fitnessvalue1)
    # END OF (FAST NON DOMINATED SORTING) THIS HELPS I
    # CONVERGANCE OF THE SOLUTION

        def cdist(fronts, pop_size, fff):
            """ the crowd distance computation
            to avoid solution clustering
            """
            fronts = [i for i in fronts if i != []]
            no_obj = 2
            fit_ = []
            sot_sols = []
            Cd_sol = [0] * pop_size
            ds1 = []
            for tt in range(len(fronts)):
                r = len(fronts[tt])
                for n in range(no_obj):
                    fit_.append([])
                for m in range(no_obj):
                    for j in range(r):
                        # extracting the solutions in the fronts
                        ee2 = fronts[tt][j][0]
                        # extracting the fitness values of the solutions
                        fit_[m].append(fff[m][ee2])
                    sot_sol = []
                    for cc in range(r):
                        fitidx = fit_[m].index(sorted(fit_[m])[cc])
                        sot_sol.append(fronts[tt][fitidx][0])
                    fit_[m] = sorted(fit_[m])
                    # Assigning a large value (inf) to the extrem solutions
                    Cd_sol[sot_sol[0]] = Cd_sol[sot_sol[-1]] = float('inf')
                    for w in range(1, r - 1):
                        if max(fit_[m]) == min(fit_[m]):
                            pass
                        else:
                            # computing the crowding distance
                            Cd_sol[sot_sol[w]] += \
                             abs(fit_[m][w + 1] - fit_[m][w - 1]) / \
                             (max(fit_[m]) - min(fit_[m]))
                    sot_sols.append(sot_sol)
                fit_ = []
            return Cd_sol, sot_sols
        crwdist = cdist(fronts, pop_size, fff)
        Cd_sol = crwdist[0]
        sot_sols = crwdist[1]

        def ranking(sol1):
            """ take in a sol and let it rank """
            rank = 5000
            for w in range(len(fronts)):
                for v in range(len(fronts[w])):
                    if sol1 == fronts[w][v][0]:
                        rank = w
            return rank

        def tornament(pop_size, ranking, Cd_sol):
            """ the selection of fittest and elimination of the weak """
            sols = list(range(len(Cd_sol)))
            random.shuffle(sols)
            mating_pool = []
            b = 0
            for n in range(pop_size - 1):
                candidate0 = sols[n]
                candidate1 = sols[n + 1]
                R_0 = ranking(candidate0)
                R_1 = ranking(candidate1)
                win = min(R_0, R_1)
                if R_0 < R_1:
                    mating_pool.append(candidate0)
                elif R_1 < R_0:
                    mating_pool.append(candidate1)
                elif R_0 == R_1:
                    W_0 = Cd_sol[sols[n]]
                    W_1 = Cd_sol[sols[n+1]]
                    if W_0 > W_1:
                        mating_pool.append(candidate0)
                    elif W_1 > W_0:
                        mating_pool.append(candidate1)
                    else:
                        W_0 == W_1
                        co = [candidate0, candidate1]
                        mating_pool.append(random.choice(co))
            candidatex = sols[-1]
            candidatey = sols[0]
            R_x = ranking(candidatex)
            R_y = ranking(candidatey)
            win = min(R_x, R_y)
            if R_x < R_y:
                mating_pool.append(candidatex)
            elif R_y < R_x:
                mating_pool.append(candidatey)
            elif R_x == R_y:
                W_x = Cd_sol[sols[0]]
                W_y = Cd_sol[sols[-1]]
                if W_x > W_y:
                    mating_pool.append(candidatex)
                elif W_y > W_x:
                    mating_pool.append(candidatey)
                else:
                    W_x == W_y
                    co = [candidatex, candidatey]
                    mating_pool.append(random.choice(co))
            return mating_pool
        winner = tornament(pop_size, ranking, Cd_sol)
        pp = []
        for e in range(len(winner)):
            pp.append(pop[winner[e]])

        def crossover(pp, crossover_rate):
            """ croosover function exsures
                exploration of the search space
            """
            palen = len(pp)    # lenght of parent
            iidx = list(range(pop_size))
            random.shuffle(iidx)
            pp = [pp[i] for i in iidx]
            cofs = []
            u = []
            betas = []
            for i in range(0, pop_size, 2):
                O1 = []
                O2 = []
                if random.random() < crossover_rate:
                    for y in range(len(bounds)):
                        ux = random.random()
                        if ux <= 0.5:
                            beta = (2 * ux)**(1/(ita_c+1))
                        else:
                            beta = (1/(2*(1-ux)))**(1/(ita_c+1))
                        bn1 = 0.5 * ((1 + beta) * pp[i][y]) + \
                            (1 - beta) * pp[i+1][y]
                        bn2 = 0.5 * ((1 - beta) * pp[i][y]) + \
                            (1 + beta) * pp[i+1][y]
                        O1.append(bn1)
                        O2.append(bn2)
                    cofs.append(O1)
                    cofs.append(O2)
                else:
                    cofs.append(pp[i])
                    cofs.append(pp[i+1])
            return cofs
        nin = crossover(pp, crossover_rate)
        for k in range(pop_size):
            for j in range(len(bounds)):
                if nin[k][j] < bounds[j][0]:
                    nin[k][j] = bounds[j][0]
                if nin[k][j] > bounds[j][1]:
                    nin[k][j] = bounds[j][1]

        def mutation(nin, mutation_rate):
            """ the mutation function ensures
                exploitation of search space
            """
            deltax = []
            u1 = []
            Oms = []
            for i in range(len(nin)):
                OO1 = []
                if random.random() < mutation_rate:
                    for m in range(len(bounds)):
                        uxx = random.random()
                        if uxx < 0.5:
                            delta = ((2 * uxx)**(1/(ita_m+1))) - 1
                        else:
                            delta = 1 - (2 * (1 - uxx))**(1/ita_m+1)
                        O_1 = nin[i][m] + (bounds[m][1] - bounds[m][0]) * delta
                        OO1.append(O_1)
                    Oms.append(OO1)
                else:
                    Oms.append(nin[i])
            return Oms
        firstoffsrping = mutation(nin, mutation_rate)
        for w in range(pop_size):
            for x in range(len(bounds)):
                if firstoffsrping[w][x] < bounds[x][0]:
                    firstoffsrping[w][x] = bounds[x][0]
                if firstoffsrping[w][x] > bounds[x][1]:
                    firstoffsrping[w][x] = bounds[x][1]
        fitnessiffso = [objective_function(r) for r in firstoffsrping]
        fitnessiffso1 = [objective_function1(r) for r in firstoffsrping]
        Uh = fitnessiffso, fitnessvalue
        Uh1 = fitnessiffso1, fitnessvalue1
        Uh = [element for sub in Uh for element in sub]
        Uh1 = [element for sub in Uh1 for element in sub]
        Uh_1 = fitnessiffso, fitnessvalue
        Uh1_1 = fitnessiffso1, fitnessvalue1
        fff1 = [Uh, Uh1]
        combinfronting = fronting(pop_size*2, Uh, Uh1)
        combcdist = cdist(combinfronting, pop_size*2, fff1)
        Cd_sol1 = combcdist[0]
        sot_sols1 = combcdist[1]
        nxt_gen = []
        for d in range(len(combinfronting)):
            L1 = len(combinfronting[d])
            L2 = sum([len(k) for k in nxt_gen])
            if L1 <= pop_size - L2:
                nxt_gen.append(combinfronting[d])
                L2 = sum([len(k) for k in nxt_gen])
                if L2 < pop_size:
                    continue
                else:
                    L2 == pop_size
                    break
            else:
                L1 > pop_size - L2
                # crowding distance of the solutions in front[3]
                solcmdist = []
                for m in range(len(combinfronting[d])):
                    solcmdist.append(combcdist[0][combinfronting[d][m][0]])
                # Checking if all crowd distance is infinity
                che1 = check(solcmdist, float('inf'))
                if che1:
                    sunn = pop_size-L2
                    # flated combinfronting[3]
                    flt = [i for sublist in combinfronting[d] for i in sublist]
                    if sunn > len(flt):
                        sunn = len(flt)
                    # sel --- selecting solution to complete population
                    sel = np.random.choice(flt, size=sunn, replace=False)
                    sel = list(sel)
                    sel = makedub(sel)
                    # for a1 in range(len(sel)):
                    nxt_gen.append(sel)
                else:
                    copi_solcmdist = copy.copy(solcmdist)
                    sort_solcmdist = solcmdist.sort(reverse=True)
                    sort_solu = []
                    for y in range(len(combinfronting[d])):
                        indx = copi_solcmdist .index(solcmdist[y])
                        new1 = combinfronting[d][indx][0]
                        sort_solu.append(new1)
                    # repeated index
                    indx_rep = [
                        i for i,
                        val in enumerate(copi_solcmdist)
                        if val in copi_solcmdist[:i]]
                    indx_rep1 = [
                        y for y,
                        val in enumerate(solcmdist)
                        if val in solcmdist[:y]]
                    for u in range(len(indx_rep)):
                        if solcmdist[u] == solcmdist[u+1]:
                            new1 = combinfronting[d][indx_rep[u]][0]
                            sort_solu.insert(indx_rep1[u], new1)
                            del sort_solu[indx_rep1[u]+1]
                    # sort_solu = list(sort_solu)
                    xx = sort_solu[0:(pop_size - L2)]
                    xx = makedub(xx)
                    nxt_gen.append(xx)
                    L2 = sum([len(k) for k in nxt_gen])
                    if L2 < pop_size:
                        continue
                    else:
                        L2 == pop_size
                        break
            if len(combinfronting[0]) == pop_size * 2:
                copi_solcmdist = copy.copy(solcmdist)
                sort_solcmdist = solcmdist.sort(reverse=True)
                sort_solu = []
                for y in range(len(combinfronting[d])):
                    indx = copi_solcmdist .index(solcmdist[y])
                    new1 = combinfronting[d][indx][0]
                    sort_solu.append(new1)
                # repeated index
                indx_rep = [
                    i for i,
                    val in enumerate(copi_solcmdist)
                    if val in copi_solcmdist[:i]]
                indx_rep1 = [
                    y for y,
                    val in enumerate(solcmdist)
                    if val in solcmdist[:y]]
                for u in range(len(indx_rep)):
                    if solcmdist[u] == solcmdist[u+1]:
                        new1 = combinfronting[d][indx_rep[u]][0]
                        sort_solu.insert(indx_rep1[u], new1)
                        del sort_solu[indx_rep1[u]+1]
                # sort_solu = list(sort_solu)
                xx = sort_solu[0: pop_size]
                xx = makedub(xx)
                nxt_gen.append(xx)
    # new codes
        popcomb = firstoffsrping + pop
        nxt_gen = [i for sublist in nxt_gen for i in sublist]
        nxt_gen = [i for sublist in nxt_gen for i in sublist]
        nxt_gen_pop = []
        for m in range(len(nxt_gen)):
            nxt_gen_pop.append(popcomb[nxt_gen[m]])
        nxt_gen_fit1 = []
        for n in range(len(nxt_gen)):
            nxt_gen_fit1.append(Uh[nxt_gen[n]])
        nxt_gen_fit2 = []
        for n in range(len(nxt_gen)):
            nxt_gen_fit2.append(Uh1[nxt_gen[n]])
        pop = nxt_gen_pop
        fff = []
        xx = fronting(pop_size, nxt_gen_fit1, nxt_gen_fit2)
        try:
            socketio.emit(
                'update',
                {'nxt_gen_fit1': nxt_gen_fit1,
                 'nxt_gen_fit2': nxt_gen_fit2})
        except Exception as e:
            print('Error emitting update event:', str(e))
        ita_count = ita_count + 1
    result_data = {
        'opti_front_obj1': nxt_gen_fit1,
        'opti_front_obj2': nxt_gen_fit2,
        'opti_para': pop
    }
    with task_results_lock:
        task_results[task_id] = result_data


def nsgaa2(
        table_data,
        table_data1,
        table_data2, table_data3, task_id):
    """
    Implementing NSGAII for three objectives
    """
    time.sleep(1)
    table_dataa = []
    # Converting the search space from string to ing
    for sublist in table_data:
        row = []
        for item in sublist:
            if item.isdigit():
                # Convert the string to an integer
                item = int(item)
            row.append(item)
        table_dataa.append(row)
    table_new = table_dataa
    dv = []  # retrieving only the decision variables
    for k in range(len(table_new)):
        dv.append(table_new[k][0])
    pop_size = table_data3[0]
    bounds = [[row[1], row[2]] for row in table_new]
    nv = len(bounds)
    iteration = table_data3[1]
    crossover_rate = table_data3[2]
    mutation_rate = table_data3[4]
    ita_c = table_data3[3]
    ita_m = table_data3[5]
    nv = 30
    ghk = []
    newpopfit = []
    pop_posi = []

    def posi(bounds, pop_size):
        for i in range(pop_size and len(bounds)):
            x = (
                bounds[i][0] +
                np.random.rand(pop_size)*(bounds[i][1]
                                          - bounds[i][0])
                ).tolist()
            pop_posi.append(x)
        return pop_posi
    pops = posi(bounds, pop_size)
    popss = list(zip(*pops))
    pop = [list(ele) for ele in popss]
    objective_function = create_objective_function(dv, table_data1[0])
    objective_function1 = create_objective_function(dv, table_data1[1])
    objective_function2 = create_objective_function(dv, table_data1[2])
    # STAGE 1 (FAST NON DOMINATED SORTING)
    # CROWDINGIN DISTANCE
    ita_count = 0
    for zz in range(iteration):
        fitnessvalue = [objective_function(h) for h in pop]
        fitnessvalue1 = [objective_function1(h) for h in pop]
        fitnessvalue2 = [objective_function2(h) for h in pop]
        fff = [fitnessvalue, fitnessvalue1, fitnessvalue2]

        # DOMINANT DEPTH METHOD
        def fronting(
                pop_size, fitnessvalue,
                fitnessvalue1, fitnessvalue2):
            """
            classify solution into different
            front base on their fitness values

            """
            Fs = []
            for v in range(pop_size):
                Fs.append([])
            # Sps is the vector containing all solutions
            # being dominated by a particular solution
            Sps = []
            for g in range(pop_size):
                Sps.append([])
            # nxs is the vector containing the numbers of
            # solution dominating a particular solution
            nxs = []
            for i in range(pop_size):
                nx = 0
                for j in range(pop_size):
                    if i != j:
                        if (dominates(
                              [fitnessvalue[i], fitnessvalue[j]],
                              [fitnessvalue1[i], fitnessvalue1[j]],
                              [fitnessvalue2[i], fitnessvalue2[j]],
                              table_data2)):
                            Sps[i].append(j)
                        elif (non_dominates(
                              [fitnessvalue[i], fitnessvalue[j]],
                              [fitnessvalue1[i], fitnessvalue1[j]],
                              [fitnessvalue2[i], fitnessvalue2[j]],
                              table_data2)):
                            continue
                        else:
                            nx = nx + 1
                # nxs-  number of solution that
                # dominates fitenessvalue[i]
                nxs.append(nx)
                # if nx equal to zero meaning the solution is not
                # dominated by any solution and they
                # belong to the first front or front 1
                if nx == 0:
                    P_rank = 0                     # is  the front one ranking
                    # Fs is the matrix containing the fronts of all solutin
                    Fs[P_rank].append([i])
            # STAGE 2 (FAST NON DOMINATED SORTING)
            b = 0
            bb = 0
            q = []
            while Fs[b] != []:
                Q = []
                for w in range(len(Fs[b])):
                    # extracting the solutions from the first front
                    ee = Fs[b][w][bb]
                    # extracting the solutions of the Sps
                    # of the first front solutions
                    q.append(Sps[ee])
                q = [item for sublist in q for item in sublist]  # FLATENED q
                for g in range(len(q)):
                    # extracting the solutions from list q
                    ee1 = q[g]
                    # the solutions of list q has values in nxs ,
                    # so extracting it
                    # and substracting 1 as per the algorigthm
                    nxs[ee1] = (nxs[ee1]) - 1
                    if nxs[ee1] == 0:
                        q_rank = b + 1
                        Q.append([ee1])
                b = b + 1
                q = []
                Fs[b] = Q
            return Fs
        fronts = fronting(pop_size, fitnessvalue, fitnessvalue1, fitnessvalue2)
    # END OF (FAST NON DOMINATED SORTING)
    # THIS HELPS I CONVERGANCE OF THE SOLUTION

        def cdist(fronts, pop_size, fff):
            """
            computes the crowding distance

            """
            fronts = [i for i in fronts if i != []]
            no_obj = 3
            fit_ = []
            sot_sols = []
            Cd_sol = [0] * pop_size
            ds1 = []
            for tt in range(len(fronts)):
                r = len(fronts[tt])
                for n in range(no_obj):
                    fit_.append([])
                for m in range(no_obj):
                    for j in range(r):
                        # extracting the solutions in the fronts
                        ee2 = fronts[tt][j][0]
                        # extracting the fitness values of the solutions
                        fit_[m].append(fff[m][ee2])
                    sot_sol = []
                    for cc in range(r):
                        fitidx = fit_[m].index(sorted(fit_[m])[cc])
                        sot_sol.append(fronts[tt][fitidx][0])
                    fit_[m] = sorted(fit_[m])
                    # Assigning a large value (inf) to the extrem solutions
                    Cd_sol[sot_sol[0]] = Cd_sol[sot_sol[-1]] = float('inf')
                    for w in range(1, r - 1):
                        if max(fit_[m]) == min(fit_[m]):
                            pass
                        else:
                            # computing the crowding distance
                            Cd_sol[sot_sol[w]] += \
                              abs(fit_[m][w + 1] - fit_[m][w - 1]) / \
                              (max(fit_[m]) - min(fit_[m]))
                    sot_sols.append(sot_sol)
                fit_ = []
            return Cd_sol, sot_sols
        crwdist = cdist(fronts, pop_size, fff)
        Cd_sol = crwdist[0]
        sot_sols = crwdist[1]

        def ranking(sol1):
            """ checks the rank of a solution """
            rank = 5000
            for w in range(len(fronts)):
                for v in range(len(fronts[w])):
                    if sol1 == fronts[w][v][0]:
                        rank = w
            return rank

        def tornament(pop_size, ranking, Cd_sol):
            """ Selection process eliminates the weak """
            sols = list(range(len(Cd_sol)))
            random.shuffle(sols)
            mating_pool = []
            b = 0
            for n in range(pop_size-1):
                candidate0 = sols[n]
                candidate1 = sols[n+1]
                R_0 = ranking(candidate0)
                R_1 = ranking(candidate1)
                win = min(R_0, R_1)
                if R_0 < R_1:
                    mating_pool.append(candidate0)
                elif R_1 < R_0:
                    mating_pool.append(candidate1)
                elif R_0 == R_1:
                    W_0 = Cd_sol[sols[n]]
                    W_1 = Cd_sol[sols[n+1]]
                    if W_0 > W_1:
                        mating_pool.append(candidate0)
                    elif W_1 > W_0:
                        mating_pool.append(candidate1)
                    else:
                        W_0 == W_1
                        co = [candidate0, candidate1]
                        mating_pool.append(random.choice(co))
            candidatex = sols[-1]
            candidatey = sols[0]
            R_x = ranking(candidatex)
            R_y = ranking(candidatey)
            win = min(R_x, R_y)
            if R_x < R_y:
                mating_pool.append(candidatex)
            elif R_y < R_x:
                mating_pool.append(candidatey)
            elif R_x == R_y:
                W_x = Cd_sol[sols[0]]
                W_y = Cd_sol[sols[-1]]
                if W_x > W_y:
                    mating_pool.append(candidatex)
                elif W_y > W_x:
                    mating_pool.append(candidatey)
                else:
                    W_x == W_y
                    co = [candidatex, candidatey]
                    mating_pool.append(random.choice(co))
            return mating_pool

        winner = tornament(pop_size, ranking, Cd_sol)
        pp = []
        for e in range(len(winner)):
            pp.append(pop[winner[e]])

        def crossover(pp, crossover_rate):
            """
            the crossover operator
            """
            palen = len(pp)    # lenght of parent
            iidx = list(range(pop_size))
            random.shuffle(iidx)
            pp = [pp[i] for i in iidx]
            cofs = []
            u = []
            betas = []
            for i in range(0, pop_size, 2):
                O1 = []
                O2 = []
                if random.random() < crossover_rate:
                    for y in range(len(bounds)):
                        ux = random.random()
                        if ux <= 0.5:
                            beta = (2 * ux)**(1/(ita_c+1))
                        else:
                            beta = (1/(2*(1-ux)))**(1/(ita_c+1))
                        bn1 = 0.5 * ((1 + beta) * pp[i][y])
                        + (1 - beta) * pp[i+1][y]
                        bn2 = 0.5 * ((1 - beta) * pp[i][y])
                        + (1 + beta) * pp[i+1][y]
                        O1.append(bn1)
                        O2.append(bn2)
                    cofs.append(O1)
                    cofs.append(O2)
                else:
                    cofs.append(pp[i])
                    cofs.append(pp[i+1])
            return cofs
        nin = crossover(pp, crossover_rate)
        for k in range(pop_size):
            for j in range(len(bounds)):
                if nin[k][j] < bounds[j][0]:
                    nin[k][j] = bounds[j][0]
                if nin[k][j] > bounds[j][1]:
                    nin[k][j] = bounds[j][1]

        def mutation(nin, mutation_rate):
            """
            the mutation operator

            """
            deltax = []
            u1 = []
            Oms = []
            for i in range(len(nin)):
                OO1 = []
                if random.random() < mutation_rate:
                    for m in range(len(bounds)):
                        uxx = random.random()
                        if uxx < 0.5:
                            delta = ((2 * uxx)**(1/(ita_m+1))) - 1
                        else:
                            delta = 1 - (2 * (1 - uxx))**(1/ita_m+1)
                        O_1 = nin[i][m] + (bounds[m][1] - bounds[m][0]) * delta
                        OO1.append(O_1)
                    Oms.append(OO1)
                else:
                    Oms.append(nin[i])
            return Oms

        firstoffsrping = mutation(nin, mutation_rate)
        # checking for boundary voilations
        for w in range(pop_size):
            for x in range(len(bounds)):
                if firstoffsrping[w][x] < bounds[x][0]:
                    firstoffsrping[w][x] = bounds[x][0]
                if firstoffsrping[w][x] > bounds[x][1]:
                    firstoffsrping[w][x] = bounds[x][1]
        fitnessiffso = [objective_function(r) for r in firstoffsrping]
        fitnessiffso1 = [objective_function1(r) for r in firstoffsrping]
        fitnessiffso2 = [objective_function2(r) for r in firstoffsrping]
        Uh = fitnessiffso, fitnessvalue
        Uh1 = fitnessiffso1, fitnessvalue1
        Uh2 = fitnessiffso2, fitnessvalue2
        Uh = [element for sub in Uh for element in sub]
        Uh1 = [element for sub in Uh1 for element in sub]
        Uh2 = [element for sub in Uh2 for element in sub]
        Uh_1 = fitnessiffso, fitnessvalue
        Uh1_1 = fitnessiffso1, fitnessvalue1
        Uh2_2 = fitnessiffso2, fitnessvalue2
        fff1 = [Uh, Uh1, Uh2]
        combinfronting = fronting(pop_size*2, Uh, Uh1, Uh2)
        combcdist = cdist(combinfronting, pop_size*2, fff1)
        Cd_sol1 = combcdist[0]
        sot_sols1 = combcdist[1]
        nxt_gen = []
        for d in range(len(combinfronting)):
            L1 = len(combinfronting[d])
            L2 = sum([len(k) for k in nxt_gen])
            if L1 <= pop_size - L2:
                nxt_gen.append(combinfronting[d])
                L2 = sum([len(k) for k in nxt_gen])
                if L2 < pop_size:
                    continue
                else:
                    L2 == pop_size
                    break
            else:
                L1 > pop_size - L2
                # crowding distance of the solutions in front[3]
                solcmdist = []
                for m in range(len(combinfronting[d])):
                    solcmdist.append(combcdist[0][combinfronting[d][m][0]])
                # Checking if all crowd distance is infinity
                che1 = check(solcmdist, float('inf'))
                if che1:  # if True
                    sunn = pop_size-L2
                    # flated combinfronting[3]
                    flt = [i for sublist in combinfronting[d] for i in sublist]
                    if sunn > len(flt):
                        sunn = len(flt)
                    # sel --- selecting solution to complete population
                    sel = np.random.choice(flt, size=sunn, replace=False)
                    sel = list(sel)
                    sel = makedub(sel)
                    nxt_gen.append(sel)
                else:
                    copi_solcmdist = copy.copy(solcmdist)
                    sort_solcmdist = solcmdist.sort(reverse=True)
                    sort_solu = []
                    for y in range(len(combinfronting[d])):
                        indx = copi_solcmdist .index(solcmdist[y])
                        new1 = combinfronting[d][indx][0]
                        sort_solu.append(new1)
                    # repeated index
                    indx_rep = [
                        i for i,
                        val in enumerate(copi_solcmdist)
                        if val in copi_solcmdist[:i]]
                    indx_rep1 = [
                        y for y,
                        val in enumerate(solcmdist)
                        if val in solcmdist[:y]]
                    for u in range(len(indx_rep)):
                        if solcmdist[u] == solcmdist[u+1]:
                            new1 = combinfronting[d][indx_rep[u]][0]
                            sort_solu.insert(indx_rep1[u], new1)
                            del sort_solu[indx_rep1[u]+1]
                    xx = sort_solu[0:(pop_size - L2)]
                    xx = makedub(xx)
                    nxt_gen.append(xx)
                    L2 = sum([len(k) for k in nxt_gen])
                    if L2 < pop_size:
                        continue
                    else:
                        L2 == pop_size
                        break
            if len(combinfronting[0]) == pop_size * 2:
                copi_solcmdist = copy.copy(solcmdist)
                sort_solcmdist = solcmdist.sort(reverse=True)
                sort_solu = []
                for y in range(len(combinfronting[d])):
                    indx = copi_solcmdist .index(solcmdist[y])
                    new1 = combinfronting[d][indx][0]
                    sort_solu.append(new1)
                # repeated index
                indx_rep = [
                    i for i,
                    val in enumerate(copi_solcmdist)
                    if val in copi_solcmdist[:i]]
                indx_rep1 = [
                    y for y,
                    val in enumerate(solcmdist)
                    if val in solcmdist[:y]]
                for u in range(len(indx_rep)):
                    if solcmdist[u] == solcmdist[u+1]:
                        new1 = combinfronting[d][indx_rep[u]][0]
                        sort_solu.insert(indx_rep1[u], new1)
                        del sort_solu[indx_rep1[u]+1]
                xx = sort_solu[0: pop_size]
                xx = makedub(xx)
                nxt_gen.append(xx)
        popcomb = firstoffsrping + pop
        nxt_gen = [i for sublist in nxt_gen for i in sublist]
        nxt_gen = [i for sublist in nxt_gen for i in sublist]
        nxt_gen_pop = []
        for m in range(len(nxt_gen)):
            nxt_gen_pop.append(popcomb[nxt_gen[m]])
        nxt_gen_fit1 = []
        for n in range(len(nxt_gen)):
            nxt_gen_fit1.append(Uh[nxt_gen[n]])
        nxt_gen_fit2 = []
        for n in range(len(nxt_gen)):
            nxt_gen_fit2.append(Uh1[nxt_gen[n]])
        nxt_gen_fit3 = []
        for n in range(len(nxt_gen)):
            nxt_gen_fit3.append(Uh2[nxt_gen[n]])
        pop = nxt_gen_pop
        fff = []
        xx = fronting(pop_size, nxt_gen_fit1, nxt_gen_fit2, nxt_gen_fit3)
        try:
            socketio.emit(
                'update',
                {'nxt_gen_fit1': nxt_gen_fit1,
                 'nxt_gen_fit2': nxt_gen_fit2, 'nxt_gen_fit3': nxt_gen_fit3})
        except Exception as e:
            print('Error emitting update event:', str(e))
        ita_count = ita_count + 1
    result_data = {
        'opti_front_obj1': nxt_gen_fit1,
        'opti_front_obj2': nxt_gen_fit2,
        'opti_front_obj3': nxt_gen_fit3,
        'opti_para': pop
    }
    with task_results_lock:
        task_results[task_id] = result_data


def create_objective_function(variables, formula):
    """
    used for dynamically constructing the objective functions
    """
    # Start constructing the function as a string
    func_str = "def objective_function(I):\n"
    # Extracting values from the list I
    for i, var in enumerate(variables):
        func_str += f"    {var} = I[{i}]\n"
    # Add the formula calculation
    func_str += f"    TCC = {formula}\n"
    # Return the calculated TCC
    func_str += "    return TCC\n"
    # Create a local dictionary to execute the function definition
    local_dict = {}
    # Execute the constructed function definition
    exec(func_str, globals(), local_dict)
    # Return the dynamically created function
    return local_dict['objective_function']


def dominates1(ind1, ind2, obj_types):
    """
    Return True if ind1 dominates ind2
    based on objective types. allow for
    dynamic selection of objectives
    """
    if obj_types[0] == "Minimization" and obj_types[1] == "Minimization":
        if (ind1[0] <= ind1[1] and ind2[0] <= ind2[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Maximization":
        if (ind1[0] >= ind1[1] and ind2[0] >= ind2[1]):
            return True
    elif obj_types[0] == "Minimization" and obj_types[1] == "Maximization":
        if (ind1[0] <= ind1[1] and ind2[0] >= ind2[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Minimization":
        if (ind1[0] >= ind1[1] and ind2[0] <= ind2[1]):
            return True
    return False


def non_dominates1(ind1, ind2, obj_types):
    """Return True if ind1 dominates ind2 based on objective types.
        allows for dynamic selection of objectives
    """
    if obj_types[0] == "Minimization" and obj_types[1] == "Minimization":
        if (ind1[0] < ind1[1] or ind2[0] < ind2[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Maximization":
        if (ind1[0] > ind1[1] or ind2[0] > ind2[1]):
            return True
    elif obj_types[0] == "Minimization" and obj_types[1] == "Maximization":
        if (ind1[0] < ind1[1] or ind2[0] > ind2[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Minimization":
        if (ind1[0] > ind1[1] or ind2[0] < ind2[1]):
            return True
    return False


def dominates(ind1, ind2, ind3, obj_types):
    """Return True if ind1 dominates ind2 based on objective types."""
    if obj_types[0] == "Minimization" and obj_types[1] == "Minimization" \
            and obj_types[2] == "Minimization":
        if (ind1[0] <= ind1[1] and ind2[0] <= ind2[1] and ind3[0] <= ind3[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Maximization" \
            and obj_types[2] == "Maximization":
        if (ind1[0] >= ind1[1] and ind2[0] >= ind2[1] and ind3[0] >= ind3[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Minimization" \
            and obj_types[2] == "Minimization":
        if (ind1[0] >= ind1[1] and ind2[0] <= ind2[1] and ind3[0] <= ind3[1]):
            return True
    elif obj_types[0] == "Minimization" and obj_types[1] == "Maximization" \
            and obj_types[2] == "Maximization":
        if (ind1[0] <= ind1[1] and ind2[0] >= ind2[1] and ind3[0] >= ind3[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Maximization" \
            and obj_types[2] == "Minimization":
        if (ind1[0] >= ind1[1] and ind2[0] >= ind2[1] and ind3[0] <= ind3[1]):
            return True
    elif obj_types[0] == "Minimization" and obj_types[1] == "Minimization" \
            and obj_types[2] == "Maximization":
        if (ind1[0] <= ind1[1] and ind2[0] <= ind2[1] and ind3[0] >= ind3[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Minimization" \
            and obj_types[2] == "Maximization":
        if (ind1[0] >= ind1[1] and ind2[0] <= ind2[1] and ind3[0] >= ind3[1]):
            return True
    elif obj_types[0] == "Minimization" and obj_types[1] == "Maximization" \
            and obj_types[2] == "Minimization":
        if (ind1[0] <= ind1[1] and ind2[0] >= ind2[1] and ind3[0] <= ind3[1]):
            return True
    return False


def non_dominates(ind1, ind2, ind3, obj_types):
    """Return True if ind1 dominates ind2 based on objective types."""
    if obj_types[0] == "Minimization" and obj_types[1] == "Minimization" \
            and obj_types[2] == "Minimization":
        if (ind1[0] < ind1[1] or ind2[0] < ind2[1] or ind3[0] < ind3[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Maximization" \
            and obj_types[2] == "Maximization":
        if (ind1[0] > ind1[1] or ind2[0] > ind2[1] or ind3[0] > ind3[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Minimization" \
            and obj_types[2] == "Minimization":
        if (ind1[0] > ind1[1] or ind2[0] < ind2[1] or ind3[0] < ind3[1]):
            return True
    elif obj_types[0] == "Minimization" and obj_types[1] == "Maximization" \
            and obj_types[2] == "Maximization":
        if (ind1[0] < ind1[1] or ind2[0] > ind2[1] or ind3[0] > ind3[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Maximization" \
            and obj_types[2] == "Minimization":
        if (ind1[0] > ind1[1] or ind2[0] > ind2[1] or ind3[0] < ind3[1]):
            return True
    elif obj_types[0] == "Minimization" and obj_types[1] == "Minimization" \
            and obj_types[2] == "Maximization":
        if (ind1[0] < ind1[1] or ind2[0] < ind2[1] or ind3[0] > ind3[1]):
            return True
    elif obj_types[0] == "Maximization" and obj_types[1] == "Minimization" \
            and obj_types[2] == "Maximization":
        if (ind1[0] > ind1[1] or ind2[0] < ind2[1] or ind3[0] > ind3[1]):
            return True
    elif obj_types[0] == "Minimization" and obj_types[1] == "Maximization" \
            and obj_types[2] == "Minimization":
        if (ind1[0] < ind1[1] or ind2[0] > ind2[1] or ind3[0] < ind3[1]):
            return True
    return False


def check(listt, val):
    """ this checks """
    bn = 0
    for xc in range(len(listt)):
        if val == listt[xc]:
            bn = bn + 1
        if bn == len(listt):
            return True
    return False


def makedub(xi):
    """ this doubles """
    xh = [[t] for t in xi]
    return xh


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5006, debug=True)
