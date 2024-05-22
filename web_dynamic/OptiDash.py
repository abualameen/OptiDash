#!/usr/bin/python3
"""
this the choice crafter flask app
"""

# from re import M
# from timeit import repeat
# from numpy.random import beta, rand, randint
# import numpy as np
# # import matplotlib.pyplot as plt
# import matplotlib
# from requests import delete
# from sympy import N

# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from numpy import append, array, exp, sqrt
# import tempfile

# from mpl_toolkits.mplot3d import axes3d
# from models import storage
# from models.criteria import Criteria
# from models.alternative import Alternative
# from models.result import Result

# from flask import Flask, render_template, request, jsonify
# from sqlalchemy import String, Float, ForeignKey
# from sqlalchemy import create_engine, Column, Integer
# from sqlalchemy.orm import sessionmaker, relationship
# from sqlalchemy.orm import declarative_base
# from sqlalchemy.ext.declarative import declarative_base
# from os import getenv
# from flask import session
# import secrets
# import json
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import time
import copy
import copy
import math
import random
import numpy as np
from cmath import inf, nan

from flask_cors import CORS






app = Flask(__name__)
CORS(app) 
socketio = SocketIO(app, cors_allowed_origins="*")

# @app.teardown_appcontext
# def close_db(error):
#     """ Remove the current SQLAlchemy Session """
#     storage.close()


@app.route('/', methods=['GET', 'POST'], strict_slashes=False)
def index():
    """ this is a index function for the index route """
    return render_template('index.html')

@app.route('/home', methods=['GET', 'POST'], strict_slashes=False)
def home():
    """ this is a index function for the index route """
    if request.method == "POST":
        data = request.get_json()
        # print('data:', data)
        table_data = data.get('tableData')
        table_data1 = data.get('tableData1')
        # print('table:', table_data)
        # print('table1:', table_data1)
        # outputs = nsga2(table_data, table_data1)
        # opti_obj1_fit = outputs[0]
        # opti_obj2_fit = outputs[1]
        # opti_dv = outputs[2]
        if len(table_data1) == 2:
            socketio.start_background_task(target=nsga2, table_data=table_data, table_data1=table_data1)
        else:
            socketio.start_background_task(target=nsgaa2, table_data=table_data, table_data1=table_data1)
    
       
        return jsonify({'status': 'Processing started'})
        
    return render_template('home.html')

@app.route('/about_us', strict_slashes=False)
def about_us():
    """ the about us route """
    return render_template('about_us.html')

@app.route('/contact_us', strict_slashes=False)
def contact_us():
    """ the contact us route """
    return render_template('contact_us.html')


def nsga2(table_data, table_data1):
    table_dataa = []
    for sublist in table_data: # Converting the search space from string to ing
        row = []
        for item in sublist:
            if item.isdigit():
                # Convert the string to an integer
                item = int(item)
            row.append(item)
        table_dataa.append(row)
    table_new = table_dataa
    dv = [] # retrieving only the decision variables
    for k in range(len(table_new)):
        dv.append(table_new[k][0])

    pop_size = 100
    # bounds = [[table_new[0][1], table_new[0][2]], [table_new[1][1], table_new[1][2]], [table_new[2][1],table_new[2][2]]]
    bounds = [[row[1], row[2]] for row in table_new]
    nv = len(bounds)
    # print(nv)
    iteration = 30
    numb_constr = 2
    crossover_rate = 0.9
    mutation_rate = 1/nv
    ita_c = 20
    ita_m = 100
    ghk = []
    newpopfit = []

    pop_posi = []
    def posi(bounds, pop_size):
        for i in range(pop_size and len(bounds)):
            x = (bounds[i][0] + np.random.rand(pop_size)*(bounds[i][1]-bounds[i][0])).tolist()
            pop_posi.append(x)
        return pop_posi
    pops= posi(bounds, pop_size)
    popss = list(zip(*pops))

    pop = [list(ele) for ele in popss]

    objective_function = create_objective_function(dv, table_data1[0])
    objective_function1 = create_objective_function(dv, table_data1[1])
    
    fitnessvalue = [objective_function(h) for h in pop]
    fitnessvalue1  = [objective_function1(h) for h in pop]
    nxt_gen_fit1 = [objective_function(h) for h in pop]
    nxt_gen_fit2  = [objective_function1(h) for h in pop]
    # socketio.emit('update', {'fit1': fitnessvalue, 'fit2': fitnessvalue1})
    try:
        socketio.emit('update', {'nxt_gen_fit1': nxt_gen_fit1, 'nxt_gen_fit2': nxt_gen_fit2})
        print('Update event emitted successfully')
    except Exception as e:
        print('Error emitting update event:', str(e))
    fff = [fitnessvalue ,fitnessvalue1]
    # print('fff:', fff)

    # fig = plt.figure()
    # plt.rcParams['font.family'] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"]
    # plt.scatter(fitnessvalue, fitnessvalue1, c = 'blue')
    # fig.suptitle('Evolution of the best chromosome')
    # plt.xlabel('TCC')
    # plt.ylabel('TEE')

    #     ############# CROWDINGIN DISTANCE ########################################
    for zz in range(iteration):
        time.sleep(1)
        fitnessvalue  = [objective_function(h) for h in pop]
        fitnessvalue1  = [objective_function1(h) for h in pop]
        fff = [fitnessvalue ,fitnessvalue1]
 
        def fronting(pop_size,fitnessvalue ,fitnessvalue1): #### DOMINANT DEPTH METHOD
            Fs = []
            for v in range(pop_size):
                Fs.append([])

            Sps = []                   # Sps is the vector containing all solutions being dominated by a particular solution
            for g in range(pop_size):
                Sps.append([])

            nxs = []                   # nxs is the vector containing the numbers of solution dominating a particular solution
            for g in range(pop_size):
                nxs.append([])

            for i in range(len(fitnessvalue)):
                Sp = []
                nx = 0
                nxx = 0
                if i != 0:
                    if fitnessvalue [i] == fitnessvalue [0] and fitnessvalue1 [i] == fitnessvalue1 [0]: 
                        pass # print( 'this solutions are identical1')

                elif fitnessvalue [i] < fitnessvalue [0] or fitnessvalue1 [i] < fitnessvalue1 [0]: #and fitnessvalue [i] < fitnessvalue [j] or fitnessvalue1 [i] < fitnessvalue1 [j]: 
                    if fitnessvalue [i] <= fitnessvalue [0] and fitnessvalue1 [i] <= fitnessvalue1 [0]:
                        Sps[i].append([0])

                elif fitnessvalue [i] < fitnessvalue [0] or fitnessvalue1 [i] < fitnessvalue1 [0]: #and fitnessvalue [i] < fitnessvalue [j] or fitnessvalue1 [i] < fitnessvalue1 [j]: 
                    if fitnessvalue [i] > fitnessvalue [0] or fitnessvalue1 [i] > fitnessvalue1 [0]:
                        nx = nx
                else:
                    nx = nx + 1
                for j in range(len(fitnessvalue)):
                    if j + 1 != i  and j+1 < len(fitnessvalue ):
                        if fitnessvalue [i] == fitnessvalue [j+1] and fitnessvalue1 [i] == fitnessvalue1 [j+1]:
                            pass # print( 'this solutions are identical2')
                        elif fitnessvalue [i] < fitnessvalue [j+1] or fitnessvalue1 [i] < fitnessvalue1 [j+1]: #and fitnessvalue [i] < fitnessvalue [j] or fitnessvalue1 [i] < fitnessvalue1 [j]: 
                            if fitnessvalue [i] <= fitnessvalue [j+1] and fitnessvalue1 [i] <= fitnessvalue1 [j+1]:
                                Sps[i].append([j+1])
                        
                        elif fitnessvalue [i] < fitnessvalue [j+1] or fitnessvalue1 [i] < fitnessvalue1 [j+1]: #and fitnessvalue [i] < fitnessvalue [j] or fitnessvalue1 [i] < fitnessvalue1 [j]: 
                            if fitnessvalue [i] > fitnessvalue [j+1] or fitnessvalue1 [i] > fitnessvalue1 [j+1]:
                                nx = 0
                        else:
                            nx = nx + 1
        
                nxs[i].append(nx)
                if nx == 0:                         # if nx equal to zero meaning the solution is not dominated by any solution and they belong to the first front or front 1
                    P_rank = 0                     # is  the front one ranking
                    Fs[P_rank].append([i])         # Fs is the matrix containing the fronts of all solutin

            ######## STAGE 2 (FAST NON DOMINATED SORTING)###########

            b = 0
            bb = 0
            q = []
            while Fs[b] != []:
                Q = []
                for w in range(len(Fs[b])):
                    ee = Fs[b][w][bb]

                    q.append(Sps[ee])
            
                q = [item for sublist in q for item in sublist] # FLATENED q
                
                for g in range(len(q)):
                    ee1 =  q[g][bb]
                    nxs[ee1][bb]= (nxs[ee1][bb]) - 1
                    if nxs[ee1][bb] == 0 :
                        q_rank = b+1
                        Q.append([ee1])
                b = b +1
                q = []
                Fs[b] = Q
            return Fs
        fronts = fronting(pop_size,fitnessvalue ,fitnessvalue1)

    ################## END OF (FAST NON DOMINATED SORTING) THIS HELPS I CONVERGANCE OF THE SOLUTION ###############################

        def cdist(fronts,pop_size,fff):
            fronts = [i for i in fronts if i !=[]]
            no_obj = 2
            fit_ = []
            sot_sol = []
            sot_sols = []
            vb = []
            rankk = [nan] * pop_size
            Cd_sol = []
            ds1 = []

            for qqs in range(pop_size):
                Cd_sol.append([])
            for tt in range(len(fronts)):
                r = len(fronts[tt])             # length of fronts
                for qq in range(r):
                    for qs in fronts[tt][qq]:
                        Cd_sol[qs] = 0                 # vector  containing crowding distance
                    
                no_obj = 2                         # numbers of objective in the problem
                for n in range(no_obj):
                    fit_.append([])
                bb1 = 0
                for m in range(no_obj):
                    for j in range(r):
                        ee2 = fronts[tt][j][0]
                        fit_[m].append(fff[m][ee2])
                    fit_c = copy.copy(fit_[m])              # fit_c is the copy of fit_
                    fit_sot = fit_[m].sort()  
                    sot_sol = []
                    for cc in range(r):
                        fitidx = fit_c.index(fit_[m][cc])
                        sot_sol.append(fronts[tt][fitidx][0])   # Fs = fronts   # sorted solutions(sot_sol)
                    sot_sols.append(sot_sol) 
                    indx_repp = [i for i, val in enumerate(fit_c) if val in fit_c[:i]]  #### repeated index
                    indx_repp1 = [y for y, val in enumerate(fit_[m]) if val in fit_[m][:y]] 
                    for u in range(len(indx_repp)):
                        if fit_[m][u] == fit_[m][u+1]:
                            new11 = fronts[tt][[indx_repp][0][u]]
                            sot_sol.insert(indx_repp1[u], new11[0])
                            del sot_sol[indx_repp1[u]+1]
                    
                    Cd_sol[sot_sol[0]]=Cd_sol[sot_sol[-1]] = float('inf')              # assigning large values to  the  distances of the extreme soluton
                    bb1 = bb1 +1
                    for w in range(1,r-1):
                        if max(fit_[m]) == min(fit_[m]):
                            Cd_sol[w] = Cd_sol[w]
                        else:
                            Cd_sol[sot_sol[w]] = Cd_sol[sot_sol[w]] + ((abs(fit_[m][w+1]) - fit_[m][w -1])/((max(fit_[m]) - min(fit_[m]))))   # computing crowding distance for solution between the two extrem solutions            
                        
                fit_ = []
            return Cd_sol,sot_sols
        crwdist = cdist(fronts,pop_size,fff)
        Cd_sol = crwdist[0]
        sot_sols = crwdist[1]

        def ranking(sol1):
            rank = 5000
            for w in range(len(fronts)):
                for v in range(len(fronts[w])):
                    if sol1 == fronts[w][v][0]:
                        rank = w    
            return rank


        def constraint(I,pop):
            pop[I]
            b = 0
            w = [0] * numb_constr
            #constraint 1
            ce = 355 - pop[I][0] - pop[I][1] - pop[I][2] 
            z1 = pop[I][0] + pop[I][1] + pop[I][2] + ce  - 355
            if z1 < 0 or z1 > 0:
                w[b] = abs(z1)
                b = b + 1
            else:
                w[b] = 0
                b = b + 1
            #constriant 2
            z2 = (pop[I][1]/(0.02 * pop[I][0]))
            if z2 < 0:
                w[b] = abs(z1)
                b = b + 1
            else:
                w[b] = 0
                b = b + 1
            
            Omega = sum(w)
            return Omega


        def tornament(pop_size,ranking,Cd_sol):
            sols = list(range(len(Cd_sol)))
            random.shuffle(sols)
            mating_pool = []
            b = 0
            for n in range(pop_size-1):
                candidate0 = sols[n]
                candidate1 = sols[n+1]
                R_0 = ranking(candidate0)
                R_1 = ranking(candidate1)
                C_1 = constraint(candidate0,pop)
                C_2 = constraint(candidate1,pop)
                win = min(R_0, R_1)
                if C_1 == 0 and C_2 != 0:
                    mating_pool.append(candidate0)
                elif C_1 != 0 and C_2 == 0:
                    mating_pool.append(candidate1)
                elif C_1 != 0 and C_2 != 0:
                    if C_1 < C_2:
                        mating_pool.append(candidate0)
                    else:
                        mating_pool.append(candidate1)
                elif C_1 == 0 and C_2 == 0: 
                    if R_0 < R_1 :
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
                        else :
                            W_0 == W_1
                            co = [candidate0,candidate1]
                            mating_pool.append(random.choice(co))  
            candidatex = sols[-1]
            candidatey = sols[0]
            R_x = ranking(candidatex)
            R_y = ranking(candidatey)
            C_x = constraint(candidatex,pop)
            C_y = constraint(candidatey,pop)
            win = min(R_x, R_y)
            if C_x == 0 and C_y != 0:
                mating_pool.append(candidatex)
            elif C_x != 0 and C_y == 0:
                mating_pool.append(candidatey)
            elif C_x != 0 and C_y != 0:
                if C_x < C_y:
                    mating_pool.append(candidatex)
                else:
                    mating_pool.append(candidatey)
            elif C_x == 0 and C_y == 0: 
                if R_x < R_y :
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
                    else :
                        W_x == W_y
                        co = [candidatex,candidatey]
                        mating_pool.append(random.choice(co))
            return mating_pool
        winner = tornament(pop_size,ranking,Cd_sol)
        pp = []
        for e in range(len(winner)):
            pp.append(pop[winner[e]])


        def crossover(pp, crossover_rate):
            palen = len(pp)    # lenght of parent
            iidx = list(range(pop_size))
            random.shuffle(iidx)
            pp =  [pp[i] for i in iidx]
            cofs = []
            u = []
            betas = []   
            for i in range(0,pop_size,2):   
                O1 = []
                O2 = []
                if random.random()  < crossover_rate:
                    for y in range(len(bounds)):
                        ux = random.random()
                        if ux <= 0.5:
                            beta = (2 * ux)**(1/(ita_c+1))
                        else:
                            beta = (1/(2*(1-ux)))**(1/(ita_c+1))
                        bn1 = 0.5 * ((1 + beta) * pp[i][y]) +(1 - beta) * pp[i+1][y]
                        bn2 = 0.5 * ((1 - beta) * pp[i][y]) +(1 + beta) * pp[i+1][y]
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
        for w in range(pop_size):           # checking for bound voilation
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
        Uh1_1 =  fitnessiffso1, fitnessvalue1 
        fff1 = [Uh ,Uh1]
        combinfronting = fronting(pop_size*2,Uh ,Uh1)
        combcdist = cdist(combinfronting ,pop_size*2,fff1)
        Cd_sol1 = combcdist[0]
        sot_sols1 = combcdist[1]


        def check(listt, val):
            bn = 0
            for xc in range(len(listt)):
                if val ==  listt[xc]:
                    bn = bn +1
                if bn == len(listt):
                    return True
            return False


        def makedub(xi):
            xh =  [[t] for t in xi]
            return xh
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
                solcmdist = []          # crowding distance of the solutions in front[3]
                for m in range(len(combinfronting[d])):
                    solcmdist.append(combcdist[0][combinfronting[d][m][0]]) 
                che1 = check(solcmdist, float('inf')) # Checking if all crowd distance is infinity
                if che1 == True:
                    sunn = pop_size-L2
                    flt = [i for sublist in combinfronting[d] for i in sublist]   # flated combinfronting[3]
                    if sunn > len(flt):
                        sunn = len(flt)
                    sel = np.random.choice(flt, size = sunn, replace = False) # sel --- selecting solution to complete population
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
                    indx_rep = [i for i, val in enumerate(copi_solcmdist) if val in copi_solcmdist[:i]]  #### repeated index
                    indx_rep1 = [y for y, val in enumerate(solcmdist) if val in solcmdist[:y]] 
                    for u in range(len(indx_rep)):
                        if solcmdist[u] == solcmdist[u+1]:
                            new1 = combinfronting[d][indx_rep[u]][0]
                            sort_solu.insert(indx_rep1[u], new1)
                            del sort_solu[indx_rep1[u]+1]
                    xx = sort_solu[0:(pop_size- L2)]
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
                indx_rep = [i for i, val in enumerate(copi_solcmdist) if val in copi_solcmdist[:i]]  #### repeated index
                indx_rep1 = [y for y, val in enumerate(solcmdist) if val in solcmdist[:y]] 
                for u in range(len(indx_rep)):
                    if solcmdist[u] == solcmdist[u+1]:
                        new1 = combinfronting[d][indx_rep[u]][0]
                        sort_solu.insert(indx_rep1[u], new1)
                        del sort_solu[indx_rep1[u]+1]
                xx = sort_solu[0: pop_size]
                xx = makedub(xx) 
                nxt_gen.append(xx)
    ##################################################################### new codes ##########################
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
    ##################################################################################################################
        pop = nxt_gen_pop
        fff = []
        xx = fronting(pop_size,nxt_gen_fit1, nxt_gen_fit2)
        # socketio.emit('update', {'fit1': nxt_gen_fit1, 'fit2': nxt_gen_fit2})
        try:
            socketio.emit('update', {'nxt_gen_fit1': nxt_gen_fit1, 'nxt_gen_fit2': nxt_gen_fit2})
            print('Update event emitted successfully1')
        except Exception as e:
            print('Error emitting update event1:', str(e))
    print('pareto optimal front TCC',nxt_gen_fit1)
    print('pareto optimal front TEE',nxt_gen_fit2)
    print("optimal KF SF MK", pop)    
    # fig = plt.figure()
    # plt.rcParams['font.family'] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"]
    # plt.scatter(nxt_gen_fit1, nxt_gen_fit2, c = 'red')
    # fig.suptitle('Generation')
    # plt.xlabel('TCC')
    # plt.ylabel('TEE')
    # plt.show()
    return (nxt_gen_fit1, nxt_gen_fit2, pop)

# table =  [['KF', '13', '34'], ['SF', '12', '56'], ['MK', '5', '66']]
# table1 =  ['349.47 + (355 - KF - SF - MK) * 0.36 + (SF * 1.59) + (MK * 1.73) + (KF * 1.18)', '59.92 + (355 - KF - MK) * 0.9 + (SF * 0.064) + (MK * 0.33) + (KF * 0.02677)']
# def create_objective_function(variables, formula):
#     # Start constructing the function as a string
#     func_str = "def objective_function(I):\n"
#     # Extracting values from the list I
#     for i, var in enumerate(variables):
#         func_str += f"    {var} = I[{i}]\n"
#     # Add the formula calculation
#     func_str += f"    TCC = {formula}\n"
#     # Return the calculated TCC
#     func_str += "    return TCC\n"
#     # Create a local dictionary to execute the function definition
#     local_dict = {}
#     # Execute the constructed function definition
#     exec(func_str, globals(), local_dict)
#     # Return the dynamically created function
#     return local_dict['objective_function']

# nsga2(table, table1)

# from cmath import inf, nan
# from re import M
# from timeit import repeat
# from numpy.random import beta, rand, randint
# import numpy as np
# # import matplotlib.pyplot as plt
# import matplotlib
# from requests import delete
# from sympy import N

# # from nuw4 import L

# # from nuw1 import Cd_sol
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from numpy import append, array, exp, sqrt
# import tempfile
# import copy
# import math
# import random
# from mpl_toolkits.mplot3d import axes3d

def nsgaa2(table_data, table_data1):
    time.sleep(1)
    table_dataa = []
    for sublist in table_data: # Converting the search space from string to ing
        row = []
        for item in sublist:
            if item.isdigit():
                # Convert the string to an integer
                item = int(item)
            row.append(item)
        table_dataa.append(row)
    table_new = table_dataa
    dv = [] # retrieving only the decision variables
    for k in range(len(table_new)):
        dv.append(table_new[k][0])

    pop_size = 100
    # bounds = [[table_new[0][1], table_new[0][2]], [table_new[1][1], table_new[1][2]], [table_new[2][1],table_new[2][2]]]
    bounds = [[row[1], row[2]] for row in table_new]

    # pop_size = 300
    # bounds = [[0, 1], [0, 1],[0,1], [0,1], [0,1], [0,1],[0,1]]#, [0,1], [0,1], [0,1],[0, 1], [0, 1], [0,1], [0,1], [0,1], [0,1],[0,1], [0,1], [0,1], [0,1],[0, 1], [0, 1], [0,1], [0,1], [0,1], [0,1],[0,1], [0,1], [0,1], [0,1]]


    nv = len(bounds)
    # print(nv)

    iteration = 150



    crossover_rate = 1.0
    mutation_rate = 1/nv
    ita_c = 30
    ita_m = 20
    nv = 30



    ghk = []




    newpopfit = []


    pop_posi = []
    def posi(bounds, pop_size):
        for i in range(pop_size and len(bounds)):
            x = (bounds[i][0] + np.random.rand(pop_size)*(bounds[i][1]-bounds[i][0])).tolist()
            pop_posi.append(x)
            
        return pop_posi
    pops= posi(bounds, pop_size)
    # print(classpop)

    popss = list(zip(*pops))

    # print(classpopp)
    pop = [list(ele) for ele in popss]
    #pop = [[0.913, 2.181], [0.599, 2.450],[0.139, 1.157],[0.867, 1.505],[0.885, 1.239],[0.658,2.040],[0.788, 2.166],[0.342, 0.756]]
    #pop = [[0.913, 2.181], [0.599, 2.450],[0.139, 1.157],[0.867, 1.505],[0.885, 1.239],[0.658,2.040],[0.788, 2.166],[0.342, 0.756]]

    # def objective_function(I):
    #     x1 = I[0]
    #     objective_max = x1
    #     return objective_max
    # fitnessvalue  = [objective_function(h) for h in pop]

    # def objective_function1(I):
    #     x1 = I[0]
    #     x2 = I[1]

    #     objective_max = 1 + x2 - (x1)**2
    #     return objective_max

    # fitnessvalue1  = [objective_function1(h) for h in pop]

    objective_function = create_objective_function(dv, table_data1[0])
    objective_function1 = create_objective_function(dv, table_data1[1])
    objective_function2 = create_objective_function(dv, table_data1[2])

    fitnessvalue = [objective_function(h) for h in pop]
    fitnessvalue1 = [objective_function1(h) for h in pop]
    fitnessvalue2 = [objective_function2(h) for h in pop]

    nxt_gen_fit1 = [objective_function(h) for h in pop]
    nxt_gen_fit2 = [objective_function1(h) for h in pop]
    nxt_gen_fit3 = [objective_function2(h) for h in pop]
    # socketio.emit('update', {'fit1': fitnessvalue, 'fit2': fitnessvalue1})
    try:
        socketio.emit('update', {'nxt_gen_fit1': nxt_gen_fit1, 'nxt_gen_fit2': nxt_gen_fit2, 'nxt_gen_fit3': nxt_gen_fit3})
        print('Update event emitted successfully')
    except Exception as e:
        print('Error emitting update event:', str(e))


    ################################# plotintin initial population  ##################################
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.rcParams['font.family'] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"]

    # # Scatter plot in 3D
    # ax.scatter(fitnessvalue, fitnessvalue1, fitnessvalue2, c='blue')

    # ax.set_title('Evolution of the best chromosome')
    # ax.set_xlabel('COP')
    # ax.set_ylabel('SCP_ads')
    # ax.set_zlabel('CT')

    # plt.show()



    # plt.show()


    # fitnessvalue  = [objective_function(h) for h in pop]
    # fitnessvalue1  = [objective_function1(h) for h in pop]
    fff = [fitnessvalue ,fitnessvalue1, fitnessvalue2]
        # if i % 10 == 0 and i > 1:
        #     print(" iter = " + str(i) + " best fitness = %.3f" % min(bestfitness))
        


        ######## STAGE 1 (FAST NON DOMINATED SORTING) #################################################



    #     ############# CROWDINGIN DISTANCE ########################################

    ita_count = 0 
    for zz in range(iteration):
        fitnessvalue  = [objective_function(h) for h in pop]
        fitnessvalue1  = [objective_function1(h) for h in pop]
        fitnessvalue2  = [objective_function2(h) for h in pop]
        fff = [fitnessvalue, fitnessvalue1, fitnessvalue2]

        
        def fronting(pop_size,fitnessvalue ,fitnessvalue1, fitnessvalue2): #### DOMINANT DEPTH METHOD
            Fs = []
            for v in range(pop_size):
                Fs.append([])

            Sps = []                   # Sps is the vector containing all solutions being dominated by a particular solution
            for g in range(pop_size):
                Sps.append([])

            nxs = []                   # nxs is the vector containing the numbers of solution dominating a particular solution
            # for g in range(pop_size):
            #     nxs.append([])

            
                
                
            for i in range(pop_size):
                nx = 0
                
        
                for j in range(pop_size):
                    if i != j:
                        if (fitnessvalue[i] <= fitnessvalue[j] and fitnessvalue1[i] <= fitnessvalue1[j] and fitnessvalue2[i] <= fitnessvalue2[j]): # if fitnessvalue[i] dominates fitnessvalue[j]
                            Sps[i].append(j)
                        elif (fitnessvalue[i] < fitnessvalue[j] or fitnessvalue1[i] < fitnessvalue1[j] or fitnessvalue2[i] < fitnessvalue2[j]): # if fitnessvalue[i] is non dominated by fitnessvalue[j]
                            continue
                        else:
                            nx = nx + 1      # nx else nx is incremented by 1 indicating the element is dominated by one element
                nxs.append(nx)             # nxs-  number of solution that dominates fitenessvalue[i] 
                if nx == 0:                         # if nx equal to zero meaning the solution is not dominated by any solution and they belong to the first front or front 1
                    P_rank = 0                     # is  the front one ranking
                    Fs[P_rank].append([i])         # Fs is the matrix containing the fronts of all solutin


        

                




            ######## STAGE 2 (FAST NON DOMINATED SORTING)###########

            b = 0
            bb = 0
            q = []
            while Fs[b] != []:
                Q = []
                for w in range(len(Fs[b])):
                    ee = Fs[b][w][bb]  #extracting the solutions from the first front

                    q.append(Sps[ee])   #extracting the solutions of the Sps of the first front solutions
            
                q = [item for sublist in q for item in sublist] # FLATENED q
                
                for g in range(len(q)):
                    ee1 =  q[g]                     #extracting the solutions from list q
                    nxs[ee1]= (nxs[ee1]) - 1    # the solutions of list q has values in nxs , so extracting it and substracting 1 as per the algorigthm 
                    if nxs[ee1] == 0 : 
                        q_rank = b + 1
                        Q.append([ee1])
                b = b +1
                q = []
                Fs[b] = Q
            return Fs
        fronts = fronting(pop_size,fitnessvalue ,fitnessvalue1, fitnessvalue2)
    # print(fronts)
    ################## END OF (FAST NON DOMINATED SORTING) THIS HELPS I CONVERGANCE OF THE SOLUTION ###############################

        def cdist(fronts, pop_size, fff):
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
                        ee2 = fronts[tt][j][0] #extracting the solutions in the fronts
                        fit_[m].append(fff[m][ee2]) #extracting the fitness values of the solutions

                    sot_sol = []
                
                    for cc in range(r):
                        fitidx = fit_[m].index(sorted(fit_[m])[cc])
                        sot_sol.append(fronts[tt][fitidx][0])
                    fit_[m] = sorted(fit_[m])

                    Cd_sol[sot_sol[0]] = Cd_sol[sot_sol[-1]] = float('inf') # Assigning a large value (inf) to the extrem solutions

                    for w in range(1, r - 1):
                        if max(fit_[m]) == min(fit_[m]):
                            pass # Cd_sol[sot_sol[w]] = 0
                        else:
                            Cd_sol[sot_sol[w]] += abs(fit_[m][w + 1] - fit_[m][w - 1]) / (max(fit_[m]) - min(fit_[m]))  #computing the crowding distance

                    sot_sols.append(sot_sol)
                fit_ = []

            return Cd_sol, sot_sols

        crwdist = cdist(fronts,pop_size,fff)
        Cd_sol = crwdist[0]
        # print('Cd_sol',Cd_sol)
        sot_sols = crwdist[1]
        # print('sot_sols',sot_sols)

        def ranking(sol1):
            rank = 5000
            for w in range(len(fronts)):
                for v in range(len(fronts[w])):
                    if sol1 == fronts[w][v][0]:
                        rank = w
            return rank

        
        def tornament(pop_size,ranking,Cd_sol):
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
                
                if R_0 < R_1 :
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
                    else :
                        W_0 == W_1
                        co = [candidate0,candidate1]
                        mating_pool.append(random.choice(co))
                
            candidatex = sols[-1]
            candidatey = sols[0]
            R_x = ranking(candidatex)
            R_y = ranking(candidatey)
            win = min(R_x, R_y)
                
            if R_x < R_y :
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
                else :
                    W_x == W_y
                    co = [candidatex,candidatey]
                    mating_pool.append(random.choice(co))
            return mating_pool
        winner = tornament(pop_size,ranking,Cd_sol)
        # winnerz = []
        # for e in range(len(winner)):
        #     winnerz.append(fitnessvalue1 [winner[e]])
        pp = []
        for e in range(len(winner)):
            pp.append(pop[winner[e]])




        def crossover(pp, crossover_rate):
            palen = len(pp)    # lenght of parent
            iidx = list(range(pop_size))
            random.shuffle(iidx)
            pp =  [pp[i] for i in iidx]
            cofs = []
            u = []
            betas = []
            
            for i in range(0, pop_size, 2):   
                O1 = []
                O2 = []
                if random.random()  < crossover_rate:
                    for y in range(len(bounds)):

                        ux = random.random()
                        if ux <= 0.5:
                            beta = (2 * ux)**(1/(ita_c+1))
                        else:
                            beta = (1/(2*(1-ux)))**(1/(ita_c+1))

                        bn1 = 0.5 * ((1 + beta) * pp[i][y]) +(1 - beta) * pp[i+1][y]
                        bn2 = 0.5 * ((1 - beta) * pp[i][y]) +(1 + beta) * pp[i+1][y]

                        O1.append(bn1)
                        O2.append(bn2)
                    cofs.append(O1)
                    cofs.append(O2)
                else:
                    cofs.append(pp[i])
                    cofs.append(pp[i+1])
            return cofs
        nin = crossover(pp, crossover_rate)
        # print('nin', nin)
        for k in range(pop_size):
            for j in range(len(bounds)):
                if nin[k][j] < bounds[j][0]:
                    nin[k][j] = bounds[j][0]

                if nin[k][j] > bounds[j][1]:
                    nin[k][j] = bounds[j][1]
        
        def mutation(nin, mutation_rate):
            deltax = []
            u1 = []
            Oms = []
            
            for i in range(len(nin)):
                OO1 = []
                # nin[i]
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
        # print('fistoff', firstoffsrping )

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
        Uh1_1 =  fitnessiffso1, fitnessvalue1 
        Uh2_2 =  fitnessiffso2, fitnessvalue2 
        fff1 = [Uh ,Uh1, Uh2]

        combinfronting = fronting(pop_size*2, Uh, Uh1, Uh2)
        # tty = len(combinfronting[0])
        # tty1 = len(combinfronting)
        # combinfronting = [element for sub in combinfronting for element in sub]
        combcdist = cdist(combinfronting ,pop_size*2,fff1)
        Cd_sol1 = combcdist[0]
        sot_sols1 = combcdist[1]
        def check(listt, val):
            bn = 0
            for xc in range(len(listt)):
                if val ==  listt[xc]:
                    bn = bn +1
                if bn == len(listt):
                    return True
            return False
        def makedub(xi):
            xh =  [[t] for t in xi]
            return xh
        # # nxt_gensize == pop_size
        # val = float('inf')
        # listtt = [float('inf'), float('inf') ]
        # gggg = check(listtt, val)
        nxt_gen = []
        for d in range(len(combinfronting)):

            
            L1 = len(combinfronting[d])
            L2 = sum([len(k) for k in nxt_gen])
            if L1 <= pop_size - L2:
                # combinfronting[d]= [i for sublist in combinfronting[d] for i in sublist]
                nxt_gen.append(combinfronting[d])
                # nxt_gen = [i for sublist in nxt_gen for i in sublist]
                
                L2 = sum([len(k) for k in nxt_gen])
                if L2 < pop_size:
                    continue
                else:
                    L2 == pop_size
                    break
            else:
                L1 > pop_size - L2
                solcmdist = []          # crowding distance of the solutions in front[3]
                for m in range(len(combinfronting[d])):
                    solcmdist.append(combcdist[0][combinfronting[d][m][0]]) 
                che1 = check(solcmdist, float('inf')) # Checking if all crowd distance is infinity
                if che1 == True:
                    sunn = pop_size-L2
                    flt = [i for sublist in combinfronting[d] for i in sublist]   # flated combinfronting[3]
                    if sunn > len(flt):
                        sunn = len(flt)
                    sel = np.random.choice(flt, size = sunn, replace = False) # sel --- selecting solution to complete population
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
                    indx_rep = [i for i, val in enumerate(copi_solcmdist) if val in copi_solcmdist[:i]]  #### repeated index
                    indx_rep1 = [y for y, val in enumerate(solcmdist) if val in solcmdist[:y]] 
                    for u in range(len(indx_rep)):
                        if solcmdist[u] == solcmdist[u+1]:
                            new1 = combinfronting[d][indx_rep[u]][0]
                            sort_solu.insert(indx_rep1[u], new1)
                            del sort_solu[indx_rep1[u]+1]
                    
                    # sort_solu = list(sort_solu)
                    xx = sort_solu[0:(pop_size- L2)]
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
                indx_rep = [i for i, val in enumerate(copi_solcmdist) if val in copi_solcmdist[:i]]  #### repeated index
                indx_rep1 = [y for y, val in enumerate(solcmdist) if val in solcmdist[:y]] 
                for u in range(len(indx_rep)):
                    if solcmdist[u] == solcmdist[u+1]:
                        new1 = combinfronting[d][indx_rep[u]][0]
                        sort_solu.insert(indx_rep1[u], new1)
                        del sort_solu[indx_rep1[u]+1]
                
                # sort_solu = list(sort_solu)
                xx = sort_solu[0: pop_size]
                xx = makedub(xx)
                    
                nxt_gen.append(xx)
    ##################################################################### new codes ##########################
    
        
    
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
        
    
    
        # hn = fronting(pop_size,nxt_gen_fit1 ,nxt_gen_fit2)

    ##################################################################################################################
        
    
        # fffv = [nxt_gen_fit1,nxt_gen_fit2]
        
        pop = nxt_gen_pop
        fff = []
        # fronts[0]
        # print(fronts[0])
        # fffv = [nxt_gen_fit1,nxt_gen_fit2]
        # print(fffv)

        xx = fronting(pop_size,nxt_gen_fit1, nxt_gen_fit2, nxt_gen_fit3)
        try:
            socketio.emit('update', {'nxt_gen_fit1': nxt_gen_fit1, 'nxt_gen_fit2': nxt_gen_fit2, 'nxt_gen_fit3': nxt_gen_fit3})
            print('Update event emitted successfully')
        except Exception as e:
            print('Error emitting update event:', str(e))
        # print('xx3', xx[0])
        # print('xx', xx)
        ita_count = ita_count + 1
        print('ita_count', ita_count)

    print('nxt_gen_fit1',nxt_gen_fit1)
    print('nxt_gen_fit2', nxt_gen_fit2) 
    print('nxt_gen_fit3', nxt_gen_fit3)

def create_objective_function(variables, formula):
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

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.rcParams['font.family'] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]

# # Scatter plot in 3D
# ax.scatter(nxt_gen_fit1, nxt_gen_fit2, nxt_gen_fit3, c = 'purple')

# ax.set_title('Evolution of the best chromosome')
# ax.set_xlabel('COP')
# ax.set_ylabel('SCP_ads')
# ax.set_zlabel('CT')

# plt.show()




























if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5006, debug=True)
    socketio.run(app, host='0.0.0.0', port=5006, debug=True)