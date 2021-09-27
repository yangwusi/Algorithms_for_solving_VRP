# -*- coding: utf-8 -*-
# @Time    : 2021/9/12 17:58
# @Author  : Praise
# @File    : ALNS_MDCVRP.py
# obj:
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
import csv

class Sol():
    def __init__(self):
        self.node_id_list=None
        self.obj=None
        self.routes=None

class Node():
    def __init__(self):
        self.id=0
        self.x_coord=0
        self.y_coord=0
        self.demand=0
        self.depot_capacity=15

class Model():
    def __init__(self):
        self.best_sol=None
        self.demand_dict = {}
        self.depot_dict = {}
        self.demand_id_list = []
        self.distance_matrix = {}
        self.opt_type=0
        self.vehicle_cap=80
        self.distance = {}
        self.rand_d_max=0.4
        self.rand_d_min=0.1
        self.worst_d_max=5
        self.worst_d_min=20
        self.regret_n=5
        self.r1=30
        self.r2=18
        self.r3=12
        self.rho=0.6
        self.d_weight=np.ones(2)*10
        self.d_select=np.zeros(2)
        self.d_score=np.zeros(2)
        self.d_history_select=np.zeros(2)
        self.d_history_score=np.zeros(2)
        self.r_weight=np.ones(3)*10
        self.r_select=np.zeros(3)
        self.r_score=np.zeros(3)
        self.r_history_select = np.zeros(3)
        self.r_history_score = np.zeros(3)

def readCsvFile(demand_file,depot_file,model):
    with open(demand_file,'r') as f:
        demand_reader=csv.DictReader(f)
        for row in demand_reader:
            node = Node()
            node.id = int(row['id'])
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.demand = float(row['demand'])
            model.demand_dict[node.id] = node
            model.demand_id_list.append(node.id)

    with open(depot_file,'r') as f:
        depot_reader=csv.DictReader(f)
        for row in depot_reader:
            node = Node()
            node.id = row['id']
            node.x_coord=float(row['x_coord'])
            node.y_coord=float(row['y_coord'])
            node.depot_capacity=float(row['capacity'])
            model.depot_dict[node.id] = node

def calDistance(model):
    for i in range(len(model.demand_id_list)):
        from_node_id = model.demand_id_list[i]
        for j in range(i+1,len(model.demand_id_list)):
            to_node_id=model.demand_id_list[j]
            dist=math.sqrt( (model.demand_dict[from_node_id].x_coord-model.demand_dict[to_node_id].x_coord)**2
                            +(model.demand_dict[from_node_id].y_coord-model.demand_dict[to_node_id].y_coord)**2)
            model.distance_matrix[from_node_id,to_node_id]=dist
            model.distance_matrix[to_node_id,from_node_id]=dist
        for _,depot in model.depot_dict.items():
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - depot.x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord -depot.y_coord)**2)
            model.distance_matrix[from_node_id, depot.id] = dist
            model.distance_matrix[depot.id, from_node_id] = dist

def selectDepot(route,depot_dict,model):
    min_in_out_distance=float('inf')
    index=None
    for _,depot in depot_dict.items():
        if depot.depot_capacity>0:
            in_out_distance=model.distance_matrix[depot.id,route[0]]+model.distance_matrix[route[-1],depot.id]
            if in_out_distance<min_in_out_distance:
                index=depot.id
                min_in_out_distance=in_out_distance
    if index is None:
        print("there is no vehicle to dispatch")
    route.insert(0,index)
    route.append(index)
    depot_dict[index].depot_capacity=depot_dict[index].depot_capacity-1
    return route,depot_dict

def splitRoutes(node_id_list,model):
    num_vehicle = 0
    vehicle_routes = []
    route = []
    remained_cap = model.vehicle_cap
    depot_dict=copy.deepcopy(model.depot_dict)
    for node_id in node_id_list:
        if remained_cap - model.demand_dict[node_id].demand >= 0:
            route.append(node_id)
            remained_cap = remained_cap - model.demand_dict[node_id].demand
        else:
            route,depot_dict=selectDepot(route,depot_dict,model)
            vehicle_routes.append(route)
            route = [node_id]
            num_vehicle = num_vehicle + 1
            remained_cap =model.vehicle_cap - model.demand_dict[node_id].demand

    route, depot_dict = selectDepot(route, depot_dict, model)
    vehicle_routes.append(route)

    return num_vehicle,vehicle_routes

def calRouteDistance(route,model):
    distance=0
    for i in range(len(route)-1):
        from_node=route[i]
        to_node=route[i+1]
        distance +=model.distance_matrix[from_node,to_node]
    return distance

def calObj(node_id_list,model):
    num_vehicle, vehicle_routes = splitRoutes(node_id_list, model)
    if model.opt_type==0:
        return num_vehicle,vehicle_routes
    else:
        distance = 0
        for route in vehicle_routes:
            distance += calRouteDistance(route, model)
        return distance,vehicle_routes

def genInitialSol(node_id_list):
    node_id_list=copy.deepcopy(node_id_list)
    random.seed(0)
    random.shuffle(node_id_list)
    return node_id_list

def createRandomDestory(model):
    d=random.uniform(model.rand_d_min,model.rand_d_max)
    reomve_list=random.sample(range(len(model.demand_id_list)),int(d*len(model.demand_id_list)))
    return reomve_list

def createWorseDestory(model,sol):
    deta_f=[]
    for node_id in sol.node_id_list:
        nodes_id_=copy.deepcopy(sol.node_id_list)
        nodes_id_.remove(node_id)
        obj,vehicle_routes=calObj(nodes_id_,model)
        deta_f.append(sol.obj-obj)
    sorted_id = sorted(range(len(deta_f)), key=lambda k: deta_f[k], reverse=True)
    d=random.randint(model.worst_d_min,model.worst_d_max)
    remove_list=sorted_id[:d]
    return remove_list

def createRandomRepair(remove_list,model,sol):
    unassigned_nodes_id=[]
    assigned_nodes_id = []
    # remove node from current solution
    for i in range(len(model.demand_id_list)):
        if i in remove_list:
            unassigned_nodes_id.append(sol.node_id_list[i])
        else:
            assigned_nodes_id.append(sol.node_id_list[i])
    # insert
    for node_id in unassigned_nodes_id:
        index=random.randint(0,len(assigned_nodes_id)-1)
        assigned_nodes_id.insert(index,node_id)
    new_sol=Sol()
    new_sol.node_id_list=copy.deepcopy(assigned_nodes_id)
    new_sol.obj,new_sol.routes=calObj(assigned_nodes_id,model)
    return new_sol

def findGreedyInsert(unassigned_nodes_id,assigned_nodes_id,model):
    best_insert_node_id=None
    best_insert_index = None
    best_insert_cost = float('inf')
    assigned_nodes_id_obj,_=calObj(assigned_nodes_id,model)
    for node_id in unassigned_nodes_id:
        for i in range(len(assigned_nodes_id)):
            assigned_nodes_id_ = copy.deepcopy(assigned_nodes_id)
            assigned_nodes_id_.insert(i, node_id)
            obj_, _ = calObj(assigned_nodes_id_, model)
            deta_f = obj_ - assigned_nodes_id_obj
            if deta_f<best_insert_cost:
                best_insert_index=i
                best_insert_node_id=node_id
                best_insert_cost=deta_f
    return best_insert_node_id,best_insert_index

def createGreedyRepair(remove_list,model,sol):
    unassigned_nodes_id = []
    assigned_nodes_id = []
    # remove node from current solution
    for i in range(len(model.demand_id_list)):
        if i in remove_list:
            unassigned_nodes_id.append(sol.node_id_list[i])
        else:
            assigned_nodes_id.append(sol.node_id_list[i])
    #insert
    while len(unassigned_nodes_id)>0:
        insert_node_id,insert_index=findGreedyInsert(unassigned_nodes_id,assigned_nodes_id,model)
        assigned_nodes_id.insert(insert_index,insert_node_id)
        unassigned_nodes_id.remove(insert_node_id)
    new_sol=Sol()
    new_sol.node_id_list=copy.deepcopy(assigned_nodes_id)
    new_sol.obj,new_sol.routes=calObj(assigned_nodes_id,model)
    return new_sol

def findRegretInsert(unassigned_nodes_id,assigned_nodes_id,model):
    opt_insert_node_id = None
    opt_insert_index = None
    opt_insert_cost = -float('inf')
    for node_id in unassigned_nodes_id:
        n_insert_cost=np.zeros((len(assigned_nodes_id),3))
        for i in range(len(assigned_nodes_id)):
            assigned_nodes_id_=copy.deepcopy(assigned_nodes_id)
            assigned_nodes_id_.insert(i,node_id)
            obj_,_=calObj(assigned_nodes_id_,model)
            n_insert_cost[i,0]=node_id
            n_insert_cost[i,1]=i
            n_insert_cost[i,2]=obj_
        n_insert_cost= n_insert_cost[n_insert_cost[:, 2].argsort()]
        deta_f=0
        for i in range(1,model.regret_n):
            deta_f=deta_f+n_insert_cost[i,2]-n_insert_cost[0,2]
        if deta_f>opt_insert_cost:
            opt_insert_node_id = int(n_insert_cost[0, 0])
            opt_insert_index=int(n_insert_cost[0,1])
            opt_insert_cost=deta_f
    return opt_insert_node_id,opt_insert_index

def createRegretRepair(remove_list,model,sol):
    unassigned_nodes_id = []
    assigned_nodes_id = []
    # remove node from current solution
    for i in range(len(model.demand_id_list)):
        if i in remove_list:
            unassigned_nodes_id.append(sol.node_id_list[i])
        else:
            assigned_nodes_id.append(sol.node_id_list[i])
    # insert
    while len(unassigned_nodes_id)>0:
        insert_node_id,insert_index=findRegretInsert(unassigned_nodes_id,assigned_nodes_id,model)
        assigned_nodes_id.insert(insert_index,insert_node_id)
        unassigned_nodes_id.remove(insert_node_id)
    new_sol = Sol()
    new_sol.node_id_list = copy.deepcopy(assigned_nodes_id)
    new_sol.obj, new_sol.routes = calObj(assigned_nodes_id, model)
    return new_sol

def selectDestoryRepair(model):
    d_weight=model.d_weight
    d_cumsumprob = (d_weight / sum(d_weight)).cumsum()
    d_cumsumprob -= np.random.rand()
    destory_id= list(d_cumsumprob > 0).index(True)

    r_weight=model.r_weight
    r_cumsumprob = (r_weight / sum(r_weight)).cumsum()
    r_cumsumprob -= np.random.rand()
    repair_id = list(r_cumsumprob > 0).index(True)
    return destory_id,repair_id

def doDestory(destory_id,model,sol):
    if destory_id==0:
        reomve_list=createRandomDestory(model)
    else:
        reomve_list=createWorseDestory(model,sol)
    return reomve_list

def doRepair(repair_id,reomve_list,model,sol):
    if repair_id==0:
        new_sol=createRandomRepair(reomve_list,model,sol)
    elif repair_id==1:
        new_sol=createGreedyRepair(reomve_list,model,sol)
    else:
        new_sol=createRegretRepair(reomve_list,model,sol)
    return new_sol

def resetScore(model):

    model.d_select = np.zeros(2)
    model.d_score = np.zeros(2)

    model.r_select = np.zeros(3)
    model.r_score = np.zeros(3)

def updateWeight(model):

    for i in range(model.d_weight.shape[0]):
        if model.d_select[i]>0:
            model.d_weight[i]=model.d_weight[i]*(1-model.rho)+model.rho*model.d_score[i]/model.d_select[i]
        else:
            model.d_weight[i] = model.d_weight[i] * (1 - model.rho)
    for i in range(model.r_weight.shape[0]):
        if model.r_select[i]>0:
            model.r_weight[i]=model.r_weight[i]*(1-model.rho)+model.rho*model.r_score[i]/model.r_select[i]
        else:
            model.r_weight[i] = model.r_weight[i] * (1 - model.rho)
    model.d_history_select = model.d_history_select + model.d_select
    model.d_history_score = model.d_history_score + model.d_score
    model.r_history_select = model.r_history_select + model.r_select
    model.r_history_score = model.r_history_score + model.r_score

def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.show()

def outPut(model):
    work=xlsxwriter.Workbook('result.xlsx')
    worksheet=work.add_worksheet()
    worksheet.write(0,0,'opt_type')
    worksheet.write(1,0,'obj')
    if model.opt_type==0:
        worksheet.write(0,1,'number of vehicles')
    else:
        worksheet.write(0, 1, 'drive distance of vehicles')
    worksheet.write(1,1,model.best_sol.obj)
    for row,route in enumerate(model.best_sol.routes):
        worksheet.write(row+2,0,'v'+str(row+1))
        r=[str(i)for i in route]
        worksheet.write(row+2,1, '-'.join(r))
    work.close()

def plotRoutes(model):
    for route in model.best_sol.routes:
        x_coord=[model.depot_dict[route[0]].x_coord]
        y_coord=[model.depot_dict[route[0]].y_coord]
        for node_id in route[1:-1]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        x_coord.append(model.depot_dict[route[-1]].x_coord)
        y_coord.append(model.depot_dict[route[-1]].y_coord)
        plt.grid()
        if route[0]=='d1':
            plt.plot(x_coord,y_coord,marker='o',color='black',linewidth=0.5,markersize=5)
        elif route[0]=='d2':
            plt.plot(x_coord,y_coord,marker='o',color='orange',linewidth=0.5,markersize=5)
        else:
            plt.plot(x_coord,y_coord,marker='o',color='b',linewidth=0.5,markersize=5)
    plt.xlabel('x_coord')
    plt.ylabel('y_coord')
    plt.show()

def run(demand_file,depot_file,rand_d_max,rand_d_min,worst_d_min,worst_d_max,regret_n,r1,r2,r3,rho,phi,epochs,pu,v_cap,opt_type):
    """
    :param demand_file: demand file path
    :param depot_file: depot file path
    :param rand_d_max: max degree of random destruction
    :param rand_d_min: min degree of random destruction
    :param worst_d_max: max degree of worst destruction
    :param worst_d_min: min degree of worst destruction
    :param regret_n:  n next cheapest insertions
    :param r1: score if the new solution is the best one found so far.
    :param r2: score if the new solution improves the current solution.
    :param r3: score if the new solution does not improve the current solution, but is accepted.
    :param rho: reaction factor of action weight
    :param phi: the reduction factor of threshold
    :param epochs: Iterations
    :param pu: the frequency of weight adjustment
    :param v_cap: Vehicle capacity
    :param opt_type: Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :return:
    """
    model=Model()
    model.rand_d_max=rand_d_max
    model.rand_d_min=rand_d_min
    model.worst_d_min=worst_d_min
    model.worst_d_max=worst_d_max
    model.regret_n=regret_n
    model.r1=r1
    model.r2=r2
    model.r3=r3
    model.rho=rho
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    readCsvFile(demand_file,depot_file, model)
    calDistance(model)
    history_best_obj = []
    sol = Sol()
    sol.node_id_list = genInitialSol(model.demand_id_list)
    sol.obj, sol.routes = calObj(sol.node_id_list, model)
    model.best_sol = copy.deepcopy(sol)
    history_best_obj.append(sol.obj)
    for ep in range(epochs):
        T=sol.obj*0.2
        resetScore(model)
        for k in range(pu):
            destory_id,repair_id=selectDestoryRepair(model)
            model.d_select[destory_id]+=1
            model.r_select[repair_id]+=1
            reomve_list=doDestory(destory_id,model,sol)
            new_sol=doRepair(repair_id,reomve_list,model,sol)
            if new_sol.obj<sol.obj:
                sol=copy.deepcopy(new_sol)
                if new_sol.obj<model.best_sol.obj:
                    model.best_sol=copy.deepcopy(new_sol)
                    model.d_score[destory_id]+=model.r1
                    model.r_score[repair_id]+=model.r1
                else:
                    model.d_score[destory_id]+=model.r2
                    model.r_score[repair_id]+=model.r2
            elif new_sol.obj-sol.obj<T:
                sol=copy.deepcopy(new_sol)
                model.d_score[destory_id] += model.r3
                model.r_score[repair_id] += model.r3
            T=T*phi
            print("%s/%s:%s/%s， best obj: %s" % (ep,epochs,k,pu, model.best_sol.obj))
            history_best_obj.append(model.best_sol.obj)
        updateWeight(model)

    plotObj(history_best_obj)
    plotRoutes(model)
    outPut(model)
    print("random destory weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.d_weight[0],
                                                                        model.d_history_select[0],
                                                                        model.d_history_score[0]))
    print("worse destory weight is {:.3f}\tselect is {}\tscore is {:.3f} ".format(model.d_weight[1],
                                                                        model.d_history_select[1],
                                                                        model.d_history_score[1]))
    print("random repair weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.r_weight[0],
                                                                       model.r_history_select[0],
                                                                       model.r_history_score[0]))
    print("greedy repair weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.r_weight[1],
                                                                       model.r_history_select[1],
                                                                       model.r_history_score[1]))
    print("regret repair weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.r_weight[2],
                                                                       model.r_history_select[2],
                                                                       model.r_history_score[2]))

if __name__=='__main__':
    demand_file = '../datasets/MDCVRP/demand.csv'
    depot_file = '../datasets/MDCVRP/depot.csv'
    run(demand_file=demand_file,depot_file=depot_file,rand_d_max=0.4,rand_d_min=0.1,
        worst_d_min=5,worst_d_max=20,regret_n=5,r1=30,r2=20,r3=10,rho=0.4,
        phi=0.9,epochs=10,pu=5,v_cap=80,opt_type=1)