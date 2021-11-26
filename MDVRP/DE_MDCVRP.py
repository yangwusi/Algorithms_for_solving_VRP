# -*- coding: utf-8 -*-
# @Time    : 2021/9/10 14:54
# @Author  : Praise
# @File    : DE_MDCVRP.py
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
        self.sol_list=[]
        self.demand_dict={}
        self.depot_dict={}
        self.demand_id_list = []
        self.distance_matrix = {}
        self.opt_type=0
        self.vehicle_cap=0
        self.Cr=0.5
        self.F=0.5
        self.popsize=4*len(self.demand_id_list)

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

def genInitialSol(model):
    demand_id_list=copy.deepcopy(model.demand_id_list)
    for i in range(model.popsize):
        seed=int(random.randint(0,10))
        random.seed(seed)
        random.shuffle(demand_id_list)
        sol=Sol()
        sol.node_id_list=copy.deepcopy(demand_id_list)
        sol.obj,sol.routes=calObj(sol.node_id_list,model)
        model.sol_list.append(sol)
        if sol.obj<model.best_sol.obj:
            model.best_sol=copy.deepcopy(sol)

def adjustRoutes(demand_id_list,model):
    all_node_list=copy.deepcopy(model.demand_id_list)
    repeat_node=[]
    for id,node_id in enumerate(demand_id_list):
        if node_id in all_node_list:
            all_node_list.remove(node_id)
        else:
            repeat_node.append(id)
    for i in range(len(repeat_node)):
        demand_id_list[repeat_node[i]]=all_node_list[i]
    return demand_id_list

# DE/rand/1/bin
def muSol(model,v1):
    x1=model.sol_list[v1].node_id_list
    while True:
        v2=random.randint(0,len(model.demand_id_list)-1)
        if v2!=v1:
            break
    while True:
        v3=random.randint(0,len(model.demand_id_list)-1)
        if v3!=v2 and v3!=v1:
            break
    x2=model.sol_list[v2].node_id_list
    x3=model.sol_list[v3].node_id_list
    mu_x=[min(int(x1[i]+model.F*(x2[i]-x3[i])),len(model.demand_id_list)-1) for i in range(len(model.demand_id_list)) ]
    return mu_x

def crossSol(model,vx,vy):
    cro_x=[]
    for i in range(len(model.demand_id_list)):
        if random.random()<model.Cr:
            cro_x.append(vy[i])
        else:
            cro_x.append(vx[i])
    cro_x=adjustRoutes(cro_x,model)
    return cro_x

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

def run(demand_file,depot_file,epochs,Cr,F,popsize,v_cap,opt_type):
    """
    :param demand_file: Demand file path
    :param depot_file: Depot file path
    :param epochs: Iterations
    :param Cr: Differential crossover probability
    :param F: Scaling factor
    :param popsize: Population size
    :param v_cap: Vehicle capacity
    :param opt_type: Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.Cr=Cr
    model.F=F
    model.popsize=popsize
    model.opt_type=opt_type

    readCsvFile(demand_file,depot_file,model)
    calDistance(model)
    best_sol = Sol()
    best_sol.obj = float('inf')
    model.best_sol = best_sol
    genInitialSol(model)
    history_best_obj = []
    for ep in range(epochs):
        for i in range(popsize):
            v1=random.randint(0,len(model.demand_id_list)-1)
            sol=model.sol_list[v1]
            mu_x=muSol(model,v1)
            u=crossSol(model,sol.node_id_list,mu_x)
            u_obj,u_routes=calObj(u,model)
            if u_obj<=sol.obj:
                sol.node_id_list=copy.deepcopy(u)
                sol.obj=copy.deepcopy(u_obj)
                sol.routes=copy.deepcopy(u_routes)
                if sol.obj<model.best_sol.obj:
                    model.best_sol=copy.deepcopy(sol)
            history_best_obj.append(model.best_sol.obj)
        print("%s/%s， best obj: %s" % (ep, epochs, model.best_sol.obj))
    plotObj(history_best_obj)
    plotRoutes(model)
    outPut(model)

if __name__ == '__main__':
    demand_file = '../datasets/MDCVRP/demand.csv'
    depot_file = '../datasets/MDCVRP/depot.csv'
    run(demand_file,depot_file, epochs=150, Cr=0.5,F=0.5, popsize=400, v_cap=80,opt_type=1)