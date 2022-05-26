# -*- coding: utf-8 -*-
# @Time    : 2021/8/26 14:56
# @Author  : Praise
# @File    : DPSO_MDCVRP.py
# @Software: PyCharm
# obj :
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
        self.sol_list=[]
        self.best_sol=None
        self.demand_dict = {}
        self.depot_dict = {}
        self.demand_id_list = []
        self.distance_matrix={}
        self.opt_type=0
        self.vehicle_cap=0
        self.pl=[]
        self.pg=None
        self.v=[]
        self.Vmax=5
        self.w=0.8
        self.c1=2
        self.c2=2

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

def genInitialSol(model,popsize):
    demand_id_list=copy.deepcopy(model.demand_id_list)
    best_sol=Sol()
    best_sol.obj=float('inf')
    for i in range(popsize):
        seed = int(random.randint(0, 10))
        random.seed(seed)
        random.shuffle(demand_id_list)
        sol=Sol()
        sol.node_id_list= copy.deepcopy(demand_id_list)
        sol.obj,sol.routes=calObj(sol.node_id_list,model)
        model.sol_list.append(sol)
        model.v.append([model.Vmax]*len(model.demand_id_list))
        model.pl.append(sol)
        if sol.obj<best_sol.obj:
            best_sol=copy.deepcopy(sol)
    model.best_sol=best_sol
    model.pg=best_sol.node_id_list

def updatePosition(model):
    w=model.w
    c1=model.c1
    c2=model.c2
    pg = model.pg
    for id,sol in enumerate(model.sol_list):
        x=sol.node_id_list
        v=model.v[id]
        pl=model.pl[id].node_id_list
        r1=random.random()
        r2=random.random()
        new_v=[]
        for i in range(len(model.demand_id_list)):
            v_=w*v[i]+c1*r1*(pl[i]-x[i])+c2*r2*(pg[i]-x[i])
            if v_>0:
                new_v.append(min(v_,model.Vmax))
            else:
                new_v.append(max(v_,-model.Vmax))
        new_x=[min(int(x[i]+new_v[i]),len(model.demand_id_list)-1) for i in range(len(model.demand_id_list)) ]
        new_x=adjustRoutes(new_x,model)
        model.v[id]=new_v

        new_x_obj,new_x_routes=calObj(new_x,model)
        if new_x_obj<model.pl[id].obj:
            model.pl[id].obj=copy.deepcopy(new_x_obj)
            model.pl[id].node_id_list=new_x
            model.pl[id].routes=new_x_routes
        if new_x_obj<model.best_sol.obj:
            model.best_sol.obj=copy.deepcopy(new_x_obj)
            model.best_sol.node_id_list=copy.deepcopy(new_x)
            model.best_sol.routes=copy.deepcopy(new_x_routes)
            model.pg=copy.deepcopy(new_x)
        model.sol_list[id].node_id_list = copy.deepcopy(new_x)
        model.sol_list[id].obj = copy.deepcopy(new_x_obj)
        model.sol_list[id].routes = copy.deepcopy(new_x_routes)

def adjustRoutes(node_id_list,model):
    all_node_list=copy.deepcopy(model.demand_id_list)
    repeat_node=[]
    for id,node_id in enumerate(node_id_list):
        if node_id in all_node_list:
            all_node_list.remove(node_id)
        else:
            repeat_node.append(id)
    for i in range(len(repeat_node)):
        node_id_list[repeat_node[i]]=all_node_list[i]
    return node_id_list

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

def run(demand_file,depot_file,epochs,popsize,Vmax,v_cap,opt_type,w,c1,c2):
    """
    :param demand_file: demand file path
    :param depot_file: depot file path
    :param epochs: Iterations
    :param popsize: Population size
    :param v_cap: Vehicle capacity
    :param Vmax :Max speed
    :param opt_type: Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :param w: Inertia weight
    :param c1: Learning factor
    :param c2: Learning factors
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    model.w=w
    model.c1=c1
    model.c2=c2
    model.Vmax=Vmax
    readCsvFile(demand_file,depot_file,model)
    calDistance(model)
    history_best_obj=[]
    genInitialSol(model,popsize)
    history_best_obj.append(model.best_sol.obj)
    for ep in range(epochs):
        updatePosition(model)
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s: best obj: %s"%(ep,epochs,model.best_sol.obj))
    plotObj(history_best_obj)
    plotRoutes(model)
    outPut(model)

if __name__ == '__main__':
    demand_file='../datasets/MDCVRP/demand.csv'
    depot_file='../datasets/MDCVRP/depot.csv'
    run(demand_file=demand_file,depot_file=depot_file,epochs=100,popsize=150,Vmax=2,v_cap=70,opt_type=1,w=0.9,c1=1,c2=5)

