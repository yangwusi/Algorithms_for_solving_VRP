# -*- coding: utf-8 -*-
# @Time    : 2021/11/10 8:54
# @Author  : Praise
# @File    : DE_MDVRPTW.py
# obj:
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
import csv
import sys
class Sol():
    def __init__(self):
        self.obj=None
        self.node_id_list=[]
        self.cost_of_distance=None
        self.cost_of_time=None
        self.route_list=[]
        self.timetable_list=[]

class Node():
    def __init__(self):
        self.id=0
        self.x_coord=0
        self.y_cooord=0
        self.demand=0
        self.depot_capacity=0
        self.start_time=0
        self.end_time=1440
        self.service_time=0

class Model():
    def __init__(self):
        self.best_sol=None
        self.sol_list=[]
        self.demand_dict={}
        self.depot_dict={}
        self.depot_id_list=[]
        self.demand_id_list=[]
        self.distance_matrix={}
        self.time_matrix={}
        self.number_of_demands=0
        self.vehicle_cap=0
        self.vehicle_speed=1
        self.opt_type=1
        self.Cr=0.5
        self.F=0.5
        self.popsize=4*len(self.demand_id_list)

def readCSVFile(demand_file,depot_file,model):
    with open(demand_file,'r') as f:
        demand_reader=csv.DictReader(f)
        for row in demand_reader:
            node = Node()
            node.id = int(row['id'])
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.demand = float(row['demand'])
            node.start_time=float(row['start_time'])
            node.end_time=float(row['end_time'])
            node.service_time=float(row['service_time'])
            model.demand_dict[node.id] = node
            model.demand_id_list.append(node.id)
        model.number_of_demands=len(model.demand_id_list)

    with open(depot_file, 'r') as f:
        depot_reader = csv.DictReader(f)
        for row in depot_reader:
            node = Node()
            node.id = row['id']
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.depot_capacity = float(row['capacity'])
            node.start_time=float(row['start_time'])
            node.end_time=float(row['end_time'])
            model.depot_dict[node.id] = node
            model.depot_id_list.append(node.id)

def calDistanceTimeMatrix(model):
    for i in range(len(model.demand_id_list)):
        from_node_id = model.demand_id_list[i]
        for j in range(i + 1, len(model.demand_id_list)):
            to_node_id = model.demand_id_list[j]
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - model.demand_dict[to_node_id].x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - model.demand_dict[to_node_id].y_coord) ** 2)
            model.distance_matrix[from_node_id, to_node_id] = dist
            model.distance_matrix[to_node_id, from_node_id] = dist
            model.time_matrix[from_node_id,to_node_id] = math.ceil(dist/model.vehicle_speed)
            model.time_matrix[to_node_id,from_node_id] = math.ceil(dist/model.vehicle_speed)
        for _, depot in model.depot_dict.items():
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - depot.x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - depot.y_coord) ** 2)
            model.distance_matrix[from_node_id, depot.id] = dist
            model.distance_matrix[depot.id, from_node_id] = dist
            model.time_matrix[from_node_id,depot.id] = math.ceil(dist/model.vehicle_speed)
            model.time_matrix[depot.id,from_node_id] = math.ceil(dist/model.vehicle_speed)

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
        sys.exit(0)
    route.insert(0,index)
    route.append(index)
    depot_dict[index].depot_capacity=depot_dict[index].depot_capacity-1
    return route,depot_dict

def calTravelCost(route_list,model):
    timetable_list=[]
    cost_of_distance=0
    cost_of_time=0
    for route in route_list:
        timetable=[]
        for i in range(len(route)):
            if i == 0:
                depot_id=route[i]
                next_node_id=route[i+1]
                travel_time=model.time_matrix[depot_id,next_node_id]
                departure=max(0,model.demand_dict[next_node_id].start_time-travel_time)
                timetable.append((departure,departure))
            elif 1<= i <= len(route)-2:
                last_node_id=route[i-1]
                current_node_id=route[i]
                current_node = model.demand_dict[current_node_id]
                travel_time=model.time_matrix[last_node_id,current_node_id]
                arrival=max(timetable[-1][1]+travel_time,current_node.start_time)
                departure=arrival+current_node.service_time
                timetable.append((arrival,departure))
                cost_of_distance += model.distance_matrix[last_node_id, current_node_id]
                cost_of_time += model.time_matrix[last_node_id, current_node_id]+ current_node.service_time\
                                + max(current_node.start_time - timetable[-1][1] + travel_time, 0)
            else:
                last_node_id = route[i - 1]
                depot_id=route[i]
                travel_time = model.time_matrix[last_node_id,depot_id]
                departure = timetable[-1][1]+travel_time
                timetable.append((departure,departure))
                cost_of_distance +=model.distance_matrix[last_node_id,depot_id]
                cost_of_time+=model.time_matrix[last_node_id,depot_id]
        timetable_list.append(timetable)
    return timetable_list,cost_of_time,cost_of_distance

def extractRoutes(node_id_list,Pred,model):
    depot_dict=copy.deepcopy(model.depot_dict)
    route_list = []
    route = []
    label = Pred[node_id_list[0]]
    for node_id in node_id_list:
        if Pred[node_id] == label:
            route.append(node_id)
        else:
            route, depot_dict=selectDepot(route,depot_dict,model)
            route_list.append(route)
            route = [node_id]
            label = Pred[node_id]
    route, depot_dict = selectDepot(route, depot_dict, model)
    route_list.append(route)
    return route_list

def splitRoutes(node_id_list,model):
    depot=model.depot_id_list[0]
    V={id:float('inf') for id in model.demand_id_list}
    V[depot]=0
    Pred={id:depot for id in model.demand_id_list}
    for i in range(len(node_id_list)):
        n_1=node_id_list[i]
        demand=0
        departure=0
        j=i
        cost=0
        while True:
            n_2 = node_id_list[j]
            demand = demand + model.demand_dict[n_2].demand
            if n_1 == n_2:
                arrival= max(model.demand_dict[n_2].start_time,model.depot_dict[depot].start_time+model.time_matrix[depot,n_2])
                departure=arrival+model.demand_dict[n_2].service_time+model.time_matrix[n_2,depot]
                if model.opt_type == 0:
                    cost=model.distance_matrix[depot,n_2]*2
                else:
                    cost=model.time_matrix[depot,n_2]*2
            else:
                n_3=node_id_list[j-1]
                arrival= max(departure-model.time_matrix[n_3,depot]+model.time_matrix[n_3,n_2],model.demand_dict[n_2].start_time)
                departure=arrival+model.demand_dict[n_2].service_time+model.time_matrix[n_2,depot]
                if model.opt_type == 0:
                    cost=cost-model.distance_matrix[n_3,depot]+model.distance_matrix[n_3,n_2]+model.distance_matrix[n_2,depot]
                else:
                    cost=cost-model.time_matrix[n_3,depot]+model.time_matrix[n_3,n_2]\
                         +model.demand_dict[n_2].start_time-departure-model.time_matrix[n_3,depot]+model.time_matrix[n_3,n_2]\
                         +model.time_matrix[n_2,depot]
            if demand<=model.vehicle_cap and departure-model.time_matrix[n_2,depot] <= model.demand_dict[n_2].end_time:
                if departure <= model.depot_dict[depot].end_time:
                    n_4=node_id_list[i-1] if i-1>=0 else depot
                    if V[n_4]+cost <= V[n_2]:
                        V[n_2]=V[n_4]+cost
                        Pred[n_2]=i-1
                    j=j+1
            else:
                break
            if j==len(node_id_list):
                break
    route_list= extractRoutes(node_id_list,Pred,model)
    return len(route_list),route_list

def calObj(sol,model):

    node_id_list=copy.deepcopy(sol.node_id_list)
    num_vehicle, sol.route_list = splitRoutes(node_id_list, model)
    # travel cost
    sol.timetable_list,sol.cost_of_time,sol.cost_of_distance =calTravelCost(sol.route_list,model)
    if model.opt_type == 0:
        sol.obj=sol.cost_of_distance
    else:
        sol.obj=sol.cost_of_time

def genInitialSol(model):
    demand_id_list=copy.deepcopy(model.demand_id_list)
    for i in range(model.popsize):
        seed=int(random.randint(0,10))
        random.seed(seed)
        random.shuffle(demand_id_list)
        sol=Sol()
        sol.node_id_list=copy.deepcopy(demand_id_list)
        calObj(sol,model)
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
    worksheet.write(0, 0, 'cost_of_time')
    worksheet.write(0, 1, 'cost_of_distance')
    worksheet.write(0, 2, 'opt_type')
    worksheet.write(0, 3, 'obj')
    worksheet.write(1,0,model.best_sol.cost_of_time)
    worksheet.write(1,1,model.best_sol.cost_of_distance)
    worksheet.write(1,2,model.opt_type)
    worksheet.write(1,3,model.best_sol.obj)
    worksheet.write(2,0,'vehicleID')
    worksheet.write(2,1,'route')
    worksheet.write(2,2,'timetable')
    for row,route in enumerate(model.best_sol.route_list):
        worksheet.write(row+3,0,'v'+str(row+1))
        r=[str(i)for i in route]
        worksheet.write(row+3,1, '-'.join(r))
        r=[str(i)for i in model.best_sol.timetable_list[row]]
        worksheet.write(row+3,2, '-'.join(r))
    work.close()

def plotRoutes(model):
    for route in model.best_sol.route_list:
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

def run(demand_file,depot_file,epochs,Cr,F,popsize,v_cap,v_speed,opt_type):
    """
    :param demand_file: demand file path
    :param depot_file: depot file path
    :param epochs: Iterations
    :param Cr: Differential crossover probability
    :param F: Scaling factor
    :param popsize: Population size
    :param v_cap: Vehicle capacity
    :param v_speed: Vehicle free speed
    :param opt_type: Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.vehicle_speed=v_speed
    model.Cr=Cr
    model.F=F
    model.popsize=popsize
    model.opt_type=opt_type

    readCSVFile(demand_file,depot_file,model)
    calDistanceTimeMatrix(model)
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
            new_sol=Sol()
            new_sol.node_id_list=u
            calObj(new_sol,model)
            if new_sol.obj<=sol.obj:
                sol=copy.deepcopy(new_sol)
                if sol.obj<model.best_sol.obj:
                    model.best_sol=copy.deepcopy(sol)
            history_best_obj.append(model.best_sol.obj)
        print("%s/%s， best obj: %s" % (ep, epochs, model.best_sol.obj))
    plotObj(history_best_obj)
    plotRoutes(model)
    outPut(model)

if __name__ == '__main__':
    demand_file='../datasets/MDVRPTW/demand.csv'
    depot_file='../datasets/MDVRPTW/depot.csv'
    run(demand_file,depot_file, epochs=150, Cr=0.6,F=0.5, popsize=400, v_cap=80,v_speed=1, opt_type=0)

