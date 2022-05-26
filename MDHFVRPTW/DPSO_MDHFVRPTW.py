# -*- coding: utf-8 -*-
# @Time    : 2021/12/21 8:54
# @Author  : Praise
# @File    : DPSO_MDHFVRPTW.py
# obj:

import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
import csv
import time

class Sol():
    def __init__(self):
        self.obj=None
        self.node_id_list=[]
        self.distance_of_routes=None
        self.time_of_routes=None
        self.route_list=[]
        self.timetable_list=[]

class Demand():
    def __init__(self):
        self.id = 0
        self.x_coord = 0
        self.y_coord = 0
        self.demand = 0
        self.service_time=0
        self.start_time = 0
        self.end_time = 1440

class Vehicle():
    def __init__(self):
        self.depot_id=0
        self.x_coord=0
        self.y_coord=0
        self.type=0
        self.capacity=0
        self.free_speed=1
        self.fixed_cost=1.0
        self.variable_cost=1.0
        self.numbers=0
        self.start_time=0
        self.end_time=1440

class Model():
    def __init__(self):
        self.best_sol=None
        self.demand_dict={}
        self.vehicle_dict={}
        self.vehicle_type_list=[]
        self.demand_id_list=[]
        self.sol_list=[]
        self.distance_matrix={}
        self.number_of_demands=0
        self.popsize=100
        self.opt_type=1 # 0: cost of travel distance; 1: cost of travel time
        self.pl=[]
        self.pg=None
        self.v=[]
        self.Vmax=5
        self.w=0.8
        self.c1=2
        self.c2=2

def readCSVFile(demand_file,depot_file,model):
    with open(demand_file,'r') as f:
        demand_reader=csv.DictReader(f)
        for row in demand_reader:
            demand = Demand()
            demand.id = int(row['id'])
            demand.x_coord = float(row['x_coord'])
            demand.y_coord = float(row['y_coord'])
            demand.demand = float(row['demand'])
            demand.start_time=float(row['start_time'])
            demand.end_time=float(row['end_time'])
            demand.service_time=float(row['service_time'])
            model.demand_dict[demand.id] = demand
            model.demand_id_list.append(demand.id)
        model.number_of_demands=len(model.demand_id_list)

    with open(depot_file, 'r') as f:
        depot_reader = csv.DictReader(f)
        for row in depot_reader:
            vehicle = Vehicle()
            vehicle.depot_id = row['depot_id']
            vehicle.x_coord = float(row['x_coord'])
            vehicle.y_coord = float(row['y_coord'])
            vehicle.type = row['vehicle_type']
            vehicle.capacity=float(row['vehicle_capacity'])
            vehicle.free_speed=float(row['vehicle_speed'])
            vehicle.numbers=float(row['number_of_vehicle'])
            vehicle.fixed_cost=float(row['fixed_cost'])
            vehicle.variable_cost=float(row['variable_cost'])
            vehicle.start_time=float(row['start_time'])
            vehicle.end_time=float(row['end_time'])
            model.vehicle_dict[vehicle.type] = vehicle
            model.vehicle_type_list.append(vehicle.type)

def calDistanceMatrix(model):
    for i in range(len(model.demand_id_list)):
        from_node_id = model.demand_id_list[i]
        for j in range(i + 1, len(model.demand_id_list)):
            to_node_id = model.demand_id_list[j]
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - model.demand_dict[to_node_id].x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - model.demand_dict[to_node_id].y_coord) ** 2)
            model.distance_matrix[from_node_id, to_node_id] = dist
            model.distance_matrix[to_node_id, from_node_id] = dist
        for _, vehicle in model.vehicle_dict.items():
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - vehicle.x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - vehicle.y_coord) ** 2)
            model.distance_matrix[from_node_id, vehicle.type] = dist
            model.distance_matrix[vehicle.type, from_node_id] = dist

def checkTimeWindow(route,model,vehicle):
    timetable=[]
    departure=0
    for i in range(len(route)):
        if i == 0:
            next_node_id = route[i + 1]
            travel_time = int(model.distance_matrix[vehicle.type, next_node_id] /vehicle.free_speed)
            departure = max(0, model.demand_dict[next_node_id].start_time - travel_time)
            timetable.append((int(departure), int(departure)))
        elif 1 <= i <= len(route) - 2:
            last_node_id = route[i - 1]
            current_node_id = route[i]
            current_node = model.demand_dict[current_node_id]
            travel_time = int(model.distance_matrix[last_node_id, current_node_id] / vehicle.free_speed)
            arrival = max(timetable[-1][1] + travel_time, current_node.start_time)
            departure = arrival + current_node.service_time
            timetable.append((int(arrival), int(departure)))
            if departure > current_node.end_time:
                departure = float('inf')
                break
        else:
            last_node_id = route[i - 1]
            travel_time = int(model.distance_matrix[last_node_id, vehicle.type] / vehicle.free_speed)
            departure = timetable[-1][1] + travel_time
            timetable.append((int(departure), int(departure)))
    if departure<vehicle.end_time:
        return True
    else:
        return False

def checkResidualCapacity(residual_node_id_list,W,model):
    residual_fleet_capacity=0
    residual_demand = 0
    for node_id in residual_node_id_list:
        residual_demand+=model.demand_dict[node_id].demand
    for k,v_type in enumerate(model.vehicle_type_list):
        vehicle=model.vehicle_dict[v_type]
        residual_fleet_capacity+=(vehicle.numbers-W[k+4])*vehicle.capacity
    if residual_demand<=residual_fleet_capacity:
        return True
    else:
        return False

def updateNodeLabels(label_list,W,number_of_lables):
    new_label_list=[]
    if len(label_list)==0:
        number_of_lables += 1
        W[0] = number_of_lables
        new_label_list.append(W)
    else:
        for label in label_list:
            if W[3]<=label[3] and sum(W[4:])<=sum(label[4:]):
                if W not in new_label_list:
                    number_of_lables += 1
                    W[0] = number_of_lables
                    new_label_list.append(W)
            elif W[3]<=label[3] and sum(W[4:])>sum(label[4:]):
                new_label_list.append(label)
                if W not in new_label_list:
                    number_of_lables += 1
                    W[0] = number_of_lables
                    new_label_list.append(W)
            elif W[3]>label[3] and sum(W[4:])<sum(label[4:]):
                new_label_list.append(label)
                if W not in new_label_list:
                    number_of_lables += 1
                    W[0] = number_of_lables
                    new_label_list.append(W)
            elif W[3]>label[3] and sum(W[4:])>=sum(label[4:]):
                new_label_list.append(label)
    return new_label_list,number_of_lables

def extractRoutes(V,node_id_list,model):
    route_list = []
    min_obj=float('inf')
    pred_label_id=None
    v_type=None
    # search the min cost label of last node of the node_id_list
    for label in V[model.number_of_demands-1]:
        if label[3]<=min_obj:
            min_obj=label[3]
            pred_label_id=label[1]
            v_type=label[2]
    # generate routes by pred_label_id
    route=[node_id_list[-1]]
    indexs=list(range(0,model.number_of_demands))[::-1]
    start=1
    while pred_label_id!=1:
        for i in indexs[start:]:
            stop=False
            for label in V[i]:
                if label[0]==pred_label_id:
                    stop=True
                    pred_label_id=label[1]
                    start=i
                    v_type_=label[2]
                    break
            if not stop:
                route.insert(0,node_id_list[i])
            else:
                route.insert(0,v_type)
                route.append(v_type)
                route_list.append(route)
                route=[node_id_list[i]]
                v_type=v_type_
    route.insert(0,v_type)
    route.append(v_type)
    route_list.append(route)
    return route_list

def splitRoutes(node_id_list,model):
    """
    V: dict，key=id，value=[n1,n2,n3,n4,n5,....]
        id：the index of the element in node_id_list
        n1: order of the current label
        n2: the id of the previous label that generated the current label
        n3: the vehicle type corresponding to the current label
        n4: the cost of the current path corresponds to the optimization target. When the optimization target is travel time,
            only the travel time between nodes is considered here to simplify the calculation, and the waiting time is omitted
        n5-: the number of vehicles of each type used as of the current label
    """
    V={i:[] for i in model.demand_id_list}
    V[-1]=[[0]*(len(model.vehicle_type_list)+4)]
    V[-1][0][0]=1
    V[-1][0][1]=1
    number_of_lables=1
    for i in range(model.number_of_demands):
        n_1=node_id_list[i]
        j=i
        load=0
        distance={v_type:0 for v_type in model.vehicle_type_list}
        while True:
            n_2=node_id_list[j]
            load=load+model.demand_dict[n_2].demand
            stop = False
            for k,v_type in enumerate(model.vehicle_type_list):
                vehicle=model.vehicle_dict[v_type]
                if i == j:
                    distance[v_type]=model.distance_matrix[v_type,n_1]+model.distance_matrix[n_1,v_type]
                else:
                    n_3=node_id_list[j-1]
                    distance[v_type]=distance[v_type]-model.distance_matrix[n_3,v_type]+model.distance_matrix[n_3,n_2]\
                                     +model.distance_matrix[n_2,v_type]
                route=node_id_list[i:j+1]
                route.insert(0,v_type)
                route.append(v_type)
                if not checkTimeWindow(route,model,vehicle):
                    continue
                for id,label in enumerate(V[i-1]):
                    if load<=vehicle.capacity and label[k+4]<vehicle.numbers:
                        stop=True
                        if model.opt_type==0:
                            cost=vehicle.fixed_cost+distance[v_type]*vehicle.variable_cost
                        else:
                            cost=vehicle.fixed_cost+distance[v_type]/vehicle.free_speed*vehicle.variable_cost
                        W=copy.deepcopy(label)
                        W[1]=V[i-1][id][0]
                        W[2]=v_type
                        W[3]=W[3]+cost
                        W[k+4]=W[k+4]+1
                        if checkResidualCapacity(node_id_list[j+1:],W,model):
                            label_list,number_of_lables=updateNodeLabels(V[j],W,number_of_lables)
                            V[j]=label_list
            j+=1
            if j>=len(node_id_list) or stop==False:
                break
    if len(V[model.number_of_demands-1])>0:
        route_list=extractRoutes(V, node_id_list, model)
        return route_list
    else:
        print("Failed to split the node id list because of the insufficient capacity")
        return None

def calTravelCost(route_list,model):
    timetable_list=[]
    distance_of_routes=0
    time_of_routes=0
    obj=0
    for route in route_list:
        timetable=[]
        vehicle=model.vehicle_dict[route[0]]
        travel_distance=0
        travel_time=0
        v_type = route[0]
        free_speed=vehicle.free_speed
        fixed_cost=vehicle.fixed_cost
        variable_cost=vehicle.variable_cost
        for i in range(len(route)):
            if i == 0:
                next_node_id=route[i+1]
                travel_time_between_nodes=model.distance_matrix[v_type,next_node_id]/free_speed
                departure=max(0,model.demand_dict[next_node_id].start_time-travel_time_between_nodes)
                timetable.append((int(departure),int(departure)))
            elif 1<= i <= len(route)-2:
                last_node_id=route[i-1]
                current_node_id=route[i]
                current_node = model.demand_dict[current_node_id]
                travel_time_between_nodes=model.distance_matrix[last_node_id,current_node_id]/free_speed
                arrival=max(timetable[-1][1]+travel_time_between_nodes,current_node.start_time)
                departure=arrival+current_node.service_time
                timetable.append((int(arrival),int(departure)))
                travel_distance += model.distance_matrix[last_node_id, current_node_id]
                travel_time += model.distance_matrix[last_node_id, current_node_id]/free_speed+\
                                + max(current_node.start_time - arrival, 0)
            else:
                last_node_id = route[i - 1]
                travel_time_between_nodes = model.distance_matrix[last_node_id,v_type]/free_speed
                departure = timetable[-1][1]+travel_time_between_nodes
                timetable.append((int(departure),int(departure)))
                travel_distance += model.distance_matrix[last_node_id,v_type]
                travel_time += model.distance_matrix[last_node_id,v_type]/free_speed
        distance_of_routes+=travel_distance
        time_of_routes+=travel_time
        if model.opt_type==0:
            obj+=fixed_cost+travel_distance*variable_cost
        else:
            obj += fixed_cost + travel_time *variable_cost
        timetable_list.append(timetable)
    return timetable_list,time_of_routes,distance_of_routes,obj

def calObj(sol,model):
    # calculate travel distance and travel time
    ret = splitRoutes(sol.node_id_list, model)
    if ret is not None:
        sol.route_list = ret
        sol.timetable_list, sol.time_of_routes, sol.distance_of_routes, sol.obj = calTravelCost(sol.route_list, model)
    else:
        sol.obj = 10**7

def generateInitialSol(model):
    demand_id_list=copy.deepcopy(model.demand_id_list)
    best_sol=Sol()
    best_sol.obj=float('inf')
    for i in range(model.popsize):
        seed = int(random.randint(0, 10))
        random.seed(seed)
        random.shuffle(demand_id_list)
        sol=Sol()
        sol.node_id_list= copy.deepcopy(demand_id_list)
        calObj(sol,model)
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
        new_sol=Sol()
        new_sol.node_id_list=new_x
        calObj(new_sol,model)
        if new_sol.obj<model.pl[id].obj:
            model.pl[id]=copy.deepcopy(new_sol)
        if new_sol.obj<model.best_sol.obj:
            model.best_sol=copy.deepcopy(new_sol)
            model.pg=copy.deepcopy(new_x)
        model.sol_list[id]= copy.deepcopy(new_sol)

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

def plotRoutes(model):
    for route in model.best_sol.route_list:
        x_coord=[model.vehicle_dict[route[0]].x_coord]
        y_coord=[model.vehicle_dict[route[0]].y_coord]
        for node_id in route[1:-1]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        x_coord.append(model.vehicle_dict[route[-1]].x_coord)
        y_coord.append(model.vehicle_dict[route[-1]].y_coord)
        plt.grid()
        if route[0]=='v1':
            plt.plot(x_coord,y_coord,marker='o',color='black',linewidth=0.5,markersize=5)
        elif route[0]=='v2':
            plt.plot(x_coord,y_coord,marker='o',color='orange',linewidth=0.5,markersize=5)
        elif route[0]=='v3':
            plt.plot(x_coord,y_coord,marker='o',color='r',linewidth=0.5,markersize=5)
        else:
            plt.plot(x_coord, y_coord, marker='o', color='b', linewidth=0.5, markersize=5)
    plt.xlabel('x_coord')
    plt.ylabel('y_coord')
    plt.show()

def outPut(model):
    work=xlsxwriter.Workbook('result.xlsx')
    worksheet=work.add_worksheet()
    worksheet.write(0, 0, 'time_of_routes')
    worksheet.write(0, 1, 'distance_of_routes')
    worksheet.write(0, 2, 'opt_type')
    worksheet.write(0, 3, 'obj')
    worksheet.write(1,0,model.best_sol.time_of_routes)
    worksheet.write(1,1,model.best_sol.distance_of_routes)
    worksheet.write(1,2,model.opt_type)
    worksheet.write(1,3,model.best_sol.obj)
    worksheet.write(2, 0,'vehicleID')
    worksheet.write(2, 1,'depotID')
    worksheet.write(2, 2, 'vehicleType')
    worksheet.write(2, 3,'route')
    worksheet.write(2, 4,'timetable')
    for row,route in enumerate(model.best_sol.route_list):
        worksheet.write(row+3,0,str(row+1))
        depot_id=model.vehicle_dict[route[0]].depot_id
        worksheet.write(row+3,1,depot_id)
        worksheet.write(row+3,2,route[0])
        r=[str(i)for i in route]
        worksheet.write(row+3,3, '-'.join(r))
        r=[str(i)for i in model.best_sol.timetable_list[row]]
        worksheet.write(row+3,4, '-'.join(r))
    work.close()

def run(demand_file,depot_file,epochs,popsize,Vmax,opt_type,w,c1,c2):
    """
    :param demand_file: demand file path
    :param depot_file: depot file path
    :param epochs: Iteration
    :param popsize: Population size
    :param Vmax : Maximum moving speed of particles
    :param opt_type: Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :param w: nertia weight
    :param c1: Learning factor
    :param c2: Learning factor
    :return:
    """
    model=Model()
    model.Vmax=Vmax
    model.opt_type=opt_type
    model.w=w
    model.c1=c1
    model.c2=c2
    model.popsize=popsize
    readCSVFile(demand_file,depot_file,model)
    calDistanceMatrix(model)
    history_best_obj=[]
    generateInitialSol(model)
    history_best_obj.append(model.best_sol.obj)
    start_time=time.time()
    for ep in range(epochs):
        updatePosition(model)
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s， best obj: %s， runtime: %s" % (ep + 1, epochs, model.best_sol.obj, time.time() - start_time))
    plotObj(history_best_obj)
    plotRoutes(model)
    outPut(model)


if __name__ == '__main__':
    demand_file = './datasets/MDHVRPTW/demand.csv'
    depot_file = './datasets/MDHVRPTW/depot.csv'
    run(demand_file=demand_file,depot_file=depot_file,epochs=200,popsize=100,Vmax=2,opt_type=1,w=0.9,c1=1,c2=5)
