import pandas as pd
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
class Sol():
    def __init__(self):
        self.nodes_seq=None
        self.obj=None
        self.routes=None
class Node():
    def __init__(self):
        self.id=0
        self.name=''
        self.seq_no=0
        self.x_coord=0
        self.y_coord=0
        self.demand=0
class Model():
    def __init__(self):
        self.best_sol=None
        self.node_list=[]
        self.node_seq_no_list=[]
        self.depot=None
        self.number_of_nodes=0
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
def readXlsxFile(filepath,model):
    # It is recommended that the vehicle depot data be placed in the first line of xlsx file
    node_seq_no = -1#the depot node seq_no is -1,and demand node seq_no is 0,1,2,...
    df = pd.read_excel(filepath)
    for i in range(df.shape[0]):
        node=Node()
        node.id=node_seq_no
        node.seq_no=node_seq_no
        node.x_coord= df['x_coord'][i]
        node.y_coord= df['y_coord'][i]
        node.demand=df['demand'][i]
        if df['demand'][i] == 0:
            model.depot=node
        else:
            model.node_list.append(node)
            model.node_seq_no_list.append(node_seq_no)
        try:
            node.name=df['name'][i]
        except:
            pass
        try:
            node.id=df['id'][i]
        except:
            pass
        node_seq_no=node_seq_no+1
    model.number_of_nodes=len(model.node_list)
def initParam(model):
    for i in range(model.number_of_nodes):
        for j in range(i+1,model.number_of_nodes):
            d=math.sqrt((model.node_list[i].x_coord-model.node_list[j].x_coord)**2+
                        (model.node_list[i].y_coord-model.node_list[j].y_coord)**2)
            model.distance[i,j]=d
            model.distance[j,i]=d
def genInitialSol(node_seq):
    node_seq=copy.deepcopy(node_seq)
    random.seed(0)
    random.shuffle(node_seq)
    return node_seq
def splitRoutes(nodes_seq,model):
    num_vehicle = 0
    vehicle_routes = []
    route = []
    remained_cap = model.vehicle_cap
    for node_no in nodes_seq:
        if remained_cap - model.node_list[node_no].demand >= 0:
            route.append(node_no)
            remained_cap = remained_cap - model.node_list[node_no].demand
        else:
            vehicle_routes.append(route)
            route = [node_no]
            num_vehicle = num_vehicle + 1
            remained_cap =model.vehicle_cap - model.node_list[node_no].demand
    vehicle_routes.append(route)
    return num_vehicle,vehicle_routes
def calDistance(route,model):
    distance=0
    depot=model.depot
    for i in range(len(route)-1):
        distance+=model.distance[route[i],route[i+1]]
    first_node=model.node_list[route[0]]
    last_node=model.node_list[route[-1]]
    distance+=math.sqrt((depot.x_coord-first_node.x_coord)**2+(depot.y_coord-first_node.y_coord)**2)
    distance+=math.sqrt((depot.x_coord-last_node.x_coord)**2+(depot.y_coord - last_node.y_coord)**2)
    return distance
def calObj(nodes_seq,model):
    num_vehicle, vehicle_routes = splitRoutes(nodes_seq, model)
    if model.opt_type==0:
        return num_vehicle,vehicle_routes
    else:
        distance=0
        for route in vehicle_routes:
            distance+=calDistance(route,model)
        return distance,vehicle_routes

def createRandomDestory(model):
    d=random.uniform(model.rand_d_min,model.rand_d_max)
    reomve_list=random.sample(range(model.number_of_nodes),int(d*model.number_of_nodes))
    return reomve_list

def createWorseDestory(model,sol):
    deta_f=[]
    for node_no in sol.nodes_seq:
        nodes_seq_=copy.deepcopy(sol.nodes_seq)
        nodes_seq_.remove(node_no)
        obj,vehicle_routes=calObj(nodes_seq_,model)
        deta_f.append(sol.obj-obj)
    sorted_id = sorted(range(len(deta_f)), key=lambda k: deta_f[k], reverse=True)
    d=random.randint(model.worst_d_min,model.worst_d_max)
    remove_list=sorted_id[:d]
    return remove_list

def createRandomRepair(remove_list,model,sol):
    unassigned_nodes_seq=[]
    assigned_nodes_seq = []
    # remove node from current solution
    for i in range(model.number_of_nodes):
        if i in remove_list:
            unassigned_nodes_seq.append(sol.nodes_seq[i])
        else:
            assigned_nodes_seq.append(sol.nodes_seq[i])
    # insert
    for node_no in unassigned_nodes_seq:
        index=random.randint(0,len(assigned_nodes_seq)-1)
        assigned_nodes_seq.insert(index,node_no)
    new_sol=Sol()
    new_sol.nodes_seq=copy.deepcopy(assigned_nodes_seq)
    new_sol.obj,new_sol.routes=calObj(assigned_nodes_seq,model)
    return new_sol

def createGreedyRepair(remove_list,model,sol):
    unassigned_nodes_seq = []
    assigned_nodes_seq = []
    # remove node from current solution
    for i in range(model.number_of_nodes):
        if i in remove_list:
            unassigned_nodes_seq.append(sol.nodes_seq[i])
        else:
            assigned_nodes_seq.append(sol.nodes_seq[i])
    #insert
    while len(unassigned_nodes_seq)>0:
        insert_node_no,insert_index=findGreedyInsert(unassigned_nodes_seq,assigned_nodes_seq,model)
        assigned_nodes_seq.insert(insert_index,insert_node_no)
        unassigned_nodes_seq.remove(insert_node_no)
    new_sol=Sol()
    new_sol.nodes_seq=copy.deepcopy(assigned_nodes_seq)
    new_sol.obj,new_sol.routes=calObj(assigned_nodes_seq,model)
    return new_sol

def findGreedyInsert(unassigned_nodes_seq,assigned_nodes_seq,model):
    best_insert_node_no=None
    best_insert_index = None
    best_insert_cost = float('inf')
    assigned_nodes_seq_obj,_=calObj(assigned_nodes_seq,model)
    for node_no in unassigned_nodes_seq:
        for i in range(len(assigned_nodes_seq)):
            assigned_nodes_seq_ = copy.deepcopy(assigned_nodes_seq)
            assigned_nodes_seq_.insert(i, node_no)
            obj_, _ = calObj(assigned_nodes_seq_, model)
            deta_f = obj_ - assigned_nodes_seq_obj
            if deta_f<best_insert_cost:
                best_insert_index=i
                best_insert_node_no=node_no
                best_insert_cost=deta_f
    return best_insert_node_no,best_insert_index

def createRegretRepair(remove_list,model,sol):
    unassigned_nodes_seq = []
    assigned_nodes_seq = []
    # remove node from current solution
    for i in range(model.number_of_nodes):
        if i in remove_list:
            unassigned_nodes_seq.append(sol.nodes_seq[i])
        else:
            assigned_nodes_seq.append(sol.nodes_seq[i])
    # insert
    while len(unassigned_nodes_seq)>0:
        insert_node_no,insert_index=findRegretInsert(unassigned_nodes_seq,assigned_nodes_seq,model)
        assigned_nodes_seq.insert(insert_index,insert_node_no)
        unassigned_nodes_seq.remove(insert_node_no)
    new_sol = Sol()
    new_sol.nodes_seq = copy.deepcopy(assigned_nodes_seq)
    new_sol.obj, new_sol.routes = calObj(assigned_nodes_seq, model)
    return new_sol

def findRegretInsert(unassigned_nodes_seq,assigned_nodes_seq,model):
    opt_insert_node_no = None
    opt_insert_index = None
    opt_insert_cost = -float('inf')
    for node_no in unassigned_nodes_seq:
        n_insert_cost=np.zeros((len(assigned_nodes_seq),3))
        for i in range(len(assigned_nodes_seq)):
            assigned_nodes_seq_=copy.deepcopy(assigned_nodes_seq)
            assigned_nodes_seq_.insert(i,node_no)
            obj_,_=calObj(assigned_nodes_seq_,model)
            n_insert_cost[i,0]=node_no
            n_insert_cost[i,1]=i
            n_insert_cost[i,2]=obj_
        n_insert_cost= n_insert_cost[n_insert_cost[:, 2].argsort()]
        deta_f=0
        for i in range(1,model.regret_n):
            deta_f=deta_f+n_insert_cost[i,2]-n_insert_cost[0,2]
        if deta_f>opt_insert_cost:
            opt_insert_node_no = int(n_insert_cost[0, 0])
            opt_insert_index=int(n_insert_cost[0,1])
            opt_insert_cost=deta_f
    return opt_insert_node_no,opt_insert_index

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
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
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

def run(filepath,rand_d_max,rand_d_min,worst_d_min,worst_d_max,regret_n,r1,r2,r3,rho,phi,epochs,pu,v_cap,opt_type):
    """
    :param filepath: Xlsx file path
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
    readXlsxFile(filepath, model)
    initParam(model)
    history_best_obj = []
    sol = Sol()
    sol.nodes_seq = genInitialSol(model.node_seq_no_list)
    sol.obj, sol.routes = calObj(sol.nodes_seq, model)
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
    file = '../data/cvrp.xlsx'
    run(filepath=file,rand_d_max=0.4,rand_d_min=0.1,
        worst_d_min=5,worst_d_max=20,regret_n=5,r1=30,r2=20,r3=10,rho=0.4,
        phi=0.9,epochs=10,pu=5,v_cap=80,opt_type=1)

