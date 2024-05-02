import numpy as np
import heapq
import torch
import random

#Dictionary that converts the string of the grade of the climb to a class index
grade_to_class = {'6B': 0, 
                '6B+': 1, 
                '6C': 2, 
                '6C+': 3,
                '7A': 4,
                '7A+': 5,
                '7B': 6,
                '7B+': 7,
                '7C': 8,
                '7C+': 9,
                '8A': 10,
                '8A+': 11,
                '8B': 12,
                '8B+': 13}

class problemGraph(object):
    def __init__(self, problem, holds):
        self.problem = problem #Keep a copy of the problem dictionary
        self.all_holds = np.array(problem['start'] + problem['mid'] + problem['end']) #Create an array of all the holds on the climb
        self.num_holds = len(self.all_holds) #Total number of holds on that climb
        self.hold_indices = np.arange(self.num_holds) #Have indices that refer to each hold on the climb
        self.holdsDict = holds #Input dictionary with information about the quality of the holds 
        
        self.init_graph() #Initialize that graph nodes and edges
        
    def init_graph(self):
        #Indentify the starting positions and ending positions
        if len(self.problem['start']) == 1:
            self.start = [(0,0)] #If there's one start hold, you need to start matched on it
        else:
            self.start = [(0,1),(1,0)] #If there are two start holds, you need to start with LH on one and RH on the other (2 options)
        
        if len(self.problem['end']) == 1:
            self.end = [(self.num_holds-1, self.num_holds-1)] #If there's one finish hold, you need to start matched on it
        else:
            self.end = [(self.num_holds-1, self.num_holds-2),(self.num_holds-2, self.num_holds-1)] #If there are two, you must end with the LH on one and RH on the other (2 options)

        #Initialize a graph as a dictionary
        self.graph = {}
        
        #Graph edges correspond to moves stored in a move list (eventually a numpy array). 
        #This allows vectorized computation of the values of the edges, while still using a dictionary to represent the graph
        move_list = []
        #In the graph, each edge is associated with the index of the move in the move list
        move_ID = 0
        
        #The climber has a left hand and a right hand that can be on any of the holds (including matched on the same hold)
        #Iterate through all the holds available on the climb with two nested loops
        for i in range(self.num_holds):
            for j in range(self.num_holds):
                #i is the hold index of the left hand (LH)
                #j is the hold index of the right hand (RH)
                
                #Get the position of the LH and RH holds
                LH_pos, RH_pos = self.all_holds[i], self.all_holds[j]
                #Get the hold information (difficulty and angle) for the LH and RH holds
                LH_hold = np.array([self.holdsDict[(LH_pos[0], LH_pos[1])]['LH'], self.holdsDict[(LH_pos[0], LH_pos[1])]['Angle']])
                RH_hold = np.array([self.holdsDict[(RH_pos[0], RH_pos[1])]['RH'], self.holdsDict[(RH_pos[0], RH_pos[1])]['Angle']])
                    
                state = np.concatenate((LH_pos, RH_pos))
                
                self.graph[(i,j)] = {'State': state,
                                       'Linked': {}}

                #Any LH/RH hold state is connected to every other state where you are moving only the left or only the right hand
                #Iterate through all the holds one more time
                for k in range(self.num_holds):
                    #First model moving the right hand, where the right hand cannot move to the same position it's currently in. 
                    if k != j:
                        #Left hand position and hold stays the same
                        LH_pos2, LH_hold2 = LH_pos, LH_hold
                        
                        #Get the new RH position and hold info
                        RH_pos2 = self.all_holds[k]
                        RH_hold2 = np.array([self.holdsDict[(RH_pos2[0], RH_pos2[1])]['RH'], self.holdsDict[(RH_pos2[0], RH_pos2[1])]['Angle']])
                        
                        #Add a edge between the position specified by (i,j) and (i,k) which returns the move ID of the move
                        self.graph[(i,j)]['Linked'][(i,k)] = move_ID
                        
                        #The move is described by an 18 element vector, as described in the readme. 
                        move = np.concatenate((LH_pos2-LH_pos, RH_pos2- RH_pos, LH_pos - RH_pos, LH_pos2 - RH_pos2, LH_hold, LH_hold2, RH_hold, RH_hold2, np.array([i==j, i==k])))
                        move_list += [[move]]
                        move_ID += 1
                        
                    #Then repeat the process for the left hand
                    if k != i:
                        #RH position and hold are the same
                        RH_pos2, RH_hold2 = RH_pos, RH_hold

                       #Get the new LH position and hold info
                        LH_pos2 = self.all_holds[k]
                        LH_hold2 = np.array([self.holdsDict[(LH_pos2[0], LH_pos2[1])]['LH'], self.holdsDict[(LH_pos2[0], LH_pos2[1])]['Angle']])
                        
                        #Add a edge between the position specified by (i,j) and (k,j) which returns the move ID of the move                    
                        self.graph[(i,j)]['Linked'][(k,j)] = move_ID
                            
                        #The move is described by an 18 element vector, as described in the readme. 
                        move = np.concatenate((LH_pos2-LH_pos, RH_pos2- RH_pos, LH_pos - RH_pos, LH_pos2 - RH_pos2, LH_hold, LH_hold2, RH_hold, RH_hold2, np.array([i==j, k==j])))
                        move_list += [[move]]
                        move_ID += 1
        
        self.move_list = np.array(move_list) #Make the list of moves into a numpy array
        self.move_list[:,:,9:16:2] = self.move_list[:,:,9:16:2]/10 #Rescale angles to be a smaller range more comparable to the other values
        self.move_list[:,:,8:16:2] = 9 - self.move_list[:,:,8:16:2] #Make hold difficulty have smaller values be easy and larger values be hard
        self.move_diff = np.zeros(len(self.move_list)) #Initialize an array that stores the difficulty score of each move
    
    def compute_edges(self, function):
        #Computes the difficulty of every move in the move_list using the input function. This function can change over time and is specified externally. 
        self.move_diff = function(self.move_list)

    def optimize_sequence(self):
        #Run the dijkstra algorithm on the graph once for each different starting position as the source node 
        seqs = []
        random.shuffle(self.start) #In the case that the left hand always seems easiest for the first move, this helps it learn to break that symmetry
        for start in self.start:
            seqs.append(dijkstra(self.graph, start, self.move_diff))
        
        best = 1e9 #Keep track of the lowest difficulty
        #For all the output sequences
        random.shuffle(self.end) #In the case that the left hand always seems easiest for the last move, this helps it learn to break that symmetry
        
        for seq in seqs:
            #Check all the possible end positions
            for end in self.end:
                if seq[end].d < best: #If the difficulty is less than the current best
                   best = seq[end].d #Update the best difficulty score
                   opt_seq = seq #The optimal sequence
                   best_end = end #And the optimal finish position
        
        #Go through the nodes in reverse order, starting at the end, to piece together the optimal sequence
        ideal_sequence = [best_end]
        parent = opt_seq[best_end].parent
        while parent is not None:
            ideal_sequence.append(parent)
            parent = opt_seq[parent].parent
        
        ideal_sequence = ideal_sequence[::-1] #Reverse ideal sequence to have it in an intuitive direction
        
        #Make an optimal sequence where it shows that state of the LH and RH from the start to the end
        optimal = []
        for el in ideal_sequence:
            optimal.append(self.graph[el]['State'])
            
        #Make a list of the optimal_moves (the 18 element vectors) associated with each of the moves from the determined sequence
        optimal_moves = []
        for i in range(len(ideal_sequence)-1):
            pos1 = ideal_sequence[i]
            pos2 = ideal_sequence[i+1]
            optimal_moves.append(self.move_list[self.graph[pos1]['Linked'][pos2],0])
            
        #Set the problemGraph optimal_seq and optimal_moves as numpy arrays
        self.optimal_seq = np.array(optimal)
        self.optimal_moves = np.array(optimal_moves)
    
class Node:
    #Simple node class used for the dijkstra algorithm
    def __init__(self):
      self.d = float('inf') #current distance from source node
      self.parent = None
      self.finished = False

def dijkstra(graph,source,value):
    #Implementation of the dijkstra algorithm for graphs with the format as specified in the problemGraph object
    #Modified from the implementation found here: https://builtin.com/software-engineering-perspectives/dijkstras-algorithm
    
    nodes={}
    for node in graph:
      nodes[node]=Node()
    nodes[source].d=0
    queue=[(0,source)] #priority queue

    while queue:
      d,node=heapq.heappop(queue)
      if nodes[node].finished:
          continue
      nodes[node].finished=True
      for neighbor in graph[node]['Linked']:
          if nodes[neighbor].finished:
              continue
          new_d=d+value[graph[node]['Linked'][neighbor]]
          if new_d<nodes[neighbor].d:
              nodes[neighbor].d=new_d
              nodes[neighbor].parent=node
              heapq.heappush(queue,(new_d,neighbor))
              
    return nodes
    
def printModelSummary(model):
    #Simple function to print a summary of a pytorch model including the number of trainable parameters
    print(model)

    num_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(f"Layer {name} has {torch.numel(param.data)} trainable parameters.")
            num_params += torch.numel(param.data)
    
    print("\nTotal numer of trainable parameters is %d." %num_params)
    
def create_mask(input_tensor, max_value = None):
    #Returns a mask used for sequences of variable length. 
    #If the input_tensor is [3, 3, 6], the output will be [[1,1,1,0,0,0],[1,1,1,0,0,0],[1,1,1,1,1,1]]
    
    if max_value is None:
        max_value = torch.max(input_tensor).int().item()
        
    output_tensor = torch.zeros(len(input_tensor), max_value)
    for i, value in enumerate(input_tensor.int()):
        output_tensor[i, :value] = 1
    return output_tensor