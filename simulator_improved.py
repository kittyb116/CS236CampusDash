from __future__ import annotations
import heapq
import itertools
from typing import Any, Optional, Tuple
import random

from parse_files import read_dashers, read_tasklog
# this is the hungarian algorithm, which requires NumPy package
from munkres import Munkres
import munkres

"""WeightedGraph class """
class WeightedGraph: 
    def __init__(self):
        self.adjList_edges = {}
    
    def addNode(self,node):
        if node not in self.adjList_edges:
            self.adjList_edges[node] = []

    def addEdge(self,node1, node2, weight):
        if node1 in self.adjList_edges and node2 in self.adjList_edges:
            self.adjList_edges[node1].append((node2, weight))
    def modifyWeight(self,node1, node2, weight):
        if (node1 not in self.adjList_edges or node2 not in self.adjList_edges):
            print("One of the nodes is not in the list")
        else:
            for i in range(len(self.adjList_edges[node1])):
                if (self.adjList_edges[node1][i][0] == node2):
                    self.adjList_edges[node1][i] = (node2,weight)
            for i in range(len(self.adjList_edges[node2])):
                if (self.adjList_edges[node2][i][0] == node1):
                    self.adjList_edges[node2][i] = (node1,weight)
    def getNeighbors(self,node):
        return self.adjList_edges.get(node, [])

    def getNodes(self):
        return self.adjList_edges.keys()
    def __str__(self):
        return f"{self.adjList_edges}"

    def dijkstra_shortest_path(self,start_node, end_node): 
        '''
        @param: start node, end node
        @return: a tuple containing a list of visited elements, total cost 
        '''
        if (start_node not in self.adjList_edges or end_node not in self.adjList_edges): 
            return ("One of the nodes does not exist... Must add first")
        if (start_node == end_node): 
            return ([start_node], 0)
        
        distances = {}
        for node in self.adjList_edges.keys():
            distances[node] = float('inf')

        distances[start_node] = 0
        
        pred = {} #predecessor that store the node we used to get to our current node
        for node in self.adjList_edges.keys():
            pred[node] = None
        visited = []
        pq = PriorityQueue()

        for node in self.adjList_edges.keys():
            pq.insert(node, distances[node]) #distances = priority
        
        while not pq.isEmpty():
            min_tuple = pq.extractMin()
            if min_tuple is None:
                break
            
            current_node, current_distance = min_tuple
            
            #avoids repetitions
            if current_node in visited:
                continue
            visited.append(current_node)

            if current_node == end_node:
                break

            neighbors = self.adjList_edges[current_node]

            for neighb, cost in neighbors:
                if neighb not in visited:
                    #calculates the new distances
                    newDist = distances[current_node] + cost
                    #updates the cost of current node 
                    if newDist < distances[neighb]:
                        distances[neighb] = newDist
                        pred[neighb] = current_node
                        pq.insert(neighb, newDist)
        if distances[end_node] == float('inf'):
            return([],float('inf'))
        path = []
        i = end_node
        # starts from the end node and iterates back to get the path
        while i is not None:
            path.insert(0,i)
            i = pred[i]
        return(path,distances[end_node])   
    
    def get_edge_weight(self, node1, node2):
        """
        Returns the base weight of the edge from node1 to node2.
        Returns None if the edge does not exist.
        """
        for neighbor, weight in self.adjList_edges.get(node1, []):
            if neighbor == node2:
                return weight
        return None



class PriorityQueue: 
    #elements = [(item,priority), (item2,priority2),....]
    def __init__(self):
        self.elements = []
    def heapifyUp(self,nodeIndex):
        if (nodeIndex > 0 ): 
            tempNode = self.elements[nodeIndex]
            parentIndex = int((nodeIndex-1)/2)
            parentNode = self.elements[parentIndex]
            if (parentNode[1] > tempNode[1]): 
                # Swap temp node with its parent
                temp = tempNode
                self.elements[nodeIndex] = parentNode
                self.elements[parentIndex] = temp
                self.heapifyUp(parentIndex)

    def heapifyDown(self, nodeIndex):
        """
        Moves a node down the heap to maintain min-heap property.
        A node is swapped with its smaller child until it's in the correct position.
        """
        smallest = nodeIndex
        leftChildIndex = 2 * nodeIndex + 1
        rightChildIndex = 2 * nodeIndex + 2
    
        # Check if left child exists and is smaller than current node
        if leftChildIndex < len(self.elements):
            if self.elements[leftChildIndex][1] < self.elements[smallest][1]:
                smallest = leftChildIndex
    
        # Check if right child exists and is smaller than current smallest
        if rightChildIndex < len(self.elements):
            if self.elements[rightChildIndex][1] < self.elements[smallest][1]:
                smallest = rightChildIndex
    
        # If smallest is not the current node, swap and continue heapifying
        if smallest != nodeIndex:
            self.elements[nodeIndex], self.elements[smallest] = \
                self.elements[smallest], self.elements[nodeIndex]
            self.heapifyDown(smallest)
    '''
    Adds an item to the queue with an associated priority.
    '''
    def insert(self,item, priority):
        self.elements.append((item,priority))
        self.heapifyUp(len(self.elements)-1) 
    '''
    Removes the item with the minimum priority from the queue and returns it. If the
    queue is empty, this can return None.
    '''
    def extractMin(self):
        if(self.isEmpty()):
            return
        min = self.elements[0]
        self.elements[0] = self.elements[len(self.elements)-1]
        del self.elements[-1]
        if (not self.isEmpty()):
            self.heapifyDown(0)
        return min      
    '''
    Finds the item in the queue and updates its priority to this
    new value provided,priority. You can assume that the new priority 
    will always be less than the item’s current priority. Note that 
    your code should place the item to its correct position after the change.'''
    def decreaseKey(self,item, priority):
        for i, (it, p) in enumerate(self.elements):
            if(it == item):
                self.elements[i] = (item, priority)
                self.heapifyUp(i)
                return

    '''
    Returns True if the priority queue is empty, and False otherwise.
    '''
    def isEmpty(self):
        return len(self.elements) == 0
    def __str__(self):
        return f"{self.elements}"

class Dasher:
    def __init__(self, start_location: Any, start_time: Any , end_time: Any, dasher_id):
        self.start_location = start_location
        self.start_time = start_time
        self.end_time = end_time
        self.available = True
        self.dasher_id = dasher_id
    
#Could be easier to create a task object to reference different attributes
class Task:
    def __init__(self, vertex_id, appear_time, target, reward, task_id):
        self.vertex_id=vertex_id
        self.appear_time=appear_time
        self.target_time= target
        self.reward = reward
        self.task_id = task_id

def read_graph(fname):
    # Open the file
    file = open(fname, "r")
    # Read the first line that contains the number of vertices
    # numVertices is the number of vertices in the graph (n)
    numVertices = file.readline()
    wg = WeightedGraph()

    # You might need to add some code here to set up your graph object
    # Next, read the edges and build the graph
    for line in file:
        # edge is a list of 3 indices representing a pair of adjacent vertices and the weight
        # edge[0] contains the first vertex (index between 0 and numVertices-1)
        # edge[1] contains the second vertex (index between 0 and numVertices-1)
        # edge[2] contains the weight of the edge (a positive integer)
        edge = line.strip().split(",")
    # Use the edge information to populate your graph object
    # TODO: Add your code here
        if (int(edge[0]) not in wg.getNodes()):
            wg.addNode(int(edge[0]))
        if (int(edge[1]) not in wg.getNodes()):
            wg.addNode(int(edge[1]))
        wg.addEdge(int(edge[0]),int(edge[1]),int(edge[2]))
    # Close the file safely after done reading
    file.close()
    return wg 


"""
simple_discrete_event_sim.py

A minimal discrete-event simulator where scheduled events carry an
event_id and optional payload. Event behavior is implemented by
overriding the Simulator.handle(event_id, payload) method using a
simple switch (if/elif) inside it.
"""
"""
you cant run sim for both algorithms at the same time..
two code files two classes for baseline and algorithm X
put flags in the code and decide which algorithm 
"""
class EventHandle:
    """Simple cancelable handle for a scheduled event."""
    __slots__ = ("_cancelled",)
    def __init__(self) -> None:
        self._cancelled = False
    def cancel(self) -> None:
        self._cancelled = True
    @property
    def cancelled(self) -> bool:
        return self._cancelled

class Simulator:
    """Minimal discrete-event simulator.
    Subclass and override handle(event_id, payload) with a switch-case.
    """
    def __init__(self, batch_time, start_time: float = 0.0) -> None:
        self.now = float(start_time)
        self._queue: list[Tuple[float, int, Any, Any, EventHandle]] = []
        self._seq = itertools.count()
        self._stopped = False
        self.events_processed = 0
        self.batch_time = batch_time

    def schedule_at(self, time: float, event_id: Any, payload: Any = None) -> EventHandle:
        if time < self.now:
            raise ValueError("Cannot schedule in the past")
        seq = next(self._seq)
        h = EventHandle()
        heapq.heappush(self._queue, (float(time), seq, event_id, payload, h))
        return h

    def _pop_next(self):
        while self._queue:
            time, seq, event_id, payload, h = heapq.heappop(self._queue)
            if not h.cancelled:
                return time, event_id, payload
            # skipped cancelled
        return None

    def step(self) -> bool:
        if self._stopped:
            return False
        item = self._pop_next()
        if item is None:
            return False
        time, event_id, payload = item
        self.now = time
        # dispatch to user-defined handler
        self.handle(event_id, payload)
        self.events_processed += 1
        return True

    def run(self, until: Optional[float] = None, max_events: Optional[int] = None) -> None:
        self._stopped = False
        processed = 0
        while not self._stopped:
            if not self._queue:
                break
            if until is not None and self._queue[0][0] > until:
                break
            if max_events is not None and processed >= max_events:
                break
            if not self.step():
                break
            processed += 1

    def stop(self) -> None:
        self._stopped = True

    def handle(self, event_id: Any, payload: Any) -> None:
        """Override in a subclass with a simple switch (if/elif) on event_id."""
        raise NotImplementedError("Override handle(event_id, payload)")
    
# simple simpulator that extends the main simulator
class SimpleSim(Simulator):
    def __init__(self, graph_file: str, batch_time: int):
        super().__init__(batch_time = batch_time)
        self.graph = read_graph(graph_file)
        self.available_tasks = []
        self.global_reward = 0
        self.task_completed = 0

    def _pop_next(self):
        dasher_available = []
        task_arrival = []
        within_time_cutoff = True
        while self._queue and within_time_cutoff:
            time_cutoff = self.now + self.batch_time
            time, seq, event_id, payload, h = self._queue[0]
            if time <= time_cutoff:
                heapq.heappop(self._queue)
                if event_id == 'dasher_arrival':
                    dasher_available.append((time, event_id, payload))
                elif event_id == 'task_arrival':
                    task_arrival.append((time, event_id, payload))
            else:
                within_time_cutoff = False
        return dasher_available, task_arrival
        
    def step(self) -> bool:
        if self._stopped:
            return False
        # get all the dasher and task arrivals within the batch time
        dashers, tasks = self._pop_next()

        # update the time
        self.now = self.now + self.batch_time

        # iterate through available task list just for available tasks
        avail_tasks = []
        for task in self.available_tasks:
            if self.now <= task.target_time:
                avail_tasks.append(task)
        self.available_tasks = avail_tasks

        # add tasks that will be available during the time period
        self.events_processed += len(tasks)
        for new_task in tasks:
            _, event_id, payload = new_task
            if self.now <= payload.target_time:
                self.handle(event_id, payload)

        # find all available dashers (exclude dashers that leave during this time)
        avail_dashers = []
        for dash_arrive in dashers:
            _, event_id, payload = dash_arrive
            dasher, loc = payload
            if self.now <= dasher.end_time:
                avail_dashers.append(dash_arrive)

        # find a matching for available tasks and drivers
        if len(avail_dashers) > 0 and len(self.available_tasks) > 0:
            matching, not_paired_dashers = self.hungarian_matching(avail_dashers)

            # handle the 'pairing' event
            self.handle('pairing', matching)
        else:
            not_paired_dashers = avail_dashers

        # reschedule the arrival of these dashers to the same idx in the future
        for dash in not_paired_dashers:
            _, event_id, payload = dash
            self.handle(event_id, payload)

        self.events_processed += len(dashers) + len(tasks)
        return True
        
    def hungarian_matching(self, available_dashers):
        # construct cost and distance matrix for later use
        cost_matrix = []
        distance_matrix = []
        for dash_event in available_dashers:
            _, event_id, payload = dash_event
            dasher, loc = payload
            rewards = []
            distance = []
            for task in self.available_tasks:
                _, time = self.graph.dijkstra_shortest_path(loc, task.vertex_id)
                if (task.target_time <= dasher.end_time) and (self.now+time <= task.target_time):
                    distance.append(time)
                    rewards.append(round(task.reward/(time+1), 2))
                else:
                    rewards.append(0)
                    distance.append(0)
            cost_matrix.append(rewards)
            distance_matrix.append(distance)

        # perform hungarian algorithm
        m = Munkres()
        hung_matrix = munkres.make_cost_matrix(cost_matrix)
        id_pairings = m.compute(hung_matrix)

        # check if the pairings are valid and eliminate any invalid pairings (if they have a cost of 0)
        pairings = []
        matched_dashers_idx = [False for _ in range(len(available_dashers))]
        for id_pair in id_pairings:
            dash_idx, task_idx = id_pair
            # since the matrix is automatically padded, making sure that valid tasks and dashers are selected
            if dash_idx < len(available_dashers) and task_idx < len(self.available_tasks):
                if cost_matrix[dash_idx][task_idx] != 0:
                    pairings.append((available_dashers[dash_idx], self.available_tasks[task_idx], distance_matrix[dash_idx][task_idx]))
                    matched_dashers_idx[dash_idx] = True
            
        # now just get unmatched dashers
        unavailable_dash = []
        for i in range(len(available_dashers)):
            if not matched_dashers_idx[i]:
                unavailable_dash.append(available_dashers[i])
            
        return pairings, unavailable_dash

    def handle(self, event_id: str, payload: Any) -> None:
        # simple switch-case implemented with if/elif
        if event_id == "dasher_arrival": #Dasher arrives at a new node (event created anytime dasher moves to a node)
            self.handle_dasher_arrival(payload)
        elif event_id == "task_arrival": #Task appears in system 
            self.handle_task_arrival(payload)
        elif event_id == "pairing":
            self.handle_pairing(payload)
        elif event_id == "stop":
            print(f"[{self.now:.3f}] stopping")
            self.stop()
        else:
            print(f"[{self.now:.3f}] unknown event {event_id!r} -> {payload}")
        
    """ This code has been modified just to reschedule the dasher to arrive at same vertex """
    def handle_dasher_arrival(self, payload):
        self.schedule_at(self.now + self.batch_time, 'dasher_arrival', payload)

    """task arrival"""
    def handle_task_arrival(self,payload):
        task = payload
        self.available_tasks.append(task)

    """ handles pairing event """
    def handle_pairing(self, payload):
        for pairs in payload:
            dash, task, time = pairs
            _, event_id, dash_info = dash
            dasher, _ = dash_info
            self.schedule_at(self.now + time, 'dasher_arrival', [dasher, task.vertex_id])
            self.global_reward += task.reward
            self.available_tasks.remove(task)
        self.task_completed += len(payload)


# Example usage with a simple switch-case style handler
if __name__ == "__main__":
# ------- Start Simulator ----------
    # relevant files
    dasher_fn = 'project_files/dashers_time_adjusted.csv'
    tasklog_fn = 'project_files/tasklog_time_adjusted.csv'
    
    dasher_info = read_dashers(dasher_fn)

    for i in range(1):
        sim = SimpleSim("project_files/grid100.txt", 3)
    
        tasklog_info = read_tasklog(tasklog_fn, (1,100))

        # create arrival and departure events along with objects 
        counter = 0
        for dasher in dasher_info:
            loc, start, end = dasher
            curr_dasher = Dasher(loc, start, end, counter)
            # create arrival and departure events already
            sim.schedule_at(start, 'dasher_arrival', [curr_dasher, loc])
            counter += 1

        counter = 0
        for tasklog in tasklog_info:
            loc, appear, end, reward = tasklog
            curr_task = Task(loc, appear, end, reward, counter)
            # create task arrival and departure event
            sim.schedule_at(appear, 'task_arrival', curr_task)
            counter += 1

        sim.run()

        print(f'Trial {i} for testing time 3')
        print("events processed:", sim.events_processed)
        print(f"Total reward:{sim.global_reward}")
        print(f"Total number of tasks completed: {sim.task_completed}")
        print()
