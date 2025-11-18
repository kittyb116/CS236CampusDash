from __future__ import annotations
import heapq
import itertools
from typing import Any, Optional, Tuple
from collections import Counter
from queue import Queue
from WeightedGraphImplementation import WeightedGraph
from numpy import random as np_ran

"""
simple_discrete_event_sim.py

A minimal discrete-event simulator where scheduled events carry an
event_id and optional payload. Event behavior is implemented by
overriding the Simulator.handle(event_id, payload) method using a
simple switch (if/elif) inside it.
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
    
class Cars:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.path = []
        self.total_cost = 0

        # I just added this attribute to make it easy to implement Dijkstra
        self.planned_paths = Queue()
    
    def add_path(self, node, time, cost):
        self.path.append((node, time))
        self.total_cost += cost
    
    def set_predetermined_path(self, path_list):
        for edge in path_list:
            self.planned_paths.put(edge)

    # just need to modify this part to get the next node
    def get_next_node(self):
        if not self.planned_paths.empty():
            return self.planned_paths.get()

class Simulator:
    """Minimal discrete-event simulator.
    Subclass and override handle(event_id, payload) with a switch-case.
    """
    def __init__(self, edge_weights, start_time: float = 0.0) -> None:
        self.now = float(start_time)
        self._queue: list[Tuple[float, int, Any, Any, EventHandle]] = []
        self._seq = itertools.count()
        self._stopped = False
        self.events_processed = 0
        self.edge_weights = edge_weights

    def schedule_at(self, time: float, event_id: Any, payload: Any = None) -> EventHandle:
        if time < self.now:
            raise ValueError("Cannot schedule in the past")
        seq = next(self._seq)
        h = EventHandle()
        heapq.heappush(self._queue, (float(time), seq, event_id, payload, h))
        return h

    def _pop_next(self):
        '''
        I have modified this code to instead keep popping off events until the next event is 
        occurs at a different time
        The method now returns a list of events 
        '''
        events = []
        same_time = True # state variable to check if the previous event is still the same time
        prev_time = None # keep track of what the prev time
        while self._queue and same_time:
            time, seq, event_id, payload, h = self._queue[0]
            if time == prev_time or prev_time is None:
                time, seq, event_id, payload, h = heapq.heappop(self._queue)
                if not h.cancelled:
                    # time, event_id will be 
                    events.append((time, event_id, payload))
                    prev_time = time
            else:
                same_time = False
            # skipped cancelled
        return events

    def step(self) -> bool:
        '''
        Changed it a little bit so to work with modifications
        '''
        if self._stopped:
            return False
        events_list = self._pop_next()
        if events_list is None:
            return False
        time = events_list[0][0]
        self.now = time

        # calculate congestion + assign which event goes with which edge
        edge_counter = Counter()
        all_edges = []
        new_events = []
        for event in events_list:
            _, event_id, payload = event
            if event_id == 'A':
                car, curr_node = payload
                next_node = car.get_next_node() # you would have to modify here to get next node and you can modify all you want!!!
                if next_node is not None:
                    edge = (curr_node,next_node)
                    payload = (car, edge)
                    new_events.append((event_id, payload))
                    all_edges.append(edge)
            else:
                new_events.append((event_id, payload))
                
        edge_counter.update(all_edges)
        
        for event in new_events:
            event_id, payload = event
            if event_id == 'A':
                car, edge = payload
                cost_of_edge = edge_counter[edge] * self.edge_weights[edge]
                print(f"curr time is {self.now} and the edge {edge} has cost {cost_of_edge}")
                payload = (car, edge, cost_of_edge)
            self.handle(event_id, payload)

        # dispatch to user-defined handler
        self.events_processed += len(events_list)
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
        if event_id == "stop":
            print(f"[{self.now:.3f}] stopping")
            self.stop()
        elif event_id == 'A':
            car, edge, cost = payload
            prev, curr = edge
            car.add_path(prev,self.now,cost)
            self.schedule_at(self.now + cost, 'A', (car, curr))
        else:
            print(f"[{self.now:.3f}] unknown event {event_id!r} -> {payload}")
            
# methods to load in the graphs and agents

def read_graph(fname):
    # Open the file
    file = open(fname, "r")
    # Read the first line that contains the number of vertices
    # numVertices is the number of vertices in the graph (n)
    numVertices = file.readline()

    # You might need to add some code here to set up your graph object
    graph = WeightedGraph()
    # also adding another dictionary to get the base weights here
    edge_weights = {}

    # Next, read the edges and build the graph
    for line in file:
        # edge is a list of 3 indices representing a pair of adjacent vertices and the weight
        # edge[0] contains the first vertex (index between 0 and numVertices-1)
        # edge[1] contains the second vertex (index between 0 and numVertices-1)
        # edge[2] contains the weight of the edge (a positive integer)
        edge = line.strip().split(",")

        graph.addNode(int(edge[0]))
        graph.addNode(int(edge[1]))
        graph.addEdge(int(edge[0]), int(edge[1]), int(edge[2]))
        edge_weights[(int(edge[0]), int(edge[1]))] = int(edge[2])
    
    # Close the file safely after done reading
    file.close()
    return graph, edge_weights

def read_agents(fname):
    # Open the file
    file = open(fname, "r")
    # Set up your list of agents
    agents=[]

    # Next, read the agents and build the list
    for line in file:
        # agent is a list of 2 indices representing a pair of vertices
        # path[0] contains the start location (index between 0 and numVertices-1)
        # path[1] contains the destination location (index between 0 and numVertices-1)
        path = line.strip().split(",")
        agents.append((int(path[0]), int(path[1])))
    
    # Close the file safely after done reading
    file.close()
    return agents

    
# Example usage with a simple switch-case style handler
if __name__ == "__main__":
    graph_fn = 'input/grid100.txt'
    agents_fn = 'input/agents100.txt'

    graph, edge_weights = read_graph(graph_fn)
    agents = read_agents(agents_fn)
    num_agents = len(agents)
    
    # generate inter-arrival times
    time_between_calls = np_ran.default_rng().exponential(scale=1, size=(num_agents-1))
    # convert it to integers
    interarrival_time = [0] + [round(i) for i in time_between_calls]

    # first instantiate all of the car objects + run dijkstra
    cars = []
    for i in range(num_agents):
        start, end = agents[i]
        get_path, cost = graph.dijkstra_shortest_path(start, end)
        cars.append(Cars(start, end))
        cars[i].set_predetermined_path(get_path)

    sim = Simulator(edge_weights=edge_weights)

    # schedule starting event for each car
    t = 0 # help to determine actual time of arrivals
    arrival_time = []
    for i in range(num_agents):
        t += interarrival_time[i]
        arrival_time.append(t)
        curr_car = cars[i]
        first_node = curr_car.get_next_node()
        if first_node is not None:
            sim.schedule_at(t, 'A', (curr_car, first_node))
    
    #sim.schedule_at(15, 'stop', None)
    
    sim.run()
    print("events processed:", sim.events_processed)

    total_cong = 0
    for i in range(num_agents):
        curr_car = cars[i]
        total_cong += curr_car.total_cost
        print(f"Car {i} ({curr_car.start}, {curr_car.end}), arrived at t={arrival_time[i]}, with path {curr_car.path}")

    print(f'Average Congestion: {total_cong/num_agents}')
    print(f'Total Congestion: {total_cong}')