# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# Azhan Zaheer - CSS 382 A - Roger Stanev


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack
from util import Queue
from util import PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # Initialize the frontier using the initial state of the problem
    frontier = Stack()
    frontier.push((problem.getStartState(), [], 0))
    
    # Initialize the explored set to keep track of visited nodes
    explored = set()
    while not frontier.isEmpty():
        # Pop the latest node (state, actions, cost) from the stack
        state, actions, cost = frontier.pop()

        # If this state is the goal, return the actions that got us here
        if problem.isGoalState(state):
            return actions

        # If the state has not been visited, proceed
        if state not in explored:
            explored.add(state)

            # Iterate over the successors of the state
            for successor, action, stepCost in problem.getSuccessors(state):
                # If the successor has not been visited, push it onto the frontier
                if successor not in explored:
                    # Append the current action to the action list
                    new_actions = actions + [action]
                    # Push the successor onto the stack along with the new actions and the updated cost
                    frontier.push((successor, new_actions, cost + stepCost))

    # If no solution is found, return an empty list
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Initialize the frontier with the starting state
    frontier = Queue()
    frontier.push((problem.getStartState(), []))
    
    # Initialize the explored set to keep track of visited nodes
    explored = set()
    while not frontier.isEmpty():
        # Dequeue a node from the frontier
        state, actions = frontier.pop()

        # If the goal state is found, return the actions that led to it
        if problem.isGoalState(state):
            return actions

        # If the state has not been visited, proceed
        if state not in explored:
            explored.add(state)

            # Get successors and iterate through them
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in explored:
                    # Add this successor's state and the action to get to it
                    new_actions = actions + [action]
                    frontier.push((successor, new_actions))

    # If no solution is found, return an empty list
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    frontier = PriorityQueue()
    frontier.push((problem.getStartState(), [], 0), 0)
    explored = set()

    while not frontier.isEmpty():
        # Priority queue ensures that we pop the lowest cost node
        state, actions, cumulative_cost = frontier.pop()

        # If the node is the goal, return the actions to reach it
        if problem.isGoalState(state):
            return actions

        # If we haven't already visited this state, expand the frontier from here
        if state not in explored:
            explored.add(state)

            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in explored:
                    # The new cost will be the cumulative cost plus the cost of the step
                    new_cost = cumulative_cost + step_cost
                    new_actions = actions + [action]
                    # The priority is the new cost
                    frontier.update((successor, new_actions, new_cost), new_cost)

    # If the search completes without finding a goal
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Initialize the frontier with the start state of the problem
    frontier = PriorityQueue()
    start_state = problem.getStartState()
    frontier.push((start_state, [], 0), 0)

    # Initialize the explored set to keep track of visited nodes
    explored = set()

    while not frontier.isEmpty():
        # Pop the node with the lowest cost + heuristic from the frontier
        state, actions, cost = frontier.pop()

        # If this state is the goal, return the actions to reach it
        if problem.isGoalState(state):
            return actions

        # If we haven't visited this state yet, mark it as visited
        if state not in explored:
            explored.add(state)

            # Expand from this state to its successors
            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in explored:
                    # Calculate the cost to reach this successor
                    new_cost = cost + step_cost
                    # Calculate the priority using the heuristic
                    priority = new_cost + heuristic(successor, problem)
                    # Push the successor to the frontier with the updated priority
                    frontier.update((successor, actions + [action], new_cost), priority)

    # If the search completes without finding a goal
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
