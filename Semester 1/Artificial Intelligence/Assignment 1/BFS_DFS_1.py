"""
Breath First Search
"""
        
class Node:
    def __init__(self, i, j, state, parent=None):
        """
        Initializes an instance of Node.

        Args:
            i (int): The value of i.
            j (int): The value of j.
            state (str): The state value.
            parent (MyClass, optional): The parent instance. Defaults to None.
        """
        self.i = i
        self.j = j
        self.state = state
        self.parent = parent

class BFS:
    def __init__(self, matrix, goal):
        """
        Initializes an instance of BFS.

        Args:
            matrix (list): A 2D list representing the matrix.
            goal (int): The goal value.

        Attributes:
            matrix (list): A 2D list representing the matrix.
            goal (int): The goal value.
            visited (set): A set to keep track of visited nodes.
            queue (list): A list to store nodes for BFS.
            states_explored (int): The number of states explored.
            root (Node): The root node of the matrix.

        """
        self.matrix = matrix
        self.goal = goal
        self.visited = set()
        self.queue = []
        self.states_explored = 0

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    self.root = Node(i, j, matrix, None)
    
    def swap(self, node, move):
        """
        Swaps the current node's state with the state obtained by making a move in a specified direction.

        Args:
            node (Node): The current node.
            move (str): The direction of the move. Can be one of 'up', 'down', 'left', or 'right'.

        Returns:
            list: The new state after the swap.

        Raises:
            ValueError: If an invalid move is provided.
        """
        new_state = [row[:] for row in node.state]
        if move == 'up':
            new_state[node.i][node.j], new_state[node.i - 1][node.j] = new_state[node.i - 1][node.j], new_state[node.i][node.j]
        elif move == 'down':
            new_state[node.i][node.j], new_state[node.i + 1][node.j] = new_state[node.i + 1][node.j], new_state[node.i][node.j]
        elif move == 'left':
            new_state[node.i][node.j], new_state[node.i][node.j - 1] = new_state[node.i][node.j - 1], new_state[node.i][node.j]
        elif move == 'right':
            new_state[node.i][node.j], new_state[node.i][node.j + 1] = new_state[node.i][node.j + 1], new_state[node.i][node.j]
        else:
            raise ValueError('Invalid move')
        return new_state


    def move_up(self, node):
        """
        Moves the given node up by one position.

        Args:
            node (Node): The node to move.

        Returns:
            Node: The new node after moving up.

        Raises:
            None

        """
        if node.i <= 0:
            return None
        return Node(node.i - 1, node.j, self.swap(node, 'up'), node)
    
    def move_down(self, node):
        """
        Moves the given node down by swapping it with the element below it.

        Args:
            node (Node): The node to be moved down.

        Returns:
            Node or None: The new node after moving down, or None if the node cannot be moved down.
        """
        if node.i >= len(node.state) - 1:
            return None
        return Node(node.i + 1, node.j, self.swap(node, 'down'), node)

    def move_left(self, node):
        """
        Moves the given node to the left.

        Args:
            node (Node): The node to be moved.

        Returns:
            Node or None: The new node after moving left, or None if the node cannot be moved left.
        """
        if node.j <= 0:
            return None
        return Node(node.i, node.j - 1, self.swap(node, 'left'), node)

    def move_right(self, node):
        """
        Moves the empty tile to the right in the puzzle.

        Args:
            node (Node): The current node representing the puzzle state.

        Returns:
            Node or None: The new node after moving the empty tile to the right, or None if it is not possible to move right.
        """
        if node.j >= len(node.state[0]) - 1:
            return None
        return Node(node.i, node.j + 1, self.swap(node, 'right'), node)

    def generate_neighbors(self, node):
        """
        Generates the neighboring nodes for a given node.

        Parameters:
        node (Node): The current node.

        Returns:
        list: A list of neighboring nodes.
        """
        neighbors = []
        for move in [self.move_up, self.move_down, self.move_left, self.move_right]:
            next_node = move(node)
            if next_node:
                neighbors.append(next_node)
        return neighbors
    
    def bfs(self):
        """
        Performs a breadth-first search (BFS) traversal on the graph.

        Returns:
            Node: The goal node if found, None otherwise.
        """
        self.queue.append(self.root)
        self.visited.add(tuple(map(tuple, self.root.state)))
        while self.queue:
            cur = self.queue.pop(0)
            self.states_explored += 1
            if cur.state == self.goal:
                return cur
            
            for neighbor in self.generate_neighbors(cur):
                if tuple(map(tuple, neighbor.state)) not in self.visited:
                    self.queue.append(neighbor)
                    self.visited.add(tuple(map(tuple, neighbor.state)))
        return None
    
    def print_path(self, node):
        """
        Prints the path from the given node to the root node.

        Args:
            node: The node from which to start printing the path.

        Returns:
            A list containing the states of the nodes in the path, from the given node to the root node.
        """
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]
    
    def solve(self):
        """
        Solves the problem using BFS algorithm.

        Returns:
            If a solution is found, returns the path to the solution.
            Otherwise, returns None.
        """
        result = self.bfs()
        if result:
            return self.print_path(result)
        else:
            return None

"""
Depth First Search
"""




import random
import numpy as np
import time

def random_array():
    """
    Generates a random 3x3 array of numbers.

    Returns:
    array (numpy.ndarray): The randomly generated array.
    """
    numbers = np.arange(9)
    np.random.shuffle(numbers)
    array = numbers.reshape((3, 3))
    print("The random Array will look like as:\n",array)
    return array

class NodeDFS:
    def __init__(self, state, parent, action):
        """
        Initializes an instance of NodeDFS.

        Args:
            state: The state of the object.
            parent: The parent object.
            action: The action performed on the object.

        Returns:
            None
        """
        self.state = state
        self.parent = parent
        self.action = action

class Frontier:
    """
    Represents a frontier for searching algorithms.

    Attributes:
        elements (list): A list of nodes in the frontier.

    Methods:
        add(node): Adds a node to the frontier.
        contains_state(state): Checks if the frontier contains a node with the given state.
        empty(): Checks if the frontier is empty.
        remove(): Removes and returns the first node from the frontier.
    """

    def __init__(self):
        """
        Initializes an instance of Frontier.
        """
        self.elements = []

    def add(self, node):
        """
        Adds a node to the elements list.

        Parameters:
        - node: The node to be added.

        Returns:
        None
        """
        self.elements.append(node)

    def contains_state(self, state):
        """
        Checks if the given state is present in the elements list.

        Parameters:
        - state: The state to be checked.

        Returns:
        - True if the state is present in the elements list, False otherwise.
        """
        return any((node.state[0] == state[0]).all() for node in self.elements)

    def empty(self):
        """
        Check if the elements list is empty.

        Returns:
            bool: True if the elements list is empty, False otherwise.
        """
        return len(self.elements) == 0

    def remove(self):
        """
        Removes and returns the first element from the frontier.

        Raises:
            Exception: If the frontier is empty.

        Returns:
            The first element from the frontier.
        """
        if self.empty():
            raise Exception("Empty Frontier")
        else:
            node = self.elements[0]
            self.elements = self.elements[1:]
        return node

class Neighbors:
    """
    A class that represents the neighbors of a given state in a matrix.

    Methods:
    --------
    get_neighbors(state):
        Returns a list of neighboring states and the corresponding actions to reach them.

    """

    @staticmethod
    def get_neighbors(state):
        """
        Returns a list of neighboring states and the corresponding actions to reach them.

        Parameters:
        -----------
        state : tuple
            A tuple representing the current state, where the first element is the matrix and the second element is the position of the empty cell.

        Returns:
        --------
        list
            A list of tuples, where each tuple contains an action and the resulting state.

        """
        mat, (row, col) = state
        results = [] 
        if row > 0:
            mat1 = np.copy(mat)
            mat1[row][col] = mat1[row - 1][col]
            mat1[row - 1][col] = 0
            results.append(('UP', [mat1, (row - 1, col)]))
        if col > 0:
            mat1 = np.copy(mat)
            mat1[row][col] = mat1[row][col - 1]
            mat1[row][col - 1] = 0
            results.append(('LEFT', [mat1, (row, col - 1)]))
        if row < 2:
            mat1 = np.copy(mat)
            mat1[row][col] = mat1[row + 1][col]
            mat1[row + 1][col] = 0
            results.append(('DOWN', [mat1, (row + 1, col)]))
        if col < 2:
            mat1 = np.copy(mat)
            mat1[row][col] = mat1[row][col + 1]
            mat1[row][col + 1] = 0
            results.append(('RIGHT', [mat1, (row, col + 1)]))
        return results

class DFS:
    """
    Depth First Search (DFS) algorithm implementation.

    Args:
        start (list): The start state of the search problem.
        startIndex (int): The index of the start state.
        goal (list): The goal state of the search problem.
        goalIndex (int): The index of the goal state.

    Attributes:
        start (list): The start state of the search problem.
        goal (list): The goal state of the search problem.
        solution (tuple): The solution path found by the algorithm.
        num_explored (int): The number of nodes explored during the search.
        search_depth (int): The depth of the solution path.
        running_time (float): The running time of the algorithm in seconds.

    Methods:
        solve: Executes the DFS algorithm to find a solution.

    """

    def __init__(self, start, startIndex, goal, goalIndex):
        """
        Initializes a new instance of the class.

        Parameters:
        - start: The starting state of the search.
        - startIndex: The index of the starting state.
        - goal: The goal state of the search.
        - goalIndex: The index of the goal state.

        Attributes:
        - self.start: A list containing the starting state and its index.
        - self.goal: A list containing the goal state and its index.
        - self.solution: The solution path found by the search algorithm.
        - self.num_explored: The number of states explored during the search.
        - self.search_depth: The depth of the solution path.
        - self.running_time: The running time of the search algorithm.
        """
        self.start = [start, startIndex]
        self.goal = [goal, goalIndex]
        self.solution = None
        self.num_explored = 0
        self.search_depth = 0
        self.running_time = 0

    def solve(self):
        """
        Executes the DFS algorithm to find a solution.

        Raises:
            Exception: If no solution is found.

        """
        start_time = time.time()
        visited = set()
        start = NodeDFS(state=self.start, parent=None, action=None)
        frontier = Frontier()
        frontier.add(start)

        while True:
            if frontier.empty():
                raise Exception("No solution found, please Generate New array!")
            node = frontier.remove()
            self.num_explored += 1
            
            if (node.state[0] == self.goal[0]).all():
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                self.search_depth = len(actions)
                break

            set_of_tuples = tuple(tuple(row) for row in node.state[0])
            set_of_tuplesgoal = tuple(tuple(row) for row in self.goal[0])
            if set_of_tuples == set_of_tuplesgoal:
                break
            visited.add(set_of_tuples)
        
            for action, state in Neighbors.get_neighbors(node.state):
                set_of_tuplesstate = tuple(tuple(row) for row in state[0])
                if set_of_tuplesstate not in visited:
                    child = NodeDFS(state=state, parent=node, action=action)
                    set_of_tupleschild = tuple(tuple(row) for row in child.state[0])
                    visited.add(set_of_tupleschild)
                    frontier.add(child)
        end_time = time.time()
        self.running_time = end_time - start_time

class DFS_Main:
    """
    Class representing Depth First Search (DFS) algorithm.

    Attributes:
        start (list): A list containing the start state and its index.
        goal (list): A list containing the goal state and its index.

    Methods:
        print(solution, num_explored, search_depth, start_state, goal_state, running_time):
            Prints the solution path, number of explored states, search depth, start state, goal state, and running time.
    """

    def __init__(self, start, startIndex, goal, goalIndex):
        """
        Initializes an instance of DFS_Main.

        Parameters:
        - start: The start value.
        - startIndex: The index of the start value.
        - goal: The goal value.
        - goalIndex: The index of the goal value.
        """
        self.start = [start, startIndex]
        self.goal = [goal, goalIndex]

    def print(self, solution, num_explored, search_depth, start_state, goal_state, running_time):
        """
        Prints the solution path, number of explored states, search depth, start state, goal state, and running time.

        Args:
            solution (tuple): A tuple containing the solution path.
            num_explored (int): The total number of states explored.
            search_depth (int): The depth of the search.
            start_state (list): The start state.
            goal_state (list): The goal state.
            running_time (float): The time taken for computation.

        Returns:
            None
        """
        print("Start State(Generated Randomly):\n", start_state, "\n")
        print("We have the Goal State as :\n", goal_state, "\n")
        print("Total States Explored: ", num_explored, "\n")
        print("Solutions are as:\n ")
        for action, cell in zip(solution[0], solution[1]):
            print("Action taken: ", action, "\n", cell[0], "\n")
            print("Goal Reached!!")
        print("\nCost of path(Depth):", len(solution[0]))
        print("Time Taken in computation:", running_time, "Seconds")
        print("\nPath Followed to achieve the Goal for DFS:\n", solution[0])

class into():
    """
    This class represents the introduction and options menu for the code window.
    It provides methods to display the introduction, course details, and options menu.
    """

    def intro(self):
        """
        Displays the introduction message and group details.
        """
        print("\n\nYou are most Welcome to the Code Window, Guest!!!!!", "\n","----------*----------"*5 ,"\nWho we are", "Group Details as below:")
        print("----------*----------"*5)
        gm1 = "Avinash Aanand-(Roll No.: 2403RES99, Email:avinash_2403res99@iitp.ac.in)"
        gm2 = "Aditya Gupta-(Roll No.:2403RES85 , Email:aditya_2403res85@iitp.ac.in)"
        gm3 = "Aman Kumar-(Roll No.:2403RES10 , Email:aman_2403res10@iitp.ac.in)"
        gl = [gm1, gm2, gm3]
        random.shuffle(gl)
        j = 1
        for i in gl:
            print(f"{j}. {i}")
            j += 1

    def course_det(self):
        """
        Displays the course details.
        """
        print("\n", "-" * 51)
        s1 = "1. Course Name: Ex. M.Tech. Program"
        s2 = "2. Session: 2024-2026"
        s3 = "3. Subject: Artificial Intelligence"
        s4 = "4. Subject Code: CS561 "
        s5 = "5. Faculty Name: Dr. Asif Ekbal"
        cd = [s1, s2, s3, s4, s5]
        for detail in cd:
            print(detail)
            print("*" * (len(detail) + 1))  # Adjust the spacing to align subsequent lines with the numbers
        print("*" * 101)


    def options(self):
        """
        Displays the options menu and handles user input.
        """
        print("\nProblem Statement:")
        print("1. Compare the Breadth First Search(BFS) and Depth First Search(DFS)concerning the number of steps required to reach the solution and whether they are reachable. If unreachable, start with a random state and retry until the Target State (given above) is reached.")
        print("2. Comment on which algorithm will be faster and when by mentioning proper intuition and examples")
        print("Please select an option:")
        print("\t1. DFS And BFS")
        print("\t2. For BFS Only")
        print("\t3. Exit (Press any other key)")
        print("\t4. Introduction Page")
        cases = {}
        while True:
            choice = input("Your selection, please: ")
            start = random_array()
            # start = np.array([[3, 2, 1], [4, 5, 6], [8, 7, 0]])
            goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
            if choice == '1':
                print("You have selected DFS (Depth First Search)")
                startIndex = (2, 2)
                goalIndex = (2, 2)
                dfs_solver = DFS(start, startIndex, goal, goalIndex)
                try:
                    dfs_solver.solve()
                    dfs = DFS_Main(start, startIndex, goal, goalIndex)
                    dfs.print(dfs_solver.solution, dfs_solver.num_explored, dfs_solver.search_depth, start, goal, dfs_solver.running_time)
                    
                    bfs_input = []
                    bfs_goal = []
                    for arr in start:
                        bfs_input.append(arr.tolist())
                    
                    for arr in goal:
                        bfs_goal.append(arr.tolist())

                    bfs = BFS(bfs_input, bfs_goal)
                    print("Path taken by BFS \n", bfs.solve())
                    cases[tuple(map(tuple, start.tolist()))] = {
                        "DFS": {
                            "states_explored": dfs_solver.num_explored,
                        },
                        "BFS": {
                            "states_explored": bfs.states_explored,
                        },
                        "difference": abs(dfs_solver.num_explored - bfs.states_explored),
                        "faster": "DFS is faster" if dfs_solver.num_explored < bfs.states_explored else "BFS is faster"
                    }
                    break
                except Exception as e:
                    print(e) 
            elif choice == '2':
                print("You have selected BFS (Breadth First Search)")
                bfs = BFS(start.tolist(), goal.tolist())
                print(bfs.solve())
                print(bfs.states_explored)
            elif choice == '3':
                print("Exiting...Thanks for coming!, See you again!!!!!")
                break
            elif choice == '4':
                print("Introduction Page:")  
            else:
                print("Invalid choice. Exiting Now. See you again!!!!!!")
                break
        
        print(cases)
        
intro = into()
intro.intro()
intro.course_det()
intro.options()