

class Node:
    """
    Represents a node in a search tree.

    Attributes:
        i (int): The value of i coordinate.
        j (int): The value of j coordinate.
        state (object): The state associated with the node.
        parent (Node, optional): The parent node of the current node.
    """

    def __init__(self, i, j, state, parent=None):
        """
        Initializes a new instance of the Node class.

        Args:
            i (int): The value of i.
            j (int): The value of j.
            state (list): The state value.
            parent (Node, optional): The parent node. Defaults to None.
        """
        self.i = i
        self.j = j
        self.state = state
        self.parent = parent

class HillClimbingSearch:
    """
    Class representing the Hill Climbing Search algorithm for solving a puzzle.

    Attributes:
        matrix (list): The initial state of the puzzle.
        goal (list): The goal state of the puzzle.
        visited (set): A set to keep track of visited states.
        states_explored (int): The number of states explored during the search.
        HEURISTICS (dict): A dictionary mapping heuristic names to their corresponding functions.
        root (Node): The root node of the search tree.
        goal_positions (dict): A dictionary mapping each goal tile to its position.

    Methods:
        swap(node, move): Swaps the current node's state with the state obtained by making a move in a specified direction.
        move_up(node): Moves the given node up by one position.
        move_down(node): Moves the given node down by swapping it with the element below it.
        move_left(node): Moves the given node to the left.
        move_right(node): Moves the empty tile to the right in the puzzle.
        generate_neighbors(node): Generates the neighboring nodes for a given node.
        print_path(node): Prints the path from the given node to the root node.
        h1(state): Heuristic function that calculates the number of misplaced tiles.
        h2(state): Heuristic function that calculates the Manhattan distance.
        solve(heuristic): Solves the puzzle using the hill climbing search algorithm.
        search(heuristic): Performs the hill climbing search to find the goal state.
    """

    def __init__(self, matrix, goal) -> None:
        """
        Initializes the HillClimb object.

        Args:
            matrix (list): The initial matrix state.
            goal (list): The goal matrix state.

        Attributes:
            matrix (list): The initial matrix state.
            goal (list): The goal matrix state.
            visited (set): A set to keep track of visited states.
            states_explored (int): The number of states explored.
            HEURISTICS (dict): A dictionary of available heuristics.
            root (Node): The root node of the search tree.
            goal_positions (dict): A dictionary mapping goal positions.

        Returns:
            None
        """
        self.matrix = matrix
        self.goal = goal
        self.visited = set()
        self.states_explored = 0
        self.HEURISTICS = {'h1': self.h1, 'h2': self.h2}

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == "B":
                    self.root = Node(i, j, matrix, None)

        self.goal_positions = {}
        for i in range(len(goal)):
            for j in range(len(goal[0])):
                self.goal_positions[goal[i][j]] = (i, j)

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
    
    def h1(self, state):
        """
        Heuristic function that calculates the number of misplaced tiles
        """
        count = 0
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] != self.goal[i][j] and state[i][j] != 0:
                    count += 1
        return count
    
    def h2(self, state):
        """
        Heuristic function that calculates the Manhattan distance
        """
        distance = 0
        for i in range(len(state)):
            for j in range(len(state[0])):
                tile = state[i][j]
                if tile != "B":
                    x, y = self.goal_positions[tile]
                    distance += abs(x - i) + abs(y - j)
        return distance

    def solve(self, heuristic):
        """
        Solves the puzzle using the hill climbing search algorithm.

        Args:
            heuristic (str): The heuristic function to use. Can be one of 'h1' or 'h2'.

        Returns:
            list: A list of states from the initial state to the goal state.
        """
        goal_node = self.search(heuristic)
        if goal_node:
            return self.print_path(goal_node)
        return None

    def search(self, heuristic):
        """
        Perform a hill climbing search using the specified heuristic.

        Args:
            heuristic (str): The name of the heuristic function to use.

        Returns:
            Node: The goal node if found, None otherwise.
        """

        current = self.root
        while True:
            self.visited.add(tuple(map(tuple, current.state)))

            if current.state == self.goal:
                return current
            
            local_min_heuristic = self.HEURISTICS[heuristic](current.state)
            best_neighbor = None
            
            neighbors = self.generate_neighbors(current)
            for neighbor in neighbors:
                neighbor_state_tuple = tuple(map(tuple, neighbor.state))
                if neighbor_state_tuple not in self.visited:
                    neighbor_heuristic = self.HEURISTICS[heuristic](neighbor.state)
                    if neighbor_heuristic < local_min_heuristic:
                        local_min_heuristic = neighbor_heuristic
                        best_neighbor = neighbor
            
            if best_neighbor is None:
                return None
            
            current = best_neighbor
            self.states_explored += 1

def course_det():
        """
        Displays the course details.

        This method prints out the details of the course, including the course name, session, subject,
        subject code, faculty name, and team members. Each detail is printed with a line of asterisks
        below it to align subsequent lines with the numbers.
        """
        print("\n", "-" * 51)
        s1 = "1. Course Name: Ex. M.Tech. Program"
        s2 = "2. Session: 2024-2026"
        s3 = "3. Subject: Artificial Intelligence"
        s4 = "4. Subject Code: CS561"
        s5 = "5. Faculty Name: Dr. Asif Ekbal"
        s6 = "6. Team Members:"
        s7 = "   a. Sanjeev Kumar (2403res117 - IITP002297)"
        s8 = "   b. Aditya Gupta (2403res85 - IITP002204)"
        s9 = "   c. Avinash Aanand (2403res99 - IITP002223)"
        s10 = "   d. Aman Kumar (2403res10 - IITP002012)"
        cd = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
        for detail in cd:
            print(detail)
            print(
                "*" * (len(detail) + 1)
            )  # Adjust the spacing to align subsequent lines with the numbers
        print("*" * 51)


def run(dry_run=False):
    course_det()
    import random
    goal = [["T1", "T2", "T3"], ["T4", "T5", "T6"], ["T7", "T8", "B"]]
    values = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "B"]
    count = 0
    for h in ["h1", "h2"]:
        while True:
            input = []
            value_added = set()
            for i in range(3):
                row = []
                for j in range(3):
                    val = random.choice(values)
                    while val in value_added:
                        val = random.choice(values)
                    row.append(val)
                    value_added.add(val)
                input.append(row)
            hill_climb = HillClimbingSearch(input, goal)
            result = hill_climb.solve(h)
            if dry_run:
                if not result:
                    print("No solution found")
                    print("Starting state:")
                    for row in input:
                        print(row)
                    print("Goal state:")
                    for row in goal:
                        print(row)
                    print("Total number of states explored before termination: ", hill_climb.states_explored)
                    print("-" * 50)
                    count += 1
                else:
                    print("Solution found")
                    print("Starting state:")
                    for row in input:
                        print(row)
                    print("Goal state:")
                    for row in goal:
                        print(row)
                    print("Total number of states explored: ", hill_climb.states_explored)
                    print(f"Total number of states to the optimal path using {h}: ", len(result))
                    print("Optimal path:")
                    for state in result:
                        for row in state:
                            print(row)
                        print()
                    print("-" * 50)
                    break
            else:
                # write everything to a file
                with open("output.txt", "a") as f:
                    if not result:
                        f.write("Solution not found\n")
                        f.write("Starting state:\n")
                        for row in input:
                            f.write(" ".join(row) + "\n")
                        f.write("Goal state:\n")
                        for row in goal:
                            f.write(" ".join(row) + "\n")
                        f.write("Total number of states explored before termination: " + str(hill_climb.states_explored) + "\n")
                        f.write("-" * 50 + "\n")
                        count += 1
                    else:
                        f.write("Solution found\n")
                        f.write("Starting state:\n")
                        for row in input:
                            f.write(" ".join(row) + "\n")
                        f.write("Goal state:\n")
                        for row in goal:
                            f.write(" ".join(row) + "\n")
                        f.write("Total number of states explored: " + str(hill_climb.states_explored) + "\n")
                        f.write(f"Total number of states to the optimal path using {h}: " + str(len(result)) + "\n")
                        f.write("Optimal path:\n")
                        for state in result:
                            for row in state:
                                f.write(" ".join(row) + "\n")
                            f.write("\n")
                        f.write("-" * 50 + "\n")
                        break
        if dry_run:
            print(f"Total number of invalid states for {h} : ", count)
            print("=" * 50)
            count = 0
        else:
            with open("output.txt", "a") as f:
                f.write(f"Total number of invalid states for {h} : " + str(count) + "\n")
                f.write("=" * 50 + "\n")
                count = 0
    if not dry_run:
        print("Output written to output.txt")

run()


    
