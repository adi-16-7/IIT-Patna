import time
import psutil
import heapq  # For priority queue


class Astar:
    """
    Astar class represents the A* search algorithm implementation.

    Attributes:
        HEURISTICS (dict): A dictionary of heuristic functions.
        states_covered (list): A list to store the states covered during the search.

    Methods:
        heuristic_function_1: A heuristic function that always returns 0.
        heuristic_function_2: A heuristic function that counts the number of misplaced tiles.
        heuristic_function_3: A heuristic function that calculates the Manhattan distance.
        heuristic_function_4: A heuristic function that combines heuristic 3 with a constant.
        get_possible_moves: Returns a list of possible moves from a given state.
        check_monotonicity: Checks if the heuristic function satisfies the monotonicity property.
        informed_astar_search: Performs the A* search algorithm using an informed heuristic.
        construct_path: Constructs the path from the start state to the target state.
        get_memory_usage: Returns the memory usage of the process.
        show_loading: Displays a loading animation.
        course_det: Displays the course details.
    """

    def __init__(self):
        """
        Initializes an instance of the class.

        Parameters:
        None

        Returns:
        None
        """
        self.HEURISTICS = {
            "heuristic_function_1": self.heuristic_function_1,
            "heuristic_function_2": self.heuristic_function_2,
            "heuristic_function_3": self.heuristic_function_3,
            "heuristic_function_4": self.heuristic_function_4,
        }
        self.states_covered = []

    def heuristic_function_1(self, state, target_state=None):
        """
        This is a sample heuristic function that always returns 0.

        Parameters:
        - state: The current state of the problem.
        - target_state: The target state of the problem (optional).

        Returns:
        - The heuristic value, which is always 0.
        """
        return 0

    def heuristic_function_2(self, state, target_state):
        """
        Calculates the heuristic value for a given state using the misplaced tiles heuristic.

        Args:
            state (list): The current state of the puzzle.
            target_state (list): The target state of the puzzle.

        Returns:
            int: The number of misplaced tiles in the current state.
        """
        return sum(
            1
            for i in range(3)
            for j in range(3)
            if state[i][j] != target_state[i][j] and state[i][j] != 0
        )

    def heuristic_function_3(self, state, target_state):
        """
        Calculates the Manhattan distance between the current state and the target state.

        Parameters:
        state (list): The current state of the puzzle.
        target_state (list): The target state of the puzzle.

        Returns:
        int: The Manhattan distance between the current state and the target state.
        """
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    for m in range(3):
                        for n in range(3):
                            if state[i][j] == target_state[m][n]:
                                distance += abs(m - i) + abs(n - j)
        return distance

    def heuristic_function_4(self, state, target_state):
        """
        Calculates the heuristic value for the given state and target state using Heuristic 3 plus a constant.

        Parameters:
        state (object): The current state.
        target_state (object): The target state.

        Returns:
        float: The heuristic value for the given state and target state.
        """
        return self.heuristic_function_3(state, target_state) + 1

    def get_possible_moves(self, state):
        """
        Returns a list of possible moves from the given state.

        Args:
            state (list): The current state of the puzzle.

        Returns:
            list: A list of possible moves, where each move is represented by a new state.

        """
        zero_row, zero_col = None, None
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    zero_row, zero_col = i, j
                    break
            if zero_row is not None:
                break

        moves = []

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = zero_row + dr, zero_col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [
                    row[:] for row in state
                ]  # Make a copy of the current state
                new_state[zero_row][zero_col], new_state[new_row][new_col] = (
                    new_state[new_row][new_col],
                    new_state[zero_row][zero_col],
                )
                moves.append(new_state)

        return moves

    def check_monotonicity(self, start_state, target_state, heuristic):
        """
        Checks if the given heuristic function is monotonic for the A* algorithm.

        Args:
            start_state: The starting state of the search.
            target_state: The target state to reach.
            heuristic: The heuristic function to evaluate the cost of reaching the target state.

        Returns:
            True if the heuristic is monotonic, False otherwise.
        """
        for next_state in self.get_possible_moves(start_state):
            cost_n_m = 1
            h_n = heuristic(start_state, target_state)
            h_m = heuristic(next_state, target_state)
            if h_n > cost_n_m + h_m:
                return False
        return True

    def informed_astar_search(self, start_state, target_state, heuristic):
        """
        Performs an informed A* search algorithm to find the optimal path from the start state to the target state.

        Args:
            start_state (list): The initial state of the search.
            target_state (list): The target state to reach.
            heuristic (str): The name of the heuristic function to use.

        Returns:
            tuple: A tuple containing the following information:
                - bool: True if a path from the start state to the target state is found, False otherwise.
                - int: The number of states explored during the search.
                - int: The length of the optimal path from the start state to the target state.
                - list: The optimal path from the start state to the target state.
        """
        heuristic_func = lambda state: self.HEURISTICS[heuristic](state, target_state)

        open_list = []
        closed_set = set()
        heapq.heappush(
            open_list, (0 + heuristic_func(start_state), start_state, None)
        )  # f = g + h

        states_explored = 0
        while open_list:
            current_node = heapq.heappop(open_list)
            current_f, current_state, parent_state = current_node
            states_explored += 1
            self.states_covered.append(current_state)

            if current_state == target_state:
                path = self.construct_path(current_state, parent_state)

                return True, states_explored, len(path), path

            closed_set.add(tuple(tuple(row) for row in current_state))

            for next_state in self.get_possible_moves(current_state):
                if tuple(tuple(row) for row in next_state) in closed_set:
                    continue

                next_g = (
                    current_f - heuristic_func(current_state) + 1
                )  # g is the cost from start to node
                next_h = heuristic_func(next_state)
                next_f = next_g + next_h

                heapq.heappush(
                    open_list, (next_f, next_state, (current_state, parent_state))
                )

        return False, states_explored, len(self.states_covered), self.states_covered

    def construct_path(self, current_state, node):
        """
        Constructs a path from the current state to the given node.

        Args:
            current_state: The current state.
            node: The target node.

        Returns:
            A list representing the path from the current state to the target node.
        """
        path = []
        path.append(current_state)
        while node:
            state, parent = node
            path.append(state)
            node = parent
        return path[::-1]

    def get_memory_usage(self):
        """
        Returns the memory usage of the current process in megabytes (MB).

        Returns:
            float: The memory usage in MB.
        """
        process = psutil.Process()
        mem_usage = process.memory_info().rss / 1024 / 1024  # in MB
        return mem_usage

    def show_loading(self):
        """
        Displays a loading animation using a sequence of frames.

        The loading animation consists of a sequence of frames that are displayed
        one after another to create the illusion of movement.

        Args:
            None

        Returns:
            None
        """
        frames = ["-", "\\", "|", "/"]
        for _ in range(50):
            for frame in frames:
                print(f"\rLoading {frame}", end="", flush=True)
                time.sleep(0.1)

    def course_det(self):
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
        print("*" * 101)


# Main
def run(dry_run=True):
    """
    Runs the A* search algorithm with different heuristic functions.

    Args:
        dry_run (bool): If True, the results will be printed to the console. If False, the results will be written to a file.

    Returns:
        None
    """
    _ = Astar()
    _.course_det()
    start_state = [[3, 2, 1], [4, 5, 6], [8, 7, 0]]
    target_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    cases = [
        "heuristic_function_1",
        "heuristic_function_2",
        "heuristic_function_3",
        "heuristic_function_4",
    ]
    for case in cases:
        start_time = time.time()

        obj = Astar()
        obj.show_loading()

        result, explored_states, optimal_path_length, path = obj.informed_astar_search(
            start_state, target_state, case
        )
        if dry_run:
            if result:
                end_time = time.time()
                execution_time = end_time - start_time
                memory_usage = obj.get_memory_usage()
                print("\nCase: ", case)
                print("Solution Found")
                print("\nStart State:")
                print("\n".join([" ".join(map(str, row)) for row in start_state]))
                print("Goal State:")
                print("\n".join([" ".join(map(str, row)) for row in target_state]))
                print("\nTotal States Explored:", explored_states)
                print("Optimal Path Length:", optimal_path_length)
                print("Time Taken in seconds:", execution_time)
                print("Memory Usage in MB:", memory_usage)
                print("\nPath:")
                print("\n".join([" ".join(map(str, row)) for row in path]))
                if case in ["heuristic_function_2", "heuristic_function_3"]:
                    print(
                        "Monotone restriction for",
                        case + ":",
                        obj.check_monotonicity(
                            start_state, target_state, obj.HEURISTICS[case]
                        ),
                    )
                print(
                    "\n------------------------------------------------------------------------------------------"
                )
            else:
                print("\nCase: ", case)
                print("No solution found")
                print("Start State:")
                print("\n".join([" ".join(map(str, row)) for row in start_state]))
                print("Goal State:")
                print("\n".join([" ".join(map(str, row)) for row in target_state]))
                print("Final State Reached:")
                print("\n".join([" ".join(map(str, row)) for row in path[-1]]))
                print("\nTotal States Explored:", explored_states)
                print("\n Path:")
                print("\n".join([" ".join(map(str, row)) for row in path]))
                print(
                    "\n------------------------------------------------------------------------------------------"
                )
        else:
            with open("output.txt", "a") as file:
                if result:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    memory_usage = obj.get_memory_usage()
                    file.write("\nCase: " + case + "\n")
                    file.write("\nSolution Found\n")
                    file.write("\nStart State:\n")
                    file.write(
                        "\n".join([" ".join(map(str, row)) for row in start_state])
                        + "\n"
                    )
                    file.write("Goal State:\n")
                    file.write(
                        "\n".join([" ".join(map(str, row)) for row in target_state])
                        + "\n"
                    )
                    file.write(
                        "\nTotal States Explored: " + str(explored_states) + "\n"
                    )
                    file.write(
                        "Optimal Path Length: " + str(optimal_path_length) + "\n"
                    )
                    file.write("Time Taken in seconds: " + str(execution_time) + "\n")
                    file.write("Memory Usage in MB: " + str(memory_usage) + "\n")
                    file.write("\nPath:\n")
                    file.write(
                        "\n".join([" ".join(map(str, row)) for row in path]) + "\n"
                    )
                    if case in ["heuristic_function_2", "heuristic_function_3"]:
                        file.write(
                            "Monotone restriction for "
                            + case
                            + ": "
                            + str(
                                obj.check_monotonicity(
                                    start_state, target_state, obj.HEURISTICS[case]
                                )
                            )
                            + "\n"
                        )
                    file.write(
                        "\n------------------------------------------------------------------------------------------\n"
                    )
                else:
                    file.write("\nCase: " + case + "\n")
                    file.write("No solution found\n")
                    file.write("Start State:\n")
                    file.write(
                        "\n".join([" ".join(map(str, row)) for row in start_state])
                        + "\n"
                    )
                    file.write("Goal State:\n")
                    file.write(
                        "\n".join([" ".join(map(str, row)) for row in target_state])
                        + "\n"
                    )
                    file.write("Final State Reached:\n")
                    file.write(
                        "\n".join([" ".join(map(str, row)) for row in path[-1]]) + "\n"
                    )
                    file.write(
                        "\nTotal States Explored: " + str(explored_states) + "\n"
                    )
                    file.write("\n Path:\n")
                    file.write(
                        "\n".join([" ".join(map(str, row)) for row in path]) + "\n"
                    )
                    file.write(
                        "\n------------------------------------------------------------------------------------------\n"
                    )


run(dry_run=False)
