import numpy as np


def tour_cost(instance, solution, problem_size):
    cost = 0
    for j in range(problem_size - 1):
        cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
    cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
    return cost


def generate_neighborhood_matrix(instance):
    instance = np.array(instance)
    n = len(instance)
    neighborhood_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        distances = np.linalg.norm(instance[i] - instance, axis=1)
        sorted_indices = np.argsort(distances)  # sort indices based on distances
        neighborhood_matrix[i] = sorted_indices

    return neighborhood_matrix


def cal_avg_distance(instance_data, n_instance, n_cities, evaluated_algorithm: callable) -> float:
    n_max = n_instance
    dis = np.ones(n_instance)
    n_instance = 0

    for instance, distance_matrix in instance_data:
        # get neighborhood matrix
        neighbor_matrix = generate_neighborhood_matrix(instance)
        destination_node = 0
        current_node = 0
        route = np.zeros(n_cities)

        for i in range(1, n_cities - 1):
            near_nodes = neighbor_matrix[current_node][1:]
            mask = ~np.isin(near_nodes, route[:i])
            unvisited_near_nodes = near_nodes[mask]
            next_node = evaluated_algorithm(current_node, destination_node, unvisited_near_nodes, distance_matrix)
            if next_node in route:
                return None
            current_node = next_node
            route[i] = current_node

        mask = ~np.isin(np.arange(n_cities), route[:n_cities - 1])
        last_node = np.arange(n_cities)[mask]
        current_node = last_node[0]
        route[n_cities - 1] = current_node
        LLM_dis = tour_cost(instance, route, n_cities)
        dis[n_instance] = LLM_dis
        n_instance += 1
        if n_instance == n_max:
            break

    ave_dis = np.average(dis)
    return ave_dis
