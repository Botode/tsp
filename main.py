import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model():
    data = {}
    # Матрица расстояний
    data['distance_matrix'] = [
        [0, 3, 8, 4, 2, 9, 6, 9, 4, 7, 5, 9, 5, 7, 8, 5, 2, 5, 5, 8],
        [2, 0, 6, 5, 9, 2, 5, 6, 7, 9, 8, 2, 2, 2, 9, 4, 9, 6, 3, 3],
        [7, 6, 0, 8, 4, 3, 3, 4, 5, 4, 9, 7, 5, 4, 3, 2, 9, 4, 7, 8],
        [6, 7, 7, 0, 7, 9, 3, 6, 6, 8, 3, 4, 5, 6, 3, 9, 8, 9, 2, 4],
        [4, 4, 4, 6, 0, 9, 4, 9, 3, 8, 5, 4, 4, 6, 3, 5, 5, 6, 2, 9],
        [8, 2, 4, 6, 5, 0, 6, 2, 9, 5, 3, 6, 7, 8, 7, 8, 5, 8, 9, 6],
        [6, 2, 8, 3, 8, 6, 0, 7, 3, 8, 3, 3, 2, 7, 6, 5, 2, 2, 3, 8],
        [4, 4, 6, 4, 3, 2, 2, 0, 6, 7, 5, 6, 2, 7, 6, 3, 5, 2, 6, 6],
        [6, 5, 6, 9, 7, 9, 2, 6, 0, 8, 6, 6, 6, 8, 2, 8, 7, 9, 2, 8],
        [8, 7, 5, 2, 4, 3, 8, 3, 4, 0, 7, 7, 7, 3, 6, 6, 7, 4, 8, 6],
        [6, 3, 4, 7, 5, 2, 6, 3, 5, 8, 0, 8, 8, 5, 5, 2, 8, 3, 5, 2],
        [3, 4, 2, 6, 9, 8, 2, 5, 3, 5, 3, 0, 6, 9, 2, 7, 7, 8, 5, 4],
        [4, 3, 2, 6, 8, 3, 2, 6, 3, 7, 7, 7, 0, 8, 6, 8, 2, 8, 8, 3],
        [5, 8, 5, 8, 4, 4, 4, 8, 5, 4, 5, 4, 7, 0, 4, 6, 7, 4, 2, 6],
        [9, 6, 9, 9, 5, 9, 5, 8, 2, 6, 4, 3, 4, 4, 0, 7, 6, 4, 3, 8],
        [8, 7, 5, 2, 6, 3, 8, 3, 8, 3, 4, 7, 3, 7, 8, 0, 5, 6, 4, 3],
        [4, 4, 5, 4, 7, 9, 6, 5, 8, 6, 5, 6, 3, 7, 3, 5, 0, 6, 3, 6],
        [2, 2, 5, 6, 4, 9, 5, 8, 8, 4, 6, 3, 8, 9, 7, 9, 4, 0, 4, 7],
        [6, 3, 6, 9, 3, 2, 2, 4, 2, 7, 2, 3, 2, 8, 5, 4, 8, 7, 0, 6],
        [9, 7, 7, 7, 3, 9, 4, 8, 9, 5, 5, 8, 3, 6, 7, 4, 7, 6, 2, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # Количество транспортных средств
    data['num_vehicles'] = 2
    # Индекс отправной точки
    data['depot'] = 0
    return data


def print_solution(data, manager, routing, solution):
    acc_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Маршрут коммивояжера {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Длина маршрута: {}\n'.format(route_distance)
        print(plan_output)
        acc_route_distance += route_distance
    print('Суммарная длина маршрутов: {}'.format(acc_route_distance))


def process_solution(data, manager, routing, solution):
    sol = {}
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            sol[(manager.IndexToNode(previous_index), manager.IndexToNode(index))] = vehicle_id
    return sol


def calc(data):
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,
        3000,
        True,
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        print_solution(data, manager, routing, solution)
        return process_solution(data, manager, routing, solution)
    print('Решение не найдено!')
    return None

def draw(data, solution):
    G = nx.from_numpy_matrix(
        np.array(data['distance_matrix']), create_using=nx.DiGraph)
    pos = nx.kamada_kawai_layout(G)

    N = G.number_of_nodes()
    node_colors = ['indigo' if i > 0 else 'red' for i in range(N)]

    sol_edges = solution.keys()
    sol_colors = ['red' if solution[i] == 0 else 'green' for i in solution]
    other_edges = [edge for edge in G.edges() if edge not in solution]
    weights = [G[u][v]['weight'] for u, v in other_edges]
    maxWeight = max(weights)
    minWeight = min(weights)
    edge_colors = weights
    edge_alphas = [0.75 - (i - minWeight) / (maxWeight - minWeight) / 2 for i in weights]

    cmap = plt.cm.plasma

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_color='white')
    edges = nx.draw_networkx_edges(
        G,
        pos,
        edgelist=other_edges,
        node_size=500,
        arrowstyle='->',
        arrowsize=10,
        connectionstyle='arc3,rad=0.05',
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=0.5,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=sol_edges,
        node_size=500,
        arrowstyle='->',
        arrowsize=10,
        connectionstyle='arc3,rad=0.05',
        edge_color=sol_colors,
        width=2,
    )
    for i in range(len(other_edges)):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(weights)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

def main():
    # Исходные данные
    data = create_data_model()
    # Нахождение маршрутов
    solution = calc(data)
    # Отрисовка данных
    draw(data, solution)

if __name__ == '__main__':
    main()
