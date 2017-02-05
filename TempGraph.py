#!/usr/bin/env python3
""" Temporal graph class. """

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.colors as colors
import matplotlib.cm as cmx

# Set style for plotting
style.use('ggplot')
#['dark_background', 'fivethirtyeight', 'ggplot', 'bmh', 'grayscale']

from helpers import read_data, adjust_axes

###############################################################################
###############################################################################
# TODO Add time window to create graph (different from the end of the last
# interaction), and add further functionality: file names, path etc.

# HINT Negative times are impossible


class TempGraph(nx.MultiDiGraph):
    """ Temporal graph class. """

    def create_graph(self,
                     path):
        """ Create the graph from the data in <path>. """

        self.path = path
        self.create_nodes(path+'parameters.yaml')
        self.create_edges(path+'interactions.txt')

    def create_nodes(self,
                     filename):
        """ Look for the <number of ants> in <filename> and create <number of
        ants> nodes. Return <n_nodes>. """

        with open(filename) as data:
            for line in data:
                temp = line.split()
                if 'n_ants' in temp[0]:
                    self.add_nodes_from(range(int(temp[1])))
                    self.graph['n_nodes'] = int(temp[1])
                    return
            print('No nodes created.')

    def create_edges(self,
                     filename,
                     window_start=0,
                     window_end=None):
        """ Load the sorted interaction data from <filename> and create the
        corresponding edges. It's possible to add only the edges between
        the times <window_start> and <window_end>. """

        with open(filename) as data:
            data.readline()
            # Add edges that are inside the time window
            n_edges = 0
            for line in data:
                temp = [float(x) for x in line.split()]
                if temp[1] >= window_start:
                    if window_end is not None:
                        if temp[1] <= window_end:
                            n_edges += 1
                            self.add_edge(int(temp[2]),
                                          int(temp[3]),
                                          attr_dict={'time': int(temp[1]),  # use the end of the interaction as time
                                                     #'duration': (int(temp[0]),
                                                     #             int(temp[1])),
                                                     #'giver_position': (int(temp[4]),
                                                     #                   int(temp[5])),
                                                     #'receiver_position': (int(temp[6]),
                                                     #                      int(temp[7])),
                                                     #'giver_food_after': temp[8],
                                                     #'reveiver_food_after': temp[9],
                                                     #'food_transfer': temp[10]
                                                     })
                            # Add position for evalutation of the diffusion design
                            #self.node[int(temp[2])]['position'] = (int(temp[4]), int(temp[5]))
                            #self.node[int(temp[3])]['position'] = (int(temp[6]), int(temp[7]))

                        else:
                            break
                    elif window_end is None:
                        n_edges += 1
                        self.add_edge(int(temp[2]),
                                      int(temp[3]),
                                      attr_dict={'time': int(temp[1]),  # use the end of the interaction as time
                                                 #'duration': (int(temp[0]),
                                                 #             int(temp[1])),
                                                 #'giver_position': (int(temp[4]),
                                                 #                   int(temp[5])),
                                                 #'receiver_position': (int(temp[6]),
                                                 #                      int(temp[7])),
                                                 #'giver_food_after': temp[8],
                                                 #'reveiver_food_after': temp[9],
                                                 'food_transfer': temp[10]
                                                 })

                        # Add position for evalutation of the diffusion design
                        #self.node[int(temp[2])]['position'] = (int(temp[4]),
                        #                                       int(temp[5]))
                        #self.node[int(temp[3])]['position'] = (int(temp[6]),
                        #                                       int(temp[7]))

            # Write degrees at positions to file
            #with open(os.path.join(self.path, 'degrees_at_positions.txt'),
            #              'w') as file:
            #    for node in self.node:
            #        file.write(str(self.node[node]['position'][0]) +
            #                   '\t' + str(self.node[node]['position'][1]) +
            #                   '\t\t' + str(nx.degree(self, node)) + '\n')

            # Set end of time window as the end of the last interaction if it
            # is not specified in the parameters
            if window_end is None:
                try:  # No temp[1] if there are no interactions
                    window_end = int(temp[1])
                except:
                    with open(self.path + 'emptytime.txt') as data:
                        for line in data:
                            window_end = float(line.split()[0])

            # Add the attributes <time_window> and <n_edges> to the graph
            self.graph['time_window'] = (window_start, window_end)
            self.graph['n_edges'] = n_edges

    def sorted_edges(self,
                     parameter,
                     entry=None):
        """ Sort all edges of according to the entry with the number <entry> of
        the <parameter> in the data of the edges. Return the list of sorted
        edges. """
        if entry is None:
            return sorted(self.edges(data=True),
                          key=lambda x: x[2][parameter])
        else:
            return sorted(self.edges(data=True),
                          key=lambda x: x[2][parameter][entry])

###############################################################################

    def create_time_windows(self,
                            n_windows,
                            window_start=None,
                            window_end=None,
                            static_start=False,
                            static_end=False):
        """ Create and return a list of <n_windows> time window tuples in the
        format <(window_begin, window_end)>. It's possible to set a
        static start and a static end. """
        # Use the whole graph if no time window is specified
        if window_start is None:
            window_start = self.graph['time_window'][0]
        if window_end is None:
            window_end = self.graph['time_window'][1]
        # Create the list of time windows and return it
        time_windows = []
        if not static_start and not static_end:
            duration = (window_end - window_start) / float(n_windows)
            for n in range(n_windows):
                time_windows.append((window_start + n*duration,
                                     window_start + (n+1)*duration))
            return time_windows
        elif static_start and not static_end:
            for n in range(n_windows):
                duration = (((window_end - window_start) / float(n_windows))
                            * (n+1))
                time_windows.append((window_start,
                                     window_start + duration))
            return time_windows
        elif static_end and not static_start:
            for n in range(n_windows):
                duration = (((window_end - window_start) / float(n_windows))
                            * (n_windows-n))
                time_windows.append((window_end - duration,
                                     window_end))
            return time_windows
        else:
            print('Static start AND static end not possible.')

    def create_time_slices(self,
                           time_windows):
        """ Create a list of time aggregated graphs for every time window tuple
        <(window_begin, window_end)> in the list <time_windows>. """
        # Create time aggregated graphs
        graphs = []
        for window in time_windows:
            window_start = window[0]
            window_end = window[1]
            graphs.append(TempGraph([(g, r, d)
                                     for (g, r, d)
                                     in self.edges(data=True)
                                     if d['time'] >= window_start
                                     # HINT Maybe the window should not include
                                     # one border (<=/>=/=)
                                     and d['time'] <= window_end
                                     ],
                                    time_window=(window_start, window_end),
                                    n_nodes=self.graph['n_nodes']
                                    )
                          )
        # Add <n_edges> to the graph attributes
        for graph in graphs:
            graph.graph['n_edges'] = graph.number_of_edges()
        return graphs

    def draw_weighted(self,
                      figsize,
                      ax=None):
        """ Plot the weighted version of the graph. """
        weighted_graph = nx.DiGraph()

        for (g, r, d) in self.edges(data=True):
            if (g, r) in weighted_graph.edges():
                weighted_graph[g][r]['weight'] += 1
                if 'food_weight' in weighted_graph[g][r]:
                    weighted_graph[g][r]['food_weight'] += d['food_transfer']
                else:
                    weighted_graph[g][r]['food_weight'] = d['food_transfer']
            elif (r, g) in weighted_graph.edges():
                weighted_graph[r][g]['weight'] += 1
                if 'food_weight' in weighted_graph[r][g]:
                    weighted_graph[r][g]['food_weight'] += d['food_transfer']
                else:
                    weighted_graph[r][g]['food_weight'] = d['food_transfer']
            else:
                weighted_graph.add_edge(g, r, weight=1)
                if 'food_weight' in weighted_graph[g][r]:
                    weighted_graph[g][r]['food_weight'] += d['food_transfer']
                else:
                    weighted_graph[g][r]['food_weight'] = d['food_transfer']
        # Convert to drawing format
        weighted_graph = weighted_graph.to_undirected()
        pos = nx.circular_layout(weighted_graph)
        # Food weight
        #weights = []
        #for (i, j, d) in weighted_graph.edges(data=True):
        #    weights.append(d['food_weight'])
        #print('Weighted according to amount of exchanged food')
        #print('Mean: %f' % np.mean(weights))
        #print('Standard deviation: %f' % np.std(weights))
        #print('Relative standard deviation: %f'
        #      % (np.std(weights)/np.mean(weights)))
        #weights = (np.array(weights)-np.mean(weights))/np.std(weights)/3
        #print(weights)

        #plt.figure()
        #plt.title('Food weight ' + str(self.graph['time_window']))
        #plt.axis('off')
        #nx.draw_networkx_edges(weighted_graph, pos=pos, width=weights)
        #nx.draw_networkx_nodes(weighted_graph, pos=pos, node_color='#56B4E9')
                               #node_size=700)

        # Number weight
        #foodweights = weights
        weights = []
        for (i, j, d) in weighted_graph.edges(data=True):
            weights.append(d['weight'])
        #print('Weighted according to number of interactions')
        #print('Mean: %f' % np.mean(weights))
        #print('Standard deviation: %f' % np.std(weights))
        #print('Relative standard deviation: %f'
        #      % (np.std(weights)/np.mean(weights)))
        #weights = (np.array(weights)-np.mean(weights))/np.std(weights)/3
        weights = np.array(weights)
        weights = weights / np.mean(weights)
        weights = np.exp(weights)
        weights = weights / np.mean(weights)
        #print(weights)
        plt.figure(figsize=figsize)
        #plt.title('Number weight ' + str(self.graph['time_window']))
        plt.axis('off')

        # Calculate colors, the broader a line, the darker it is
        cm = plt.get_cmap('binary')
        cNorm = colors.Normalize(vmin=0, vmax=max(weights))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        colorList = []
        for n in range(len(weights)):
            colorVal = scalarMap.to_rgba(weights[n])
            colorList.append(colorVal)
        # Plot
        nx.draw_networkx_edges(weighted_graph, pos=pos, width=weights,
                               edge_color=colorList, alpha=1.0)
        nx.draw_networkx_nodes(weighted_graph, pos=pos, node_color='#56B4E9',
                               node_size=300, alpha=1.0)
        #nx.draw_networkx_labels(weighted_graph, pos=pos, font_size=22)
        plt.tight_layout()
        #print('Correlation between both:')
        #print(np.corrcoef(foodweights, weights))
        #print('\n')

###############################################################################

    def draw_all_layouts(self,
                         figsize,
                         arrows=True):
        """ Draw the graph with all possible standard layouts. """

        for layout in ['sprin', 'circular', 'shell', 'spectral']:

            plt.figure(figsize=figsize)
            #plt.title(layout+str(self.graph['time_window']))
            plt.axis('off')

            # Calculate the positions of the layout
            pos = getattr(nx, layout+'_layout')(self)

            # Draw the layout
            nx.draw_networkx_nodes(self, pos=pos, node_color='#56B4E9',
                                   node_size=300, alpha=1.0)
            #nx.draw_networkx_labels(self, pos=pos)
            nx.draw_networkx_edges(self, pos=pos, alpha=0.1, arrows=arrows,
                                   width=0.2)
            #nx.draw_networkx_edge_labels(self, pos=pos)
            plt.tight_layout()

    def draw_temporal(self,
                      figsize=(16, 9),
                      n_lines=None,
                      lines=True,
                      arrows=True,
                      ax=None):
        """ Draw graph with temporal layout. If <lines> is true, <n_lines>
        vertical lines representing the nodes are drawn. If <arrows> is true,
        the edges are directed. """

        # Use <n_nodes> for <n_lines> if <n_lines> is not specified
        if n_lines is None:
            n_lines = self.graph['n_nodes']
        window = self.graph['time_window']
        window_duration = window[1]-window[0]

        # Create the graph for plotting
        n = 0
        TempG = nx.DiGraph()
        for (g, r, d) in self.edges(data=True):
            TempG.add_edge(n, n+1)  # directed edge from giver to receiver
            TempG.node[n]['ant'] = g  # giver number
            TempG.node[n]['time'] = d['time']
            TempG.node[n+1]['ant'] = r  # receiver number
            TempG.node[n+1]['time'] = d['time']
            n += 2

        # Calculate the positions of the nodes in the temporal layout
        pos = {}
        for nodetuple in TempG.nodes(data=True):
            pos[nodetuple[0]] = ((nodetuple[1]['ant']*figsize[0]/(n_lines-1)),
                                 ((nodetuple[1]['time']-window[0])*figsize[1]
                                  / (window_duration))
                                 )

        # Draw the graph with temporal layout
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
            #plt.title(str(window[0]) + '-' + str(window[1]), fontsize=18)
            #plt.text(-1, 4.5,
            #         'Timesteps ' + str(window[0]) + '-' + str(window[1]),
            #         rotation='vertical',
            #         verticalalignment='center',
            #         horizontalalignment='left',
            #         fontsize=22)
        nx.draw_networkx_edges(TempG, pos, alpha=0.2, arrows=arrows,
                               width=0.5, ax=ax)

        # Draw vertical lines representing the nodes
        if lines is True:
            Lines = nx.Graph()
            lines_pos = {}
            for n in range(0, 2*n_lines, 2):
                Lines.add_edge(n, n+1)
                lines_pos[n] = (n/2*figsize[0]/(n_lines-1), 0)
                lines_pos[n+1] = (n/2*figsize[0]/(n_lines-1), figsize[1])
            nx.draw_networkx_edges(Lines,
                                   lines_pos,
                                   alpha=0.5,
                                   edge_color='#56B4E9',
                                   width=0.5,
                                   ax=ax)
        ax.set_axis_off()
        plt.tight_layout()

    def draw_aggregated_evolution(self,
                                  n_windows=1,
                                  directed=True,
                                  static_start=False,
                                  static_end=False):
        """ Draw the temporal evolution of the network, i.e. <n_windows> time
        aggregated graphs. """

        # Create <n_windows> time aggregated graphs
        graphs = self.create_time_slices(
            self.create_time_windows(n_windows,
                                     static_start=static_start,
                                     static_end=static_end))
        if directed is False:
            for n in range(len(graphs)):
                graphs[n] = graphs[n].to_undirected()

        # Draw
        if n_windows > 1:
            fig, axes = plt.subplots(nrows=1, ncols=len(graphs),
                                     figsize=(16*len(graphs), 16))
            for n in range(len(graphs)):
                nx.draw_circular(graphs[n], ax=axes[n], node_color='#56B4E9',
                                 with_labels=False, node_size=300,
                                 width=0.2)
                axes[n].set_title(str(int(graphs[n].graph['time_window'][0]))
                                  + '-'
                                  + str(int(graphs[n].graph['time_window'][1])),
                                  fontsize=22)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 16))
            ax.set_title(str(int(graphs[0].graph['time_window'][0])) + '-'
                         + str(int(graphs[0].graph['time_window'][1])),
                         fontsize=22)
            nx.draw_circular(graphs[0], ax=ax, node_color='#56B4E9',
                             with_labels=False, node_size=300,
                             width=0.2)

    def draw_ordered_evolution(self,
                               n_windows=1,
                               lines=True,
                               arrows=True,
                               static_start=False,
                               static_end=False):
        """ Draw the temporal evolution of the network, i.e. <n_windows> time
        ordered graphs. """

        # Create <n_windows> time aggregated graphs
        graphs = self.create_time_slices(
            self.create_time_windows(n_windows,
                                     static_start=static_start,
                                     static_end=static_end))
        #Plot
        if len(graphs) > 1:
            fig, axes = plt.subplots(nrows=1, ncols=len(graphs),
                                     figsize=(16*len(graphs), 16))
            for n in range(len(graphs)):
                graphs[n].draw_temporal(lines=lines, arrows=arrows, ax=axes[n])
                plt.axis('off')
                axes[n].set_title(str(int(graphs[n].graph['time_window'][0]))
                                  + '-'
                                  + str(int(graphs[n].graph['time_window'][1])),
                                  fontsize=22)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 16))
            ax.set_title(str(int(graphs[n].graph['time_window'][0])) + '-'
                         + str(int(graphs[n].graph['time_window'][1])),
                         fontsize=22)
            graphs[0].draw_temporal(lines=lines, arrows=arrows, ax=ax)

###############################################################################

    def plot_degree_distributions(self,
                                  n_windows=1):
        """ Plot <n_windows> distributions of degree, in-degree, and
        out-degree. """

        # Create <n_windows> time aggregated graphs
        graphs = self.create_time_slices(self.create_time_windows(n_windows))

        # Plot the degree distributions
        fig, axes = plt.subplots(nrows=1,
                                 ncols=len(graphs),
                                 sharex=True,
                                 )
        plt.title('Degree, in-degree, out-degree')
        for n in range(len(graphs)):
            # Degree
            degrees = graphs[n].degree().values()
            max_degree = max(degrees)
            hist = [0]*(max_degree+1)
            for degree in degrees:
                hist[degree] += 1
            axes[n].scatter(range(max_degree+1), hist, color='b')
            # In-degree
            degrees = graphs[n].in_degree().values()
            hist = [0]*(max_degree+1)
            for degree in degrees:
                hist[degree] += 1
            axes[n].scatter(range(max_degree+1), hist, color='r')
            # Out-degree
            degrees = graphs[n].out_degree().values()
            hist = [0]*(max_degree+1)
            for degree in degrees:
                hist[degree] += 1
            axes[n].scatter(range(max_degree+1), hist, color='g')

###############################################################################

    def plot_degree_histograms(self,
                               plot=True,
                               n_windows=1,
                               save=True,
                               filename='degrees.dat'):
        """ Plot <n_windows> histograms for the degree distribution. """

        graphs = self.create_time_slices(self.create_time_windows(n_windows))

        if plot:
            fig, axes = plt.subplots(nrows=1,
                                     ncols=len(graphs),
                                     sharex=True,
                                     )
            plt.xlabel(r'Number of interactions')
            plt.ylabel(r'Number of ants')

            if len(graphs) > 1:
                for n in range(len(graphs)):
                    values = list(graphs[n].degree().values())
                    axes[n].hist(list(graphs[n].degree().values()),
                                 bins=range(max(graphs[n].degree().values())),
                                 color='grey')
                    adjust_axes()
                    plt.tight_layout()
                #axes[1][n].hist(list(graphs[n].in_degree().values()),
                #                 bins=range(max(graphs[n].degree().values())),
                #                 color='grey')
                #axes[2][n].hist(list(graphs[n].out_degree().values()),
                #                 bins=range(max(graphs[n].degree().values())),
                #                 color='grey')
            else:
                n = 0
                values = list(graphs[n].degree().values())
                if plot:
                    axes.hist(values,
                              bins=np.array(range(min(values),
                                                  max(values)+2))-0.5,
                              color='grey')
                    adjust_axes()
                    plt.tight_layout()
                #axes[1].hist(list(graphs[n].in_degree().values()),
                #              bins=range(max(graphs[n].degree().values())),
                #              color='grey')
                #axes[2].hist(list(graphs[n].out_degree().values()),
                #              bins=range(max(graphs[n].degree().values())),
                #              color='grey')

        #Save the degree data to a file
        if save:
            values = list(graphs[0].degree().values())
            with open(os.path.join(self.path, filename), 'w') as file:
                for value in values:
                    file.write(str(value)+'\n')

    def plot_centrality_histograms(self):
        """ Plot histograms for the distribution of various centralities. """

        plt.figure()
        plt.title('degree centrality')
        plt.hist(list(nx.degree_centrality(self).values()))

        plt.figure()
        plt.title('in degree centrality')
        plt.hist(list(nx.in_degree_centrality(self).values()))

        plt.figure()
        plt.title('out degree centrality')
        plt.hist(list(nx.out_degree_centrality(self).values()))

        plt.figure()
        plt.title('closeness centrality')
        plt.hist(list(nx.closeness_centrality(self).values()))

        plt.figure()
        plt.title('betweenness centrality')
        plt.hist(list(nx.betweenness_centrality(self).values()))

        plt.figure()
        plt.title('eigenvector centrality')
        plt.hist(list(nx.eigenvector_centrality(self).values()))

        plt.figure()
        plt.title('load centrality')
        plt.hist(list(nx.load_centrality(self).values()))

###############################################################################

    def plot_food_source(self):
        """ Plot the amount of food in the food source over time. """
        # Read data
        path = self.path + '/'
        times, values = read_data(path + 'entrance_food.txt')
        # Plot
        plt.figure()
        #plt.title('plot_food_source(%s) ' % self.path)
        plt.xlabel('Time step')
        plt.ylabel(r'$\frac{\textrm{Food in food source}}'
                   r'{\textrm{Food capacity of an ant}}$')
        plt.step(times, values, where='post', color='black', linewidth=2)
        # Adjust axes limits
        adjust_axes()
        plt.tight_layout()

    def plot_pickups(self):
        """ Plot the number of ants that have picked up food from the food
        source over time (the points in time when the pick up ended) and the
        number of pickups. """

        times = []
        ants = set({})
        values = []
        pickups = []
        # Read data
        path = self.path + '/'
        with open(path + 'pickups.txt') as file:
            for line in file:
                data = line.split()
                if data[0] != '#':
                    times.append(int(float(data[1])))
                    ants.add(int(data[2]))
                    values.append(len(ants))
                    if len(pickups) > 0:
                        pickups.append(pickups[-1]+1)
                    else:
                        pickups.append(1)
        # Plot
        plt.figure()
        #plt.title('plot_pickups(%s) ' % self.path)
        plt.xlabel('Time step')
        plt.ylabel(r'Number of pickups' '\n'
                   r'Number of ants that have picked up food')
        plt.step(times, pickups, where='post', linewidth=2, alpha=0.8,
                 label='Number of pickups')
        plt.step(times, values, where='post', linewidth=2, alpha=0.8,
                 label='Number of ants')
        # Adjust axes limits
        adjust_axes()
        plt.legend(loc='best')
        plt.tight_layout()

    def plot_amount_picked_up(self):
        """ Plot the amounts of food that were picked up through pickups from
        the food source over time. """
        times = []
        values = []
        # Read data
        path = self.path + '/'
        with open(path + 'pickups.txt') as file:
            for line in file:
                data = line.split()
                if data[0] != '#':
                    times.append(int(float(data[1])))
                    values.append(float(data[7]))

        # Plot
        plt.figure()
        #plt.title('plot_amount_picked_up(%s) ' % self.path)
        plt.xlabel('Time step')
        plt.ylabel(r'$\frac{\textrm{Amount of food picked up}}'
                   r'{\textrm{Food capacity of an ant}}$')
        plt.plot(times, values, 'o', alpha=0.5)
        # Adjust axes limits
        adjust_axes()
        plt.legend(loc='best')
        plt.tight_layout()

    def plot_n_interactions(self):
        """ Plot the number of trophallactic interactions over time. """

        times = []
        values = []
        # Read data
        path = self.path + '/'
        with open(path + 'n_interactions.txt') as file:
            for line in file:
                data = line.split()
                if data[0] != '#':
                    times.append(int(data[0]))
                    values.append(int(data[1]))
        values = np.cumsum(values)
        # Plot
        plt.figure()
        #plt.title('plot_number_of_interactions(%s) ' % self.path)
        plt.xlabel('Time step')
        plt.ylabel('Total number of interactions')
        adjust_axes()
        plt.tight_layout()

    def plot_amounts_exchanged(self):
        """ Plot the amounts of food that were exchanged through trophallactic
        interactions over time. """

        times = []
        values = []
        # Read data
        path = self.path + '/'
        with open(path + 'interactions.txt') as file:
            for line in file:
                data = line.split()
                if data[0] != '#':
                    times.append(int(data[1]))
                    values.append(float(data[-1]))
        # Plot
        plt.figure()
        plt.xlabel(r'Time step')
        plt.ylabel(r'$\frac{\textrm{Amount of food exchanged}}'
                   r'{\textrm{Food capacity of an ant}}$')
        plt.plot(times, values, 'o', alpha=0.5, markersize=2.0)
        adjust_axes()
        plt.tight_layout()

        # Write data
        with open(os.path.join(self.path,
                               'amounts_exchanged.txt'), 'w') as file:
            values = np.cumsum(values)
            for n in range(len(times)):
                file.write(str(times[n]) + ' ' + str(values[n]) + '\n')

    def plot_total_amount_exchanged(self):
        """ Plot the total amount of food that has been exchanged through
        trophallactic interactions over time. """

        times = []
        values = []
        # Read data
        path = self.path + '/'
        with open(path + 'interactions.txt') as file:
            for line in file:
                data = line.split()
                if data[0] != '#':
                    times.append(int(data[1]))
                    values.append(float(data[-1]))
        # Write data
        with open(os.path.join(self.path,
                               'total_amount_exchanged.txt'), 'w') as file:
            values = np.cumsum(values)
            for n in range(len(times)):
                file.write(str(times[n]) + ' ' + str(values[n]) + '\n')

        # Plot
        plt.figure()
        plt.xlabel(r'Time step')
        plt.ylabel(r'$\frac{\textrm{Total amount of food exchanged}}'
                   r'{\textrm{Food capacity of an ant}}$')
        plt.plot(times, values, linewidth=2, color='black')
        # Adjust axes limits
        adjust_axes()
        plt.tight_layout()

    def plot_food_in_ants(self):
        """ Plot the amounts of food in the ants for every point in time. """

        path = self.path + '/'
        n_ants = self.number_of_nodes()
        times = []
        values = [list([]) for ant in range(n_ants)]
        with open(path + 'ant_foods.txt') as file:
            for line in file:
                data = line.split()
                if data[0] != '#':
                    for i in range(len(data)-1):
                        values[i].append(data[i+1])
        # Plot
        times = list(range(len(values[0])))
        plt.figure()
        plt.xlabel('Time step')
        plt.ylabel(r'$\frac{\textrm{Food in an ant}}'
                   r'{\textrm{Food capacity of an ant}}$')
        for i in range(len(values)):
            plt.plot(times, values[i], 'ro',  alpha=0.05, markersize=2.0)
        adjust_axes()
        plt.tight_layout()

    def plot_n_given(self,
                     plot=True,
                     axes=None,
                     save=True,
                     filename='n_given.dat'):
        """ Plot the number of individuals to which an individual has given
        food via direct trophalaxis and the average thereof over time.
        """
        # TODO Allow plotting to an axes instance

        # List of sets that contain the reached individuals for each individual
        reached = [set({}) for n in range(self.graph['n_nodes'])]
        # List of time points of first interactions for every individual
        interactions = [[] for n in range(self.graph['n_nodes'])]
        # Fill the lists
        for (g, r, d) in self.edges(data=True):
            if r not in reached[g]:
                reached[g].add(r)
                interactions[g].append(d['time'])

        # The individuals #####################################################
        if plot:
            # Create a figure if necessary
            if axes is None:
                plt.figure()
                #plt.title('plot_n_given(%s)' % self.path)
                plt.xlabel(r'Time step')
                plt.ylabel(r'Number of unique outgoing interactions')
        #i = -1  # For writting the ant number to the file
        for times in interactions:
            #i += 1
            # Add a zero at the beginning of <times>
            times = [0] + times
            # Create the corresponding list of <values>
            values = list(range(len(times)))
            # Add last value
            if times[-1] != self.graph['time_window'][1]:
                times.append(self.graph['time_window'][1])
                values.append(values[-1])
            if plot:
                # Plot as step function
                plt.step(sorted(times), values, where='post', alpha=0.3)

        # The average #########################################################
        # Create a dictionary with time points as keys and the number of
        # unique interactions at (!) those points as values
        unique = {}
        for times in interactions:
            for t in times:
                if t in unique:
                    unique[t] += 1
                else:
                    unique[t] = 1

        # Convert the dictionary to two list (containing time points and
        # interactions until (!) those points) and sort them
        times = sorted(unique.keys())
        values = np.cumsum([float(unique[x]) for x in times])

        # Add a zero at the beginning
        times = [0] + times
        values = np.insert(values, 0, 0)
        # Add last value if necessary
        if times[-1] != self.graph['time_window'][1]:
            times.append(self.graph['time_window'][1])
            values = np.append(values, values[-1])
        # Caculate the average
        values /= self.graph['n_nodes']
        # Save the data to <self.path>+<filename>
        if save:
            # Write the data
            with open(os.path.join(self.path, filename), 'w') as file:
                #file.write('# average\n')  # Description as comment
                for n in range(len(times)):
                    file.write(str(times[n]) + '\t' + str(values[n]) + '\n')

        if plot:

            # Plot as a step function
            plt.step(times, values, where='post', linewidth=2, color='black',
                     label='Average')
            # Adjust axes limits
            adjust_axes()
            plt.legend(loc='best')
            plt.tight_layout()

    def plot_n_received(self,
                        plot=True,
                        axes=None,
                        save=True,
                        filename='n_received.dat'):
        """ Plot the number of individuals from which an individual has
        received food via direct trophalaxis and the average thereof over time.
        """
        # TODO Allow plotting to an axes instance

        # List of sets that contain the reached individuals for each individual
        reached = [set({}) for n in range(self.graph['n_nodes'])]
        # List of time points of first interactions for every individual
        interactions = [[] for n in range(self.graph['n_nodes'])]
        # Fill the lists
        for (g, r, d) in self.edges(data=True):
            if g not in reached[r]:
                reached[r].add(g)
                interactions[r].append(d['time'])

        # The average #########################################################
        # Create a dictionary with time points as keys and the number of
        # unique interactions at (!) those points as values
        unique = {}
        for times in interactions:
            for t in times:
                if t in unique:
                    unique[t] += 1
                else:
                    unique[t] = 1

        # Convert the dictionary to two list (containing time points and
        # interactions until (!) those points) and sort them
        times = sorted(unique.keys())
        values = np.cumsum([float(unique[x]) for x in times])

        # Add a zero at the beginning
        times = [0] + times
        values = np.insert(values, 0, 0)
        # Add last value if necessary
        if times[-1] != self.graph['time_window'][1]:
            times.append(self.graph['time_window'][1])
            values = np.append(values, values[-1])
        # Caculate the average
        values /= self.graph['n_nodes']

        # Save the data to <self.path>+<filename>
        if save:
            # Write the data
            with open(os.path.join(self.path, filename), 'w') as file:
                #file.write('# average\n')  # Description as comment
                for n in range(len(times)):
                    file.write(str(times[n]) + '\t' + str(values[n]) + '\n')

        if plot:
            # Create a figure if necessary
            if axes is None:
                plt.figure()
                #plt.title('plot_n_received(%s)' % self.path)
                plt.xlabel('Time step')
                plt.ylabel('incoming interactions per individual')
            # Plot as step function
            plt.step(times, values, where='post', linewidth=2, color='black')

        # The individuals #####################################################
        #i = -1  # For writting the ant number to the file
        for times in interactions:
            #i += 1
            # Add a zero at the beginning of <times>
            times = [0] + times
            # Create the corresponding list of <values>
            values = list(range(len(times)))
            # Add last value
            if times[-1] != self.graph['time_window'][1]:
                times.append(self.graph['time_window'][1])
                values.append(values[-1])

            if plot:
                # Plot as step function
                plt.step(sorted(times), values, where='post', alpha=0.3)

    def plot_n_reached(self,
                       plot=True,
                       cut=True,
                       axes=None,
                       save=True,
                       filename='n_reached.dat'):
        """ Plot the number of individuals that an individual has already
        reached via direct or indirect trophalaxis and the average thereof
        over time. Every individual can reach itself. """
        # TODO Allow plotting to axes object

        # Create empty dictionary with ant numbers as keys. The value for each
        # ant is a dictionary with times as keys, whose values are all
        # individuals that the ant has reached until those times.
        tree = {n: ([], []) for n in self.nodes()}
        # Fill the tree
        for (g, r, d) in self.sorted_edges('time'):
            t = d['time']
            # Add the receiver and the already existing interactions of the
            # giver and to the dictionary at time t
            if len(tree[g][0]) > 0:
                if tree[g][0][-1] != t:
                    tree[g][0].append(t)
                    tree[g][1].append(set({r}) | tree[g][1][-1])
                else:
                    tree[g][1][-1].add(r)
            else:
                tree[g][0].append(t)
                tree[g][1].append(set({r}))
            # Add the receiver to the already existing interactions of the
            # the other ants that had already reached the giver
            for n in tree.keys():
                if len(tree[n][0]) > 0:
                        if g in tree[n][1][-1] and r not in tree[n][1][-1]:
                            if tree[n][0][-1] != t:
                                tree[n][0].append(t)
                                tree[n][1].append(set({r}) | tree[n][1][-1])
                            else:
                                tree[n][1][-1].add(r)

        # The individuals #####################################################
        #i = -1  # For writting the ant number to the file
        if plot:
            # Create the figure for plotting
            if axes is None:
                plt.figure()
                #plt.title('plot_n_reached(%s)' % self.path)
                plt.xlabel('Time step')
                plt.ylabel('Number of reached ants')
        for ant in tree:
        #    i += 1
            times = tree[ant][0]
            values = [len(reached) for reached in tree[ant][1]]
            # Add zeros for a proper beginning of the plot
            times = [0] + times
            values = [0] + values
            # Add last value for a proper ending if necessary
            if times[-1] != self.graph['time_window'][1]:
                times.append(self.graph['time_window'][1])
                values.append(values[-1])

            if plot:
                # Plot as step function
                if cut:  # Cut the plot where 100% are reached
                    maxindex = values.index(max(values))
                    values = values[:maxindex+1]
                    times = times[:maxindex+1]
                plt.step(times, values, where='post', linewidth=1, alpha=0.3)

        # The average #########################################################
        # Create a dictionary with time points as keys and the number of
        # unique 'reachings' at (!) those points as values
        unique = {}
        for ant in tree:
            for n, t in enumerate(tree[ant][0]):
                if t not in unique:
                    if n > 0:
                        unique[t] = (len(tree[ant][1][n])
                                     - len(tree[ant][1][n-1]))
                    else:
                        unique[t] = len(tree[ant][1][n])
                else:
                    if n > 0:
                        unique[t] += ((len(tree[ant][1][n])
                                       - len(tree[ant][1][n-1])))
                    else:
                        unique[t] += len(tree[ant][1][n])

        # Convert the dictionary to two sorted lists that contain the time
        # points and the 'reachings' until (!) those points respectively
        times = sorted(unique.keys())
        values = np.cumsum([float(unique[t]) for t in times])
        # Adding zeros for a proper beginning of the plot
        times = [0] + times
        values = np.insert(values, 0, 0)
        # Add last value for a proper ending
        if times[-1] != self.graph['time_window'][1]:
            times.append(self.graph['time_window'][1])
            values = np.append(values, values[-1])
        # Calculate the average
        values /= self.graph['n_nodes']

        # Save the data to <self.path>+<filename>
        if save:
            # Write the data
            with open(os.path.join(self.path, filename), 'w') as file:
                #file.write('# average\n')
                for n in range(len(times)):
                    file.write(str(times[n]) + '\t' + str(values[n]) + '\n')

        if plot:
            # Plot as step function
            if cut:  # Cut the plot where 100% are reached
                maxindex = values.argmax()
                values = values[:maxindex+1]
                times = times[:maxindex+1]
            plt.step(times, values, where='post', linewidth=2, color='black',
                     label='Average')
            plt.legend(loc='best')
            # Adjust axes limits
            adjust_axes()
            plt.tight_layout()

    def plot_n_reached_by(self,
                          plot=True,
                          cut=True,
                          axes=None,
                          save=True,
                          filename='n_reached_by.dat'):
        """ Plot the number of individuals that have already reached an
        individual via direct or indirect trophalaxis and the average thereof
        over time. Every individual can reach itself. """
        # TODO Allow plotting to axes object

        # Create empty dictionary with ant numbers as keys. The value for each
        # ant is a dictionary with times as keys, whose values are all
        # individuals that have reached the ant until those times.
        tree = {n: {} for n in range(self.graph['n_nodes'])}
        # Fill the tree
        for (g, r, d) in self.sorted_edges('time'):
            t = d['time']
            # Add the already existing interactions of the receiver
            try:
                t_max = max(tree[r].keys())
                tree[r][t] = {g} | tree[r][t_max]
            except:
                tree[r][t] = {g}
            # Add the already existing interactions of the giver
            try:
                t_max = max(tree[g].keys())
                tree[r][t] = tree[r][t] | (tree[g][t_max])
            except:
                None

        # The average #########################################################
        # Create a dictionary with time points as keys and the number of
        # unique 'reachings' at (!) those points as values
        unique = {}
        for ant in tree:
            times = sorted(tree[ant])
            for n, t in enumerate(times):
                if t not in unique:
                    if n > 0:
                        unique[t] = (len(tree[ant][times[n]])
                                     - len(tree[ant][times[n-1]]))
                    else:
                        unique[t] = len(tree[ant][times[n]])
                else:
                    if n > 0:
                        unique[t] += (len(tree[ant][times[n]])
                                      - len(tree[ant][times[n-1]]))
                    else:
                        unique[t] += len(tree[ant][times[n]])

        # Convert the dictionary to two sorted lists that contain the time
        # points and the 'reachings' until (!) those points respectively
        times = sorted(unique.keys())
        values = np.cumsum([float(unique[t]) for t in times])
        # Adding zeros for a proper beginning of the plot
        times = [0] + times
        values = np.insert(values, 0, 0)
        # Add last value for a proper ending
        if times[-1] != self.graph['time_window'][1]:
            times.append(self.graph['time_window'][1])
            values = np.append(values, values[-1])
        # Calculate the average
        values /= self.graph['n_nodes']

        # Save the data to <self.path>+<filename>
        if save:
            # Write the data
            with open(os.path.join(self.path, filename), 'w') as file:
                #file.write('# average\n')
                for n in range(len(times)):
                    file.write(str(times[n]) + '\t' + str(values[n]) + '\n')

        if plot:
            # Create the figure for plotting
            if axes is None:
                plt.figure()
                #plt.title('plot_n_reached_by(%s)' % self.path)
                plt.xlabel('Time step')
                plt.ylabel('number of individuals an individual was reached by'
                           ' directly and indirectly')
            # Plot as step function
            if cut:  # Cut the plot where 100% are reached
                maxindex = values.argmax()
                values = values[:maxindex+1]
                times = times[:maxindex+1]
            plt.step(times, values, where='post', linewidth=2, color='black')

        # The individuals #####################################################
        #i = -1  # For writting the ant number to the file
        for ant in tree:
        #    i += 1
            times = sorted(tree[ant].keys())
            values = [len(tree[ant][point]) for point in times]
            # Add zeros for a proper beginning of the plot
            times = [0] + times
            values = [0] + values
            # Add last value for a proper ending if necessary
            if times[-1] != self.graph['time_window'][1]:
                times.append(self.graph['time_window'][1])
                values.append(values[-1])

            if plot:
                # Plot as step function
                if cut:  # Cut the plot where 100% are reached
                    maxindex = values.index(max(values))
                    values = values[:maxindex+1]
                    times = times[:maxindex+1]
                plt.step(times, values, where='post', alpha=0.3)

    def plot_fed(self,
                 plot=True,
                 axes=None,
                 save=True,
                 filename='fed.dat'):
        """ Plot the number of already directly fed individuals over time.
        Plot ends when last individual ist fed. Pickups from the food source
        are not included. """
        # TODO Allow plotting to axes instance

        fed = set({})
        times = [0]
        # Calculate the data
        for (g, r, d) in self.sorted_edges('time'):
            if r not in fed:
                times.append(d['time'])
                fed.add(r)
        values = list(range(len(times)))
        # Plot as step function
        if plot:
            plt.figure()
            #plt.title('plot_fed(%s)' % self.path)
            plt.xlabel('Time step')
            plt.ylabel('Total number of fed ants')
            plt.step(times, values, where='post', color='black', linewidth=2)
            # Adjust axes limits
            adjust_axes()
            plt.tight_layout()

        # Save the data to a file
        if save:
            # Write the data
            with open(os.path.join(self.path, filename), 'w') as file:
                for n in range(len(times)):
                    file.write(str(times[n]) + ' ' + str(values[n]) + '\n')


###############################################################################
