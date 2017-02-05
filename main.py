#!usr/bin/env python3
""" Main function. """

import time
#import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt

#print (plt.rcParams['figure.figsize'])

from matplotlib import rc
rc('font',
   **{'family': 'serif',
      'serif': ['Palatino'],
      # 'family': 'sans-serif',
      # 'sans-serif': ['sans-serif'],
      'family': 'serif',
      'serif': ['Palatino'],
      'size': 18
      })
rc('text',
   usetex=True)
rc('figure',
   **{'figsize': (16, 9)  # ,
      #'dpi': 300
      }
   )

###############################################################################
start = time.clock()

#from GraphSeries import GraphSeries
#for n in [1,2,3,4,5,10,15,20,30,40,50,80,100]:
    #S = GraphSeries(path='../data/speed/' + str(n) + '/')
#S.analyze()
#S.plot_fed()
#S.plot_n_given()
#S.plot_n_given_np()
#S.plot_n_reached()
#S.plot_n_reached_by()
#S.plot_degree_histogram()
#S.calc_avg_emptytime()
#S.calc_avg_degrees_at_positions()
#S.calc_n_pickups()
#S.calc_n_interactions()
    #S.calc_amounts_exchanged()
    #S.calc_amount_per_interaction()
    #S.calc_pickups_exact()


from TempGraph import TempGraph
G = TempGraph()
G.create_graph(path='../data/simple/')
#G.draw_temporal(figsize=(16, 18), arrows=False)
#plt.savefig('../data/simple/temporal.png', dpi=150)
#G.draw_all_layouts(figsize=(16, 16), arrows=False)
#plt.savefig('../data/simple/aggregated.png', dpi=150)

#slices = G.create_time_slices(G.create_time_windows(n_windows=3))
#for graph in slices:
    #graph.draw_weighted()
#G.draw_weighted(figsize=(16, 16))
#plt.savefig('../data/simple/weighted.png', dpi=150)


# Distributions of characteristics
#G.plot_degree_distributions(n_windows=4)
#G.plot_degree_histogramms(n_windows=1)
#G.plot_centrality_histogramms()

# Temporal evolution
#G.draw_ordered_evolution(n_windows=3, lines=True, arrows=False)
#G.draw_aggregated_evolution(n_windows=3, directed=False)

# Interaction statistics
#G.plot_food_source()
G.plot_pickups()
#G.plot_amount_picked_up()
#G.plot_number_of_interactions()
#G.plot_amounts_exchanged()
#G.plot_total_amount_exchanged()
#G.plot_food_in_ants()

# Network statistics
#G.plot_n_given()
#G.plot_n_received()
#G.plot_n_reached()
#G.plot_n_reached_by()
#G.plot_fed()

print ('Total duration:', time.clock()-start)
