#!/usr/bin/env python3
""" Graph series class. """

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from helpers import read_data, adjust_axes, truncate_colormap
from TempGraph import TempGraph


class GraphSeries(object):

    def __init__(self, path):

        # Set the path of the directory whose folders are to be analyzed
        self.path = path
        # Create a list of the folders (the names have to be numbers)
        self.folders = sorted([folder
                               for folder in os.listdir(self.path)
                               if folder.isdigit()
                               ],
                              key=lambda x: int(x)
                              )

###############################################################################

    def analyze(self):

        """ Analyzes all the folders in <self.path> that contain data. Only
        accepts numbers as names of folders. """
        i = 0
        for folder in self.folders:
            i += 1
            path = self.path + folder + '/'
            print ('Folder:', path, '('+str(i)+'/'+str(len(self.folders))+')')
            G = TempGraph()
            G.create_graph(path=path)
            G.plot_degree_histograms(plot=False)
            G.plot_fed(plot=False)
            G.plot_n_given(plot=False)
            #G.plot_n_received(plot=False)
            G.plot_n_reached(plot=False)
            #G.plot_n_reached_by(plot=False)
            del G

###############################################################################

    def plot_degree_histogram(self,
                              filename='degrees.dat',
                              fit = True,
                              colors=False,
                              valuemax=None,
                              totalmax=None):
        """ Plot the average degree histogram and fit a Gaussian function to
        it. """
        # TODO Allow plotting to axes instance
        # TODO Maybe allow saving of the data

        degrees = []
        # Read the degree data from the file
        for folder in self.folders:
            path = self.path + folder + '/'
            with open(path+filename) as file:
                for line in file:
                    degrees.append(int(line))
        # Calculate the histogram
        hist, edges = np.histogram(degrees, bins=(range(min(degrees),
                                                        max(degrees)+2)))
        hist = hist / len(self.folders)
        # Plot the histogram
        plt.figure()
        #plt.title('Degree histogram (%s)' % self.path)
        plt.xlabel('Number of interactions')
        plt.ylabel('Average number of ants per run')

        print('Max degree:', max(edges))

        #plt.bar(left=(edges[:-1]-0.5), height=hist, width=1, color='grey', label='Data')

        # For diffusion plot
        if colors:
            plt.bar(left=(edges[:-1]-0.5), height=hist, width=1,
                    color=truncate_colormap(cm.spectral, valuemax=valuemax, totalmax=totalmax)((edges[:-1]/max(edges[:-1]))),
                    edgecolor=truncate_colormap(cm.spectral)((edges[:-1]/max(edges[:-1]))))
        else:
            plt.bar(left=(edges[:-1]-0.5), height=hist, width=1,
                          color='grey', label='Data')

        if fit:
            # Fit a Gaussian
            from scipy.optimize import curve_fit

            def gaus(x, a, x0, sigma):
                return a*np.exp(-(x-x0)**2/(2*sigma**2))

            x = edges[:-1]-0.5
            y = hist

            popt, pcov = curve_fit(gaus, x, y,
                                   p0=[1.5, np.mean(edges), np.std(edges)])
            print('a, x0, sigma:', popt)
            y_fit = gaus(x, popt[0], popt[1], popt[2])
            plt.plot(x, y_fit, linewidth=2, label='Fit')

            with open(os.path.join(self.path, 'gaus.txt'), 'w') as file:
                file.write(str(popt[0]) + '\t' + str(popt[1])
                           + '\t' + str(popt[2]))
        plt.legend(loc='best')
        adjust_axes()
        plt.tight_layout()

###############################################################################

    def plot_n_given(self,
                     filename='n_given.dat'):
        """ Plot the average number of individuals to which an individual has
        already given food via direct trophalaxis over time."""
        # HINT Extends the list of time points during the run. Is a little
        # faster than np_given_np() for small amounts of data (3 and 10 runs
        # of the simulation).
        # TODO Allow plotting to axes instance
        # TODO Maybe allow saving of the data

        # Create the figure for plotting
        plt.figure()
        #plt.title('plot_n_given(%s)' % self.path)
        plt.xlabel('Time step')
        plt.ylabel('Average number of unique outgoing interactions per run')

        # Create a list containing the values for each point in time (the time
        # points are the indeces of the list)
        points = [0]
        maxpoint = 0
        # Load the data from the folders
        for folder in self.folders:
            path = self.path + folder + '/'
            times, values = read_data(file=path+filename)

            # Plot the data of this run of the simulation as a step function
            plt.step(times, values, where='post', alpha=0.3)

            # Convert from step function to normal function
            #times, values = to_normal_function(times, values)

            # Extend the points list if necessary
            if maxpoint < times[-1]:
                for n in range(maxpoint, times[-1]):
                    points.append(points[maxpoint])
                maxpoint = times[-1]

            # Add the values
            for n in range(len(times)):
                try:  # List index out of range for the last n
                    for m in range(times[n], times[n+1]):
                        points[m] += values[n]
                except:  # Add the last value until the end of the points list
                    for m in range(times[n], len(points)):
                        points[m] += values[-1]

        # Calculate the average and create a list with the times
        points = np.array(points)/len(self.folders)
        times = list(range(len(points)))

        # Plot as step function
        plt.step(times, points, where='post', linewidth=2, color='black',
                 label='Average')
        adjust_axes()
        plt.legend(loc='best')

    def plot_n_given_np(self,
                        filename='n_given.dat'):
        """ Plot the average number of individuals to which an individual has
        already given food via direct trophalaxis over time."""
        # HINT Look first for the maximum time point to create point lists with
        # apropiate lengths. Is a little faster than plot_given() for large
        # amounts of data (100 runs of the simulation).
        # TODO Allow plotting to axes instance
        # TODO Maybe allow saving of the data

        # Create the figure for plotting
        plt.figure()
        #plt.title('plot_n_given_np(%s)' % self.path)
        plt.xlabel('Time step')
        plt.ylabel('Average number of unique outgoing interactions per run')

        # Find the maximum time
        maxtime = 0
        for folder in self.folders:
            path = self.path + folder + '/'
            with open(path+filename) as file:
                for line in file:
                    pass
                last = int(float(line.split()[0]))
                if maxtime < last:
                    maxtime = last

        # Create a list containing the values for each point in time (the time
        # points are the indeces of the list)
        points = np.zeros(maxtime)
        # Load the data from the folders
        for folder in self.folders:
            path = self.path + folder + '/'
            times, values = read_data(file=path+filename)

            # Plot the data of this run of the simulation as a step function
            plt.step(times, values, where='post', alpha=0.3)
            # Convert from step function to normal function
            #times, values = to_normal_function(times, values)

            # Add the values
            for n in range(len(times)):
                try:  # List index out of range for the last n
                    for m in range(times[n], times[n+1]):
                        points[m] += values[n]
                except:  # Add the last value until the end of the points list
                    for m in range(times[n], len(points)):
                        points[m] += values[-1]

        # Calculate the average and create a list with the times
        points = points / len(self.folders)
        times = list(range(len(points)))
        # Plot as step function
        plt.step(times, points, where='post', linewidth=2, color='black',
                 alpha=0.3)

    # HINT There is no need of a function n_received() for a graph series
    # because the average of of n_given() equals the average of n_received()
    # for each TempGraph

    def plot_n_reached(self,
                       cut=True,
                       filename='n_reached.dat'):
        """ Plot the average number of individuals to which an individual
        has already given food via direct trophalaxis over time. """
        # TODO Allow plotting to axes instance
        # TODO Maybe allow saving of the data

        # Create the figure for plotting
        plt.figure()
        #plt.title('plot_n_reached_by(%s)' % self.path)
        plt.xlabel('Time step')
        plt.ylabel('Average number of reached ants per run')

        # Create a list containing the values for each point in time (the time
        # points are the indeces of the list)
        points = [0]
        maxpoint = 0
        # Load the data from the folders
        for folder in self.folders:
            path = self.path + folder + '/'
            times, values = read_data(file=path+filename)

            # Plot the data of this run of the simulation as a step function
            if cut:
                maxindex = values.index(max(values))
                plt.step(times[:maxindex+1], values[:maxindex+1],
                         where='post', alpha=0.3)
            else:
                plt.step(times[:maxindex+1], values[:maxindex+1], where='post',
                         alpha=0.3)
            # Convert from step function to normal function
            #times, values = to_normal_function(times, values)

            # Extend the points list if necessary
            if maxpoint < times[-1]:
                for n in range(maxpoint, times[-1]):
                    points.append(points[maxpoint])
                maxpoint = times[-1]
            # Add the values
            for n in range(len(times)):
                try:  # List index out of range for the last n
                    for m in range(times[n], times[n+1]):
                        points[m] += values[n]
                except:  # Add the last value until the end of the points list
                    for m in range(times[n], len(points)):
                        points[m] += values[-1]

        # Cuts the points where they reach their maximum value (-3 sec)
        if cut:
            last = points[-1]
            n = 0
            for point in points:
                n += 1
                if point == last:
                    points = points[:n]
                    break

        # Calculate the average and create a list with the times
        points = np.array(points)/len(self.folders)
        times = list(range(len(points)))
        # Plot as a step function
        plt.step(times, points, where='post', linewidth=2, color='black',
                 label='Average')
        adjust_axes()
        plt.legend(loc='best')

    def plot_n_reached_by(self,
                          cut=True,
                          filename='n_reached_by.dat'):
        """ Plot the average number of individuals to which an individual
        has already given food via direct trophalaxis over time. """
        # TODO Allow plotting to axes instance
        # TODO Maybe allow saving of the data

        # Create the figure for plotting
        plt.figure()
        #plt.title('plot_n_reached_by(%s)' % self.path)
        plt.xlabel('Time step')
        plt.ylabel('average number of individuals an individual was reached by'
                   ' directly and indirectly per run')

        # Create a list containing the values for each point in time (the time
        # points are the indeces of the list)
        points = [0]
        maxpoint = 0
        # Load the data from the folders
        for folder in self.folders:
            path = self.path + folder + '/'
            times, values = read_data(file=path+filename)

            # Plot the data of this run of the simulation as a step function
            if cut:
                maxindex = values.index(max(values))
                plt.step(times[:maxindex+1], values[:maxindex+1],
                         where='post', alpha=0.3)
            else:
                plt.step(times[:maxindex+1], values[:maxindex+1], where='post',
                         alpha=0.3)
            # Convert from step function to normal function
            #times, values = to_normal_function(times, values)

            # Extend the points list if necessary
            if maxpoint < times[-1]:
                for n in range(maxpoint, times[-1]):
                    points.append(points[maxpoint])
                maxpoint = times[-1]
            # Add the values
            for n in range(len(times)):
                try:  # List index out of range for the last n
                    for m in range(times[n], times[n+1]):
                        points[m] += values[n]
                except:  # Add the last value until the end of the points list
                    for m in range(times[n], len(points)):
                        points[m] += values[-1]

        # Cuts the points where they reach their maximum value (-3 sec)
        if cut:
            last = points[-1]
            n = 0
            for point in points:
                n += 1
                if point == last:
                    points = points[:n]
                    break

        # Calculate the average and create a list with the times
        points = np.array(points)/len(self.folders)
        times = list(range(len(points)))
        # Plot as a step function
        plt.step(times, points, where='post', linewidth=2, color='black')

    def plot_fed(self,
                 filename='fed.dat'):
        """ Plot the average number of already fed individuals over time. """
        # TODO Allow plotting to axes instance
        # TODO Maybe allow saving of the data

        # Create the figure for plotting
        plt.figure()
        #plt.title('plot_feeded(%s)' % self.path)
        plt.xlabel(r'Time step')
        plt.ylabel(r'Average number of fed ants per run')

        # Create a list containing the values for each point in time (the time
        # points are the indeces of the list)
        points = [0]
        maxpoint = 0
        # Load the data from the folders
        for folder in self.folders:
            path = self.path + folder + '/'
            times, values = read_data(file=path+filename)
            # Plot the data of this run of the simulation as step function
            plt.step(times, values, where='post', alpha=0.3)
            # Convert from step function to normal function
            #times, values = to_normal_function(times, values)

            # Extend the points list if necessary
            if maxpoint < times[-1]:
                for n in range(maxpoint, times[-1]):
                    points.append(points[maxpoint])
                maxpoint = times[-1]
            # Add the values
            for n in range(len(times)):
                try:  # List index out of range for the last n
                    for m in range(times[n], times[n+1]):
                        points[m] += values[n]
                except:  # Add the last value until the end of the points list
                    for m in range(times[n], len(points)):
                        points[m] += values[-1]

        # Calculate the average and create a list with the times
        points = np.array(points)/len(self.folders)
        times = list(range(len(points)))
        # Plot as a step function
        plt.step(times, points, where='post', linewidth=2, color='black',
                 label='Average')
        adjust_axes()
        plt.legend(loc='best')

    def calc_avg_emptytime(self,
                           filename='emptytime.txt'):
        """ Calculate the average point in time at which the food source is
        empty and its standard deviation and save them to <filename>. """

        emptytimes = []
        for folder in self.folders:
            path = self.path + folder + '/'
            with open(path + filename) as file:
                emptytimes.append(int(file.readline().split()[0]))
        #print(emptytimes)
        #print('Average: %f' % np.mean(emptytimes))
        #print('Standard deviation: %f' % np.std(emptytimes))

        # Write the data
        with open(os.path.join(self.path, filename), 'w') as file:
            file.write(str(np.mean(emptytimes)) + '\t'
                       + str(np.std(emptytimes)))

    def calc_avg_degrees_at_positions(self,
                                      filename='degrees_at_positions.txt',
                                      vmax=None):
        """ Plot the average number of interactions per grid point using a
        colormap. """

        degreedict = {}
        for folder in self.folders:
            path = self.path + folder + '/'
            with open(path + filename) as file:
                for line in file:
                    data = line.split()
                    if (int(data[0]), int(data[1])) in degreedict.keys():
                        degreedict[(int(data[0]),
                                    int(data[1]))] += int(data[2])
                    else:
                        degreedict[(int(data[0]),
                                    int(data[1]))] = int(data[2])
        positions = np.zeros((15, 15))
        for key in degreedict.keys():
            positions[key[0], key[1]] = degreedict[key]/len(self.folders)
        #print(positions)
        positions = np.transpose(positions)
        positions = np.flipud(positions)
        plt.figure()
        plt.axis('off')
        plt.imshow(positions,
                   cmap=truncate_colormap(cm.spectral),
                   interpolation='none', vmax=vmax)
        cbar = plt.colorbar()
        cbar.set_label('Number of interactions', rotation=270, labelpad=30)
        plt.tight_layout()

    def calc_n_pickups(self,
                       filename='pickups.txt'):
        """ Calculate the average number of pickups and its standard deviation
        and save them in <filename>. """

        n_pickups = []
        for folder in self.folders:
            n_pickups.append(0)
            path = self.path + folder + '/'
            with open(path + filename) as file:
                for line in file:
                    data = line.split()
                    if data[0] != '#':
                        n_pickups[-1] += 1

        print('Average:', np.mean(n_pickups))
        print('Standard deviation:', np.std(n_pickups))

        # Write the data
        with open(os.path.join(self.path, filename), 'w') as file:
            file.write(str(np.mean(n_pickups)) + '\t'
                       + str(np.std(n_pickups)))

    def calc_n_interactions(self,
                            filename='interactions.txt'):

        """ Calculate the average number of interactions and its standard
        deviation and save the them in <filename>. """
        n_interactions = []
        for folder in self.folders:
            n_interactions.append(0)
            path = self.path + folder + '/'
            with open(path + filename) as file:
                for line in file:
                    data = line.split()
                    if data[0] != '#':
                        n_interactions[-1] += 1

        print('Average:', np.mean(n_interactions))
        print('Standard deviation:', np.std(n_interactions))

        # Write the data
        with open(os.path.join(self.path, filename), 'w') as file:
            file.write(str(np.mean(n_interactions)) + '\t'
                       + str(np.std(n_interactions)))

    def calc_amounts_exchanged(self,
                               filename='interactions.txt'):
        """ Calculate the total amount of food that was exchanged through
        trophollactic interactions and its standard deviation and save them in
        <amounts_exchanged.txt>. """

        amounts = []
        for folder in self.folders:
            amounts.append(0.0)
            path = self.path + folder + '/'
            with open(path + filename) as file:
                for line in file:
                    data = line.split()
                    if data[0] != '#':
                        amounts[-1] += float(data[-1])

        print('Average:', np.mean(amounts))
        print('Standard deviation:', np.std(amounts))

        # Write the data
        with open(os.path.join(self.path, 'amounts_exchanged.txt'), 'w') as file:
            file.write(str(np.mean(amounts)) + '\t'
                       + str(np.std(amounts)))

    def calc_amount_per_interaction(self,
                                     filename='interactions.txt'):
        """ Calculate the average amount of food that was exchanged through
        trophollactic interactions and its standard deviation and save them in
        <amounts_per_interaction.txt>. """

        amounts = []
        for folder in self.folders:
            path = self.path + folder + '/'
            with open(path + filename) as file:
                for line in file:
                    data = line.split()
                    if data[0] != '#':
                        amounts.append(float(data[-1]))

        print('Average:', np.mean(amounts))
        print('Standard deviation:', np.std(amounts))

        # Write the data
        with open(os.path.join(self.path, 'amount_per_interaction.txt'), 'w') as file:
            file.write(str(np.mean(amounts)) + '\t'
                       + str(np.std(amounts)))

    def calc_pickups_exact(self,
                           filename='pickups.txt'):
        """ Calculate the average amount of food that was picked up from the
        food source and its standard deviation and save them in
        <pickups_exact.txt>. """

        pickups = []
        for folder in self.folders:
            path = self.path + folder + '/'
            with open(path + filename) as file:
                for line in file:
                    data = line.split()
                    if data[0] != '#':
                        pickups.append(float(data[-1]))

        print('Average:', np.mean(pickups))
        print('Standard deviation:', np.std(pickups))

        # Write the data
        with open(os.path.join(self.path, 'pickups_exact.txt'), 'w') as file:
            file.write(str(np.mean(pickups)) + '\t'
                       + str(np.std(pickups)))
