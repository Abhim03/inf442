import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def main():
    # usage message
    if len(sys.argv) != 5:
        print(f'''
Usage: {sys.argv[0]} csv_file lc nb_clusters ld

where:
        - lc is the comma-separated list of covariates to be used for clustering
          (write \":\" if all covariates should be used)

        - ld is the comma-separated list of covariates to be used for display
          (must be a subset of the covariates used for clustering)

Examples
        {sys.argv[0]} ../csv/cereals.csv : 2 protein,weight,cups

        {sys.argv[0]} ../csv/cereals.csv protein,fat,weight 3 protein,weight
''')
        sys.exit(1)


    # manage arguments
    clust_col_labels = sys.argv[2].split(',')
    nb_clusters = int(sys.argv[3])
    disp_col_labels = sys.argv[4].split(',')

    # read in data
    data = pd.read_csv(sys.argv[1], index_col=0, delimiter='\t')
    if not clust_col_labels[0].startswith(':'):
        data = data.loc[:,clust_col_labels]
        print("Selected labels: " + str(clust_col_labels))
    print('Read in the (' + str(data.shape[0]) + ',' + str(data.shape[1]) + ') data:')
    print(data)

    # normalize data using Sklearn
    print("\nMin-max normalizing data...", end='')
    scaler = MinMaxScaler()
    ndata = scaler.fit_transform(data)
    print(" done")

    # cluster normalized data using Sklearn
    print("\nComputing k-means clustering on normalized data...", end='')
    cl = KMeans(n_clusters=nb_clusters, random_state=0, init='k-means++', n_init='auto').fit(ndata)
    print(" done")

    # generate score curve for elbow heuristic using Sklearn
    print("\nGenerating the score curve...", end='')
    min_nb_clusters = 11 # Change to 2!
    max_nb_clusters = 10
    km = [] # Add the appropriate KMeans objects
    score = [km[i].fit(ndata).inertia_ for i in range(max_nb_clusters - min_nb_clusters + 1)]
    print(" done")

    ########### the rest of the code below is just for display ###########

    fig =  plt.figure(figsize=plt.figaspect(0.33))

    # Plot labeled input data and cluster centers in disp_col_labels space
    if len(disp_col_labels) <= 2:
        ax = fig.add_subplot(1, 3, 1)
        ax.set_title("clusters in " + str(disp_col_labels) + " space")
        data.plot(ax=ax, kind="scatter", x=disp_col_labels[0], y=disp_col_labels[1], c=cl.labels_, marker='o', s=50, colormap=plt.cm.rainbow, colorbar=False, legend=True)
        centers = scaler.inverse_transform(cl.cluster_centers_)
        x = data.columns.get_loc(disp_col_labels[0])
        y = data.columns.get_loc(disp_col_labels[1])
        ax.scatter(centers[:,x], centers[:,y], c=list(range(nb_clusters)), marker='*', s=50, cmap=plt.cm.rainbow)

    else :  # len(disp_col_labels) >= 3
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.set_title("clusters in " + str(disp_col_labels) + " space")
        x = data.columns.get_loc(disp_col_labels[0])
        y = data.columns.get_loc(disp_col_labels[1])
        z = data.columns.get_loc(disp_col_labels[2])
        ax.scatter3D(data.values[:,x], data.values[:,y], data.values[:,z], c=cl.labels_, marker='o', s=50, cmap=plt.cm.rainbow)
        centers = scaler.inverse_transform(cl.cluster_centers_)
        ax.scatter3D(centers[:,x], centers[:,y], centers[:,z], c=list(range(nb_clusters)), marker='*', s=50, cmap=plt.cm.rainbow)

    #Plot labeled input data in cluster-distance space
    cldata = cl.transform(ndata)
    if nb_clusters <= 2:
        ax = fig.add_subplot(1, 3, 2)
        ax.set_title("clusters in cluster-distance space")
        ax.scatter(cldata[:,0], cldata[:,1], c=cl.labels_, marker='o', s=50, cmap=plt.cm.rainbow)

    else :  # nb_clusters >= 3
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.set_title("clusters in cluster-distance space")
        ax.scatter3D(cldata[:,0], cldata[:,1], cldata[:,2], c=cl.labels_, marker='o', s=50, cmap=plt.cm.rainbow)

    # plot score curve
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("score curve")
    ax.plot(range(min_nb_clusters, max_nb_clusters + 1), score)

    fig.savefig('result.png')

if __name__ == '__main__':
    main()
