import pandas as pd
import pathlib
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# Create folder if it does not exist
pathlib.Path('output').mkdir(exist_ok=True)


# support function to plot clusters
def plot_cluster(df_x, df_y, df_class, file_path=None, x_label='x', y_label='y', title='clustering',
                 centers=None, plot=None):
    fig, ax = plt.subplots()
    cluster_plot = ax.scatter(df_x, df_y, c=df_class, s=40, alpha=0.6, marker='o', cmap='brg')
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], marker='s',
                   c="white", alpha=1, s=75, edgecolor='black')
        for i, c in enumerate(centers):
            ax.scatter(c[0], c[1], marker='$%d$' % i, s=40, alpha=1, edgecolor='r')

    # plot labels, title and legends
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    legend = ax.legend(*cluster_plot.legend_elements(), title="classes",
                       bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.add_artist(legend)
    plt.tight_layout()
    if plot is not None:
        plt.show()
    if file_path:
        print('saving plot of resultant cluster into', file_path)
        fig.savefig(file_path, bbox_inches='tight')


print('*' * 20)
print('Question 1')
print('*' * 20)

# Read csv for Q1
print('Read the csv question_1.csv...')
df_q1 = pd.read_csv('specs/question_1.csv')

# details of data
# # df_q1.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 16 entries, 0 to 15
# Data columns (total 2 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   x       16 non-null     float64
#  1   y       16 non-null     float64

# plot original data
# df_q1.plot(kind='scatter',x='x', y='y', s=100, alpha=1, marker='o', cmap='brg')
# plt.xlabel('x', fontsize=18)
# plt.ylabel('y', fontsize=16)
# plt.title('question_1 data')
# plt.tight_layout()
# # plt.show()

# Q1.1 define k-means clustering model and run algorithm : 3 clusters, random state = 0.
kmeans_q1 = KMeans(n_clusters=3, random_state=0)
print('run kmeans on data in question_1.csv...')
kmeans_q1.fit(df_q1)
# print(kmeans_q1.cluster_centers_)
# [[16.64 15.16], [ 5.15  7.56], [ 1.3  22.8 ]]
df_q1['cluster'] = kmeans_q1.predict(df_q1)

# Q1.2 save resultant data with assigned cluster
print('save data with assigned cluster to output/question_1.csv...')
df_q1.to_csv('output/question_1.csv', index=False)

# Q1.3 save plot of clusters in pdf file
center_list = kmeans_q1.cluster_centers_
plot_cluster(df_q1['x'], df_q1['y'], df_q1['cluster'], "output/question_1.pdf",
             x_label='x', y_label='y', title='k-means clustering',
             centers=center_list)

print('*' * 20)
print('Question 2')
print('*' * 20)

# Read csv for Q2
print('Read the csv question_2.csv...')
df_q2 = pd.read_csv('specs/question_2.csv')

# Q2.1 Discard the columns NAME, MANUF, TYPE, and RATING from dataframe
df_q2.drop(['NAME', 'MANUF', 'TYPE', 'RATING'], axis=1, inplace=True)


# details of data
# print range of values in data features
# print(df_q2.describe())
#          CALORIES    PROTEIN        FAT  ...      SHELF     WEIGHT       CUPS
# count   77.000000  77.000000  77.000000  ...  77.000000  77.000000  77.000000
# min     50.000000   1.000000   0.000000  ...   1.000000   0.500000   0.250000
# max    160.000000   6.000000   5.000000  ...   3.000000   1.500000   1.500000

# Q2.2 define k-means clustering model and run algorithm : clusters=5, random state=0, runs=5, optimization steps=100
kmeans_q2_2 = KMeans(n_clusters=5, random_state=0, n_init=5, max_iter=100)
print('run kmeans on data in question_2.csv : n_clusters=5, random_state=0, n_init=5...')
df_q2['config1'] = kmeans_q2_2.fit_predict(df_q2)

# Q2.3 define k-means clustering model and run algorithm : clusters=5, random state=0, runs=100, optimization steps=100
kmeans_q2_3 = KMeans(n_clusters=5, random_state=0, n_init=100, max_iter=100)
print('run kmeans on data in question_2.csv : n_clusters=5, random_state=0, n_init=100...')
df_q2['config2'] = kmeans_q2_3.fit_predict(df_q2.drop(['config1'], axis=1))

# Q2.4 discussion of results in 2.2 and 2.3 is included in /report.pdf

# Q2.5 define k-means clustering model and run algorithm : clusters=3, random state=0, runs=100, optimization steps=100
# drop result columns in from previous stages
kmeans_q2_5 = KMeans(n_clusters=3, random_state=0, n_init=100, max_iter=100)
print('run kmeans on data in question_2.csv : n_clusters=3, random_state=0, n_init=100...')
df_q2['config3'] = kmeans_q2_5.fit_predict(df_q2.drop(['config1', 'config2'], axis=1))

## ******* Attempt kmeans on dimensionality reduced data *******
## USE SAME MODEL PARAMETERS AS Q2.5 IN FOR K-MEANS model definition
# reduce dimensionality using Principle component analysis; so that at least 95% of variance is explained
df_q2_PCA = df_q2.drop(['config1', 'config2', 'config3'], axis=1)
pca_model = PCA(n_components=0.90)
np_Q2_PCA = pca_model.fit_transform(df_q2_PCA)
df_Q2_PCA = pd.DataFrame(np_Q2_PCA)
# k-means
kmeans_q2_PCA = KMeans(n_clusters=3, random_state=0, n_init=100, max_iter=100)
print('run kmeans on PCA reduced data in question_2.csv : n_clusters=3, random_state=0, n_init=100...')
df_Q2_PCA['PCA_labels'] = kmeans_q2_PCA.fit_predict(df_Q2_PCA)
# plot of clusters
center_list = kmeans_q2_PCA.cluster_centers_
plot_cluster(df_Q2_PCA[0], df_Q2_PCA[1], df_Q2_PCA['PCA_labels'], 'output/question_2_PCA.pdf',
             x_label='dim1', y_label='dim2', title='PCA reduced data : k-means clustering',
             centers=center_list)
#labels of previous clustering and dimension reduced clustering

# save resultant labels from Q2.5 clustering and dimensionality reduced clustering
print('save cluster labels to output/question_2_PCA.csv...')
df_q2[['config1','config2','config3']].join(df_Q2_PCA[['PCA_labels']]).to_csv('output/question_2_PCA.csv', index=False)

# Q2.6 silhouette_score metric of classification is calculated to measure effectiveness of classification
print('silhouette_score - n_clusters=5, random_state=0, n_init=5, max_iter=100 : {:.4f}'.format(
    silhouette_score(df_q2.iloc[:,:-3], df_q2[['config1']])))
print('silhouette_score - n_clusters=5, random_state=0, n_init=100, max_iter=100 : {:.4f}'.format(
    silhouette_score(df_q2.iloc[:,:-3], df_q2[['config2']])))
print('silhouette_score - n_clusters=3, random_state=0, n_init=100, max_iter=100 : {:.4f}'.format(
    silhouette_score(df_q2.iloc[:,:-3], df_q2[['config3']])))
print('silhouette_score PCA - n_clusters=3, random_state=0, n_init=100, max_iter=100 : {:.4f}'.format(
    silhouette_score(df_Q2_PCA.iloc[:,:-1], df_Q2_PCA[['PCA_labels']])))

# Q2.7 save resultant data with assigned clusters
print('save data with assigned cluster to output/question_2.csv...')
df_q2.to_csv('output/question_2.csv', index=False)

print('*' * 20)
print('Question 3')
print('*' * 20)

# Read csv for Q3
print('Read the csv question_3.csv...')
df_q3 = pd.read_csv('specs/question_3.csv')

# details of data
# # df_q3.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 322 entries, 0 to 321
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   ID      322 non-null    int64
#  1   x       322 non-null    float64
#  2   y       322 non-null    float64

# plot data
# df_q3.plot(kind='scatter',x='x', y='y', s=100, alpha=1, marker='o', cmap='brg')
# plt.xlabel('x', fontsize=18)
# plt.ylabel('y', fontsize=16)
# plt.title('question_3 data')
# plt.tight_layout()
# # plt.show()


# Discard the column ID from dataframe
df_q3.drop(['ID'], axis=1, inplace=True)

# Q3.1 define k-means clustering model and run algorithm : clusters=7, random state=0, runs=5, optimization steps=100
kmeans_q3_1 = KMeans(n_clusters=7, random_state=0, n_init=5, max_iter=100)
print('run kmeans on data in question_3.csv : n_clusters=7, random_state=0, n_init=5...')
df_q3['kmeans'] = kmeans_q3_1.fit_predict(df_q3)

# Q3.2 save plot of clusters in pdf file
center_list = kmeans_q3_1.cluster_centers_
plot_cluster(df_q3['x'], df_q3['y'], df_q3['kmeans'], "output/question_3_1.pdf",
             x_label='x', y_label='y', title='k-means clustering',
             centers=center_list)

# Min-Max transformation of attribute 'x' and 'y'
df_q3['x'] = MinMaxScaler(feature_range=[0.0, 1.0]).fit_transform(df_q3[['x']])
df_q3['y'] = MinMaxScaler(feature_range=[0.0, 1.0]).fit_transform(df_q3[['y']])

# Q3.3 define dbscan clustering model and run algorithm : epsilon=0.04, minimum points for neighborhood evaluation=4
dbscan_q3_3 = DBSCAN(eps=0.04, min_samples=4)
print('run DBSCAN on data in question_3.csv : epsilon=0.04, minimum points for neighborhood evaluation=4...')
df_q3['dbscan1'] = dbscan_q3_3.fit_predict(df_q3.drop(['kmeans'], axis=1))

# save plot of clusters in pdf file
# center_list = dbscan_q3_3.cluster_centers_
plot_cluster(df_q3['x'], df_q3['y'], df_q3['dbscan1'], "output/question_3_2.pdf",
             x_label='x', y_label='y', title='DBSCAN clustering : epsilon=0.04')

# Q3.4 define dbscan clustering model and run algorithm : epsilon=0.08, minimum points for neighborhood evaluation=4
dbscan_q3_4 = DBSCAN(eps=0.08, min_samples=4)
print('run DBSCAN on data in question_3.csv : epsilon=0.08, minimum points for neighborhood evaluation=4...')
df_q3['dbscan2'] = dbscan_q3_4.fit_predict(df_q3.drop(['kmeans', 'dbscan1'], axis=1))

# save plot of clusters in pdf file
# center_list = dbscan_q3_3.cluster_centers_
plot_cluster(df_q3['x'], df_q3['y'], df_q3['dbscan2'], "output/question_3_3.pdf",
             x_label='x', y_label='y', title='DBSCAN clustering: epsilon=0.08'
             )

# Q3.5 save resultant data with assigned clusters
print('save data with assigned cluster to output/question_3.csv...')
df_q3.to_csv('output/question_3.csv', index=False)

# Q3.6
# silhouette_score metric of classification is calculated to measure effectiveness of classification
print('silhouette_score - K-MEANS n_clusters=7, random_state=0, n_init=5, max_iter=100 : {:.4f}'.format(
    silhouette_score(df_q3.iloc[:,:-3], df_q3[['kmeans']])))
print('silhouette_score - DBSCAN epsilon=0.04, minimum points for neighborhood evaluation=4 : {:.4f}'.format(
    silhouette_score(df_q3.iloc[:,:-3], df_q3[['dbscan1']])))
print('silhouette_score - DBSCAN epsilon=0.08, minimum points for neighborhood evaluation=4 : {:.4f}'.format(
    silhouette_score(df_q3.iloc[:,:-3], df_q3[['dbscan2']])))

# Q3.6
# calinski_harabasz_score metric of classification is calculated to measure effectiveness of classification
print('calinski_harabasz_score - K-MEANS n_clusters=7, random_state=0, n_init=5, max_iter=100 : {:.4f}'.format(
    calinski_harabasz_score(df_q3.iloc[:,:-3], df_q3[['kmeans']])))
print('calinski_harabasz_score - DBSCAN epsilon=0.04, minimum points for neighborhood evaluation=4 : {:.4f}'.format(
    calinski_harabasz_score(df_q3.iloc[:,:-3], df_q3[['dbscan1']])))
print('calinski_harabasz_score - DBSCAN epsilon=0.08, minimum points for neighborhood evaluation=4 : {:.4f}'.format(
    calinski_harabasz_score(df_q3.iloc[:,:-3], df_q3[['dbscan2']])))
