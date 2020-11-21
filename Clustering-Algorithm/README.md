### Unsupervised learning - Clustering
---
A clustering model tries to group together similar training data points into “Classes”. New data belonging to a class C would have similar characteristics like existing class member data points. Also, it shall be closer to those training data points in n dimensional feature space. This “Tendency of cooccurrence” is used in unsupervised classification method of clustering.

## k-means clustering
* [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering#Algorithms) is a partition based algorithm. It fits algorithm to find centroids which define an “area of coverage”.
* Classically, when k-means model is fitted, it starts with 3 random centroids. By default,sklearn uses kmeans++ algorithm, which initializes a datapoint as first  centroid  using  random_state.  Remaining  centroids  are  synthetically  generated  so  that distance between all clusters is maximized; which shall yield better results
* k-means forms **convex shaped clusters**

### Results

 * Convex boundaries formed due to centroid mechanism produce wrong clustering.
 *In  case  of  cluster 2and 3,  we clearly see that original data has 2 distinct shapes resembling “C” and “✓”;  and  their edge points are misclustered.
 * Also, all outliers are made a part of nearby clusters.

<br>

## DBSCAN clustering
[DBSCAN clustering](https://en.wikipedia.org/wiki/DBSCAN),  in  contrast  does  not  define  a  central  tendency  which  defines  range  of cluster.
* DBSCAN clusters  areas  of  high  density  separated  by  areas  of  low  density. 
It  defines distance regulator **epsilon**, which is maximum distance between data samples to be considered as neighbours; hence part of cluster. 
* Dense part of cluster is formed by **core-samples**; which have more  than **min_samples** within  distance  less  than epsilon. DBSCAN allows to declare outliers.
* **Non-core  samples** are  in  proximity  of  a  cluster, but do not satisfy min_samplescriteria.
* Any data point which are  not within  eps  distance  of  a  cluster  data  point  is marked  as  outlier. 
* As,  distance  for  all  samples  is  checked pairwise  with  each  other, DBSCAN  is **memory expensive algorithm**.

### Results

 * Smaller value of epsilon = 0.04 force stringent clustering. Hence, highly dense data points are  part  of  cluster.  Whereas,  less  dense points  which  formed  cluster  4  and  5  in  case  of kmeans are defined as outliers.

<br>
 * When epsilon is made   less   stringent   i.e.   0.08,   more non-core samples   are   incorporated   in   cluster. But, it lost  distinct  cluster  3&4 obtained in previously. Both are fused into cluster 3, as shown in Figure.Same thing happens with cluster 1&2 in previous Figure, which fuse together into cluster2.

<br>

## Quantitative Evaluation
* Performance of all algorithms is evaluated with **[silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)**. It is average of silhouette coefficients; which  gives a measure of how close each point in one cluster is to points in the neighbouring clusters.
Kmeans and dbscan2 have good scores. 
* But in  case  of  dbscan1, silhouette  scorefor  DBSCAN  is heavily penalized  given  presence  of outliers,  which  essentially  gave  no  cluster. So, silhoutte score is  not  powerful  metric  to  assess  DBSCAN. Hence, **[calinski_harabasz  score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html#sklearn.metrics.calinski_harabasz_score)**[4]  is  used, which  is defined as  ratio  between  the  within-cluster  dispersion and  the  between-cluster  dispersion.  Minimum  score  shows  good  clustering. 

## Inference
dbscan model with eps=0.8 and min_samples=4 is model of choice for given dataset. But,  results  can  be  finetuned  by  trying  out  values  for  eps  in  range  0.04 < eps < 0.08  to  get  optimal classification which leaves out outliers and disintegrate fused clusters discussed in previously.