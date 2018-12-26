# Birch

데이터를 Characteristic Feature nodes (CF Nodes)들의 집합으로 압축하여 Characteristic Feature Tree (CFT)를 구축한다.

CF Nodes들은 여러 CF Subclusters를 가지고, CF Subcluster들은 CF Nodes를 children으로 가질 수 있다.

CF Subclusteres가 저장하는 정보들:

- subcluster 안의 sample 개수
- 모든 sample 벡터들의 Linear Sum, Squared Sum
- Centroids(linear sum / n_samples)와 이 squared norm 값

Birch 알고리즘의 parameter들:

- branching factor: node 안의 최대 subcluster 수
- threshold: 기존 subcluster에 새로 넣을 수 있는 sample과의 최대 거리

Birch 알고리즘은 입력 데이터를 여러 subcluster로 data reduction 하는 것으로 볼 수 있으므로, 이에 따라 data reduction 결과를 global clusterer에서 clustering할 수도 있다. (n_cluster 정할 경우)

## References

- [scikit-learn documentation](https://scikit-learn.org/stable/modules/clustering.html#birch)
- Tian Zhang, Raghu Ramakrishnan, Maron Livny BIRCH: An efficient data clustering method for large databases. [http://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf]