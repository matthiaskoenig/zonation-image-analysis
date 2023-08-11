

## General concepts

### Delaunay triangulation
Close relationship between Delaunay triangulation and the Voronoi diagram.

The Delaunay triangulation of a discrete point set in general position corresponds to the dual graph of the Voronoi diagram for P. The circumcenters of Delaunay triangles are the vertices of the Voronoi diagram. 
https://en.wikipedia.org/wiki/Delaunay_triangulation

For modelling terrain or other objects given a point cloud, the Delaunay triangulation gives a nice set of triangles to use as polygons in the model. 

Typically, the domain to be meshed is specified as a coarse simplicial complex; for the mesh to be numerically stable, it must be refined, for instance by using **Ruppert's algorithm**.

- **Delaunay refinement**: https://en.wikipedia.org/wiki/Delaunay_refinement#Ruppert's_algorithm

- A subdivision S is a maximal planar subdivision such that adding one edge connecting two vertices of S destroys the planarity of S. What does destroying the planarity mean? Adding an edge intersects at least one existing edge. 

Deﬁnition (triangulation of a point set)
Let P be a set of points in the plane. A triangulation of the point set P is a maximal planar subdivision which has P as its vertex set.
