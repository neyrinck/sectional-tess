# By Mark Neyrinck, 

import numpy as np
from scipy.spatial import ConvexHull#,Voronoi,voronoi_plot_2d

def lift_paraboloid(xy, pot):
    """
    Produce vertices on a paraboloid, whose convex hull can be used to get 
    (generalized) Delaunay and Voronoi tessellations
    
    xy = generator positions within the space being tessellated
    pot = potential at these points = additive weight/"power"
        = z^2/2, where z is distance at each generator to the space being tessellated
    """
    
    z = np.sum(xy**2, axis=1) - 2. * pot
    return np.concatenate([xy.T, np.expand_dims(z, 0)], 0).T

class Delaunay_from_ConvexHull:
    """
    Measure (generalized) Delaunay triangluation from Convex Hull
    input: ch = scipy.spatial.ConvexHull object
    
    Delaunay needs further testing
    """
    def __init__(self, ch, faceup=[], valid=[]): 
        if len(faceup) == 0: 
            self.faceup = self.get_faceup(ch)
        else:
            self.faceup = faceup

        self.points = ch.points[:,:-1]
        self.simplices = ch.simplices[self.faceup]
        self.neighbors = ch.neighbors[self.faceup]
        self.equations = ch.equations[self.faceup]
        self.coplanar = ch.coplanar

    def get_faceup(self,ch):
        return np.where(np.dot(ch.equations[:,0:-1], [0, 0, -1]) > 1e-6)[0]


class Voronoi_from_ConvexHull:
    """
    Measure (generalized) Voronoi diagram from Convex Hull
    input: ch = scipy.spatial.ConvexHull object
    
    Structure based on scipy.spatial.Voronoi object, but incomplete, currently only 2D.
    
    returns:
    points (ndarray of double, shape (npoints, ndim)) Coordinates of input points.
    vertices (ndarray of double, shape (nvertices, ndim)) Coordinates of the Voronoi vertices.
    ridge_points (ndarray of ints, shape (nridges, 2)) Indices of the points between which each Voronoi ridge lies.
    ridge_vertices (list of list of ints, shape (nridges, *)) Indices of the Voronoi vertices forming each Voronoi ridge.
        (In 2D, exactly 2 vertices per ridge, vertex=-1 if the ridge goes to infinity)
    scipy.spatial.Voronoi fields regions and point_region are currently not implemented
    
    """
    def __init__(self, ch, faceup=[]):
        if len(faceup) == 0:
            self.faceup = self.get_faceup(ch)
        else:
            self.faceup = faceup
            
        self.points = ch.points[:,:-1]
        self.vertices = -0.5 * ch.equations[:,:2]
        w = np.where(ch.equations[:,2] != 0.)
        if len(w) > 0:
            w=w[0]
            self.vertices[w,:] /= ch.equations[w,2][:,None]
        self.vertices = self.vertices[self.faceup,:]
        uniq,self.faceup_ind,self.faceup_inv = np.unique(self.faceup,return_index=True,return_inverse=True)

        self.ridge_points,self.ridge_vertices = self.edges(ch)

    def get_faceup(self,ch):
        return np.where(np.dot(ch.equations[:,0:-1], [0, 0, -1]) > 1e-6)[0]

    def edges(self,ch):
        dimparab = ch.points.shape[-1] # dimension of the paraboloid (space + 1)
        nb = np.zeros(shape=(len(self.faceup), 2*dimparab), dtype=int)
        simps = np.zeros(shape=(len(self.faceup), 2*dimparab), dtype=int)
        nb[:,1::2] = ch.neighbors[self.faceup]
        nb[:,0::2] = self.faceup[:,None]

        simps[:,0:2] = ch.simplices[self.faceup,1:]
        simps[:,2:4] = ch.simplices[self.faceup,::2]
        simps[:,4:6] = ch.simplices[self.faceup,:2]
        # for 2D (3 adjacencies per simplex); needs to be generalized for != 2D
        
        nb=nb.reshape([-1,2])
        simps = simps.reshape([-1,2])
        arg = np.argsort(nb,axis=1)
        nvertwithdups = nb.shape[0]
        nb_ord = 1*nb
        nb_ord[:,0]=nb[np.arange(nvertwithdups),arg[np.arange(nvertwithdups),0]]
        nb_ord[:,1]=nb[np.arange(nvertwithdups),arg[np.arange(nvertwithdups),1]]
        simps_ord = 1*simps
        simps_ord[:,0]=simps[np.arange(nvertwithdups),arg[np.arange(nvertwithdups),0]]
        simps_ord[:,1]=simps[np.arange(nvertwithdups),arg[np.arange(nvertwithdups),1]]

        uniq,uniq_arg = np.unique(nb_ord,return_index=True,axis=0)

        ridge_points = simps_ord[uniq_arg,:]
        ridge_vertices = nb_ord[uniq_arg,:]
        isinfaceup = np.isin(ridge_vertices,self.faceup)

        ridge_vertices = (ridge_vertices + 1) * isinfaceup - 1

        rvshape = ridge_vertices.shape
        uniq,ridge_vertices = np.unique(ridge_vertices.flatten(),return_inverse=True)
        ridge_vertices=ridge_vertices.reshape(rvshape)-1

        return ridge_points, ridge_vertices

def sectional_voronoi(xy,pot):
    """
    Compute sectional-Voronoi or power diagram, given
    xy = generator positions within the space being tessellated
    pot = potential at these points = additive weight in each distance function/"power"
        = z^2/2, where z is distance at each generator to the space being tessellated
    """
    return Voronoi_from_ConvexHull( ConvexHull( lift_paraboloid(xy, pot)))

def sectional_delaunay(xy,pot):
    """
    Compute "sectional-Delaunay" or regular triangulation, given
    xy = generator positions within the space being tessellated
    pot = potential at these points = additive weight in each distance function/"power"
        = z^2/2, where z is distance at each generator to the space being tessellated
    Delaunay needs further testing
    """
    return Delaunay_from_ConvexHull( ConvexHull( lift_paraboloid(xy, pot)))
