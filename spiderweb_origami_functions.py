import numpy as np
import sectional_tess
import pylab as plt
from matplotlib import collections
from matplotlib.patches import Polygon

def plot_mink_fold(vor,delaun, alpha,rotangle):
    """
    Generates an origami-tessellation crease pattern from a spiderweb geometry.
    
    'vor' and 'delaun' are voronoi or delaunay tessellations, generated from
    the native scipy.spatial.Delaunay or .Voronoi tessellations, or from
    our 'sectional-tess' functions, sectional_voronoi and sectional_delaunay.
    
    'alpha' is the scaling factor between the voronoi and delaunay 
    tessellations. alpha=1 gives the same size to both; large alpha gives 
    large twist fold (Delaunay) triangles, while small alpha gives large
    Voronoi polygons inbetween

    'rotangle' is the rotation angle, in degrees. rotangle=0 keeps vor and
    delaun tessellations perpendicular. A typical origami twist angle 
    (~ 20 degr) will have rotangle 90 +/- 20.
    """

    fullvorv = vor.vertices
    nvertstot = fullvorv.shape[0]
    nsimps = delaun.simplices.shape[0]
    # these should be equal
    
    unrot_edge_points = np.empty((nsimps,4,2),dtype=np.float)
    unrot_edge_points_aneg = np.empty((nsimps,4,2),dtype=np.float)
    unrot_edge_points[:,:3,:] = alpha*delaun.points[delaun.simplices.flatten(),:2].reshape(nsimps,3,2)
    unrot_edge_points_aneg[:,:3,:] = -alpha*delaun.points[delaun.simplices.flatten(),:2].reshape(nsimps,3,2)
    unrot_edge_points[:,3,:] = unrot_edge_points[:,0,:]
    unrot_edge_points_aneg[:,3,:] = unrot_edge_points_aneg[:,0,:]
        
    # rotate the points
    if rotangle > 0.:
        theta = np.radians(rotangle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s), (s, c)))
        Rneg = np.array(((c,s), (-s, c)))
        rot_edge_points = np.einsum('ij,klj->kli',R,unrot_edge_points)
        rot_edge_points_aneg = np.einsum('ij,klj->kli',Rneg,unrot_edge_points_aneg)
    else:
        rot_edge_points = 1.*unrot_edge_points
        rot_edge_points_aneg = 1.*unrot_edge_points
    
    for i in range(4):
        unrot_edge_points[:,i,:] += fullvorv
        unrot_edge_points_aneg[:,i,:] += fullvorv
        rot_edge_points[:,i,:] += fullvorv
        rot_edge_points_aneg[:,i,:] += fullvorv
        
    maxxie = np.max(np.abs(rot_edge_points))
    
    dists = np.empty((3,3,2),dtype=np.float)
    thresh = float(2**16+1)/float(2**16)

    filament_nodes = np.empty((6*nvertstot,2),np.int)
    filament_verts = np.empty((6*nvertstot,2),np.int)
    filcount = 0
    filpatches = []
    nodepatches = []
    
    for node in range(nvertstot):
        nodepatches.append(Polygon(rot_edge_points_aneg[node,:3,:]))
        
        #plt.text(rot_edge_points[node,0,0],rot_edge_points[node,0,1],'%d'%(node))
        for adj in range(3):
            #for each adjacent Lagrangian triangle, only draw the shortest 2 out of 9 links
            adjnode = delaun.neighbors[node,adj]
            if adjnode >= 0: #real neighbor
                for vert1 in range(3):
                    for vert2 in range(3):
                        dists[vert1,vert2,:] = unrot_edge_points[node,vert1,:]-\
                                                unrot_edge_points[adjnode,vert2,:]
                                                   
                sumdist2 = np.sum(dists**2,-1)
                mindist2 = np.min(sumdist2)
                wheremin = np.where(sumdist2/mindist2 < thresh)

                #filaments[filcount,:,:]=[node_edge_points[node,wheremin[0][0],:],
                #                         node_edge_points[node,wheremin[0][0],:]+\
                #                         alphasign*dists[wheremin[0][0],wheremin[1][0],:]]
                filament_nodes[filcount,:]=[node,adjnode]
                filament_verts[filcount,:] = [wheremin[0][0],wheremin[1][0]]
                filcount += 1
                filament_nodes[filcount,:]=[node,adjnode]
                filament_verts[filcount,:] = [wheremin[0][1],wheremin[1][1]]
                filcount += 1
                
                filpatches.append(Polygon([rot_edge_points_aneg[node,wheremin[0][0],:],
                                           rot_edge_points_aneg[adjnode,wheremin[1][0],:],
                                           rot_edge_points_aneg[adjnode,wheremin[1][1],:],
                                            rot_edge_points_aneg[node,wheremin[0][1],:]]))
                                                   


    filaments = np.empty((filcount,2,2))
    filaments = rot_edge_points[filament_nodes[:filcount,:],filament_verts[:filcount,:],:]

    filaments_aneg = rot_edge_points_aneg[filament_nodes[:filcount,:],filament_verts[:filcount,:],:]

    fig = plt.figure(figsize=[12,4])

    fig.subplots_adjust(left=0.,right=1.,top=1.,bottom=0.,hspace=0.,wspace=0.)

    ax = fig.add_subplot(131)
    ax.add_collection(collections.LineCollection(rot_edge_points[:,:,:], linewidths=0.5, color='black'))
    ax.add_collection(collections.LineCollection(filaments,linewidths=0.5, color='C0',alpha=1.))    
    ax.axis('equal')
    ax.set_xticks([]); ax.set_yticks([])
    
    ax2 = fig.add_subplot(132)
    ax2.add_collection(collections.LineCollection(rot_edge_points_aneg[:,:,:], linewidths=0.5, color='black',alpha=0.5))
    ax2.add_collection(collections.LineCollection(filaments_aneg,linewidths=0.5, color='C0',alpha=0.25))
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.axis('equal')
    #plt.savefig('foldedform_creases.pdf',transparent=True)

    ax3 = fig.add_subplot(133)
    ax3.add_collection(collections.PatchCollection(filpatches,alpha=1./12.,color='C0'))
    ax3.add_collection(collections.PatchCollection(nodepatches,alpha=1./12.,color='black'))
    ax3.axis('equal')
    ax3.set_yticks([])
    ax3.set_xticks([])
    
    plt.show()
    return


