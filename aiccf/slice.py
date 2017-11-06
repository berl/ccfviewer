

class CCFAtlasSlice(object):
    """Represents a 2D or 3D slice through volumetric atlas data.
    
    The slice specifies the position and orientation of a plane or rectangular
    slice, and manages extracting data from the volume as well as generating
    the relevant coordinate transforms. 
    """
    def __init__(self, atlas_data):
        self.atlas_data = atlas_data
        self.shape = shape
        self.origin = origin
        self.vectors = vectors
    
    def set_atlas_data(self, atlas):
        self.atlas_data = atlas
    
    def set_slice(self, shape, origin, vectors):
        self.shape = shape
        self.origin = origin
        self.vectors = vectors
    
    def atlas_transform(self):
        """Return a transform that maps from the 2D/3D coordinates of the slice
        to the 3D voxel coordinates of the atlas.
        """

    def ccf_transform(self):
        return self.atlas_transform() * self.atlas_data.ccf_transform()

    def stereotaxic_transform(self):
        return self.atlas_transform() * self.atlas_data.stereotaxic_transform()


