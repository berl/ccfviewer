import os, sys
import pyqtgraph as pg
from pyqtgraph import metaarray


class CCFAtlasData(object):
    def __init__(self, image_cache_file=None, label_cache_file=None):
        self.image = None
        self.label = None
        self.ontology = None
        
        if sys.platform == 'win32':
            cache_path = os.path.join(os.getenv("APPDATA"), 'aiccf')
        else:
            cache_path = os.path.join(os.path.expanduser("~"), ".local", "share", 'aiccf')
            
        if image_cache_file is None:
            image_cache_file = os.path.join(cache_path, 'ccf_image.ma')
        if label_cache_file is None:
            label_cache_file = os.path.join(cache_path, 'ccf_label.ma')
        self._image_cache_file = image_cache_file
        self._label_cache_file = label_cache_file
        
        if os.path.exists(image_cache_file):
            self.load_image_cache()
            
        if os.path.exists(label_cache_file):
            self.load_label_cache()

    def load_image_data(self, filename):
        self.image = read_nrrd_atlas(filename)
        writeFile(self.image, self._image_cache_file)
        
    def load_label_data(self, label_file, ontology_file):
        self.label = loadNRRDLabels(filename)
        writeFile(self.image, self._image_cache_file)
        
    def load_image_cache(self):
        """Load a MetaArray-format atlas image file.
        """
        filename = self._image_cache_file
        self.image = metaarray.MetaArray(file=filename, readAllData=True)
        
    def load_label_cache(self):
        """Load a MetaArray-format atlas label file.
        """
        filename = self._label_cache_file
        self.label = metaarray.MetaArray(file=filename, readAllData=True)
        self.ontology = self.label._info[-1]['ontology']
        
    def ccf_transform(self):
        """Return a 3D transform that maps from atlas voxel coordinates to CCF
        coordinates (in unscaled meters).
        """
        
    def stereotaxic_transform(self):
        """Return a 3D transform that maps from atlas voxel coodrinates to 
        stereotaxic coordinates.
        """

    
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


def read_nrrd_atlas(nrrd_file=None):
    """
    Download atlas files from:
      http://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas
    """
    import nrrd

    data, header = nrrd.read(nrrd_file)

    # convert to ubyte to compress a bit
    np.multiply(data, 255./data.max(), out=data, casting='unsafe')
    data = data.astype('ubyte')

    # data must have axes (anterior, dorsal, right)
    # rearrange axes to fit -- CCF data comes in (posterior, inferior, right) order.
    data = data[::-1, ::-1, :]

    # voxel size in um
    vxsize = 1e-6 * float(header['space directions'][0][0])

    info = [
        {'name': 'anterior', 'values': np.arange(data.shape[0]) * vxsize, 'units': 'm'},
        {'name': 'dorsal', 'values': np.arange(data.shape[1]) * vxsize, 'units': 'm'},
        {'name': 'right', 'values': np.arange(data.shape[2]) * vxsize, 'units': 'm'},
        {'vxsize': vxsize}
    ]
    ma = metaarray.MetaArray(data, info=info)
    return ma


def read_nrrd_labels(nrrdFile=None, ontologyFile=None):
    """
    Download label files from:
      http://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas

    Download ontology files from:
      http://api.brain-map.org/api/v2/structure_graph_download/1.json

      see:
      http://help.brain-map.org/display/api/Downloading+an+Ontology%27s+Structure+Graph
      http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies#AtlasDrawingsandOntologies-StructuresAndOntologies

    This method compresses the annotation data down to a 16-bit array by remapping
    the larger annotations to smaller, unused values.
    """
    global onto, ontology, data, mapping, inds, vxsize, info, ma

    import nrrd

    with pg.ProgressDialog("Loading annotation file...", 0, 5, wait=0) as dlg:
        print "Loading annotation file..."
        app.processEvents()
        # Read ontology and convert to flat table
        onto = json.load(open(ontoFile, 'rb'))
        onto = parse_ontology(onto['msg'][0])
        l1 = max([len(row[2]) for row in onto])
        l2 = max([len(row[3]) for row in onto])
        ontology = np.array(onto, dtype=[('id', 'int32'), ('parent', 'int32'), ('name', 'S%d'%l1), ('acronym', 'S%d'%l2), ('color', 'S6')])    

        if dlg.wasCanceled():
            return
        dlg += 1

        # read annotation data
        data, header = nrrd.read(nrrdFile)

        if dlg.wasCanceled():
            return
        dlg += 1

        # data must have axes (anterior, dorsal, right)
        # rearrange axes to fit -- CCF data comes in (posterior, inferior, right) order.
        data = data[::-1, ::-1, :]

        if dlg.wasCanceled():
            return
        dlg += 1

    # compress down to uint16
    print "Compressing.."
    u = np.unique(data)
    
    # decide on a 32-to-64-bit label mapping
    mask = u <= 2**16-1
    next_id = 2**16-1
    mapping = OrderedDict()
    inds = set()
    for i in u[mask]:
        mapping[i] = i
        inds.add(i)
   
    with pg.ProgressDialog("Remapping annotations to 16-bit (please be patient with me; this can take several minutes) ...", 0, (~mask).sum(), wait=0) as dlg:
        app.processEvents()
        for i in u[~mask]:
            while next_id in inds:
                next_id -= 1
            mapping[i] = next_id
            inds.add(next_id)
            data[data == i] = next_id
            ontology['id'][ontology['id'] == i] = next_id
            ontology['parent'][ontology['parent'] == i] = next_id
            if dlg.wasCanceled():
                return
            dlg += 1
        
    data = data.astype('uint16')
    mapping = np.array(list(mapping.items()))    
 
    # voxel size in um
    vxsize = 1e-6 * float(header['space directions'][0][0])

    info = [
        {'name': 'anterior', 'values': np.arange(data.shape[0]) * vxsize, 'units': 'm'},
        {'name': 'dorsal', 'values': np.arange(data.shape[1]) * vxsize, 'units': 'm'},
        {'name': 'right', 'values': np.arange(data.shape[2]) * vxsize, 'units': 'm'},
        {'vxsize': vxsize, 'ai_ontology_map': mapping, 'ontology': ontology}
    ]
    ma = metaarray.MetaArray(data, info=info)
    return ma


def parse_ontology(root, parent=-1):
    ont = [(root['id'], parent, root['name'], root['acronym'], root['color_hex_triplet'])]
    for child in root['children']:
        ont += parse_ontology(child, root['id'])
    return ont


def writeFile(data, filename):
    dataDir = os.path.dirname(filename)
    if dataDir != '' and not os.path.exists(dataDir):
        os.makedirs(dataDir)

    if max(data.shape) > 200 and min(data.shape) > 200:
        data.write(filename + '.tmp', chunks=(200, 200, 200))
    else:
        data.write(filename + '.tmp')
        
    os.rename(file+'.tmp', filename)
