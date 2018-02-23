import sys, os, traceback
sys.path.append(os.path.join(os.path.dirname(__file__)))
import pyqtgraph as pg

from aiccf.data import CCFAtlasData
from aiccf.viewer import AtlasViewer


if __name__ == '__main__':

    app = pg.mkQApp()

    v = AtlasViewer()
    v.setWindowTitle('CCF Viewer')
    v.show()

    resolution = int(sys.argv[1]) if len(sys.argv) == 2 else None
    atlas_data = CCFAtlasData(resolution=resolution)
    
    v.set_data(atlas_data)

    if sys.flags.interactive == 0:
        app.exec_()
