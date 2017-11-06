import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.functions as fn
from .signal import SignalBlock



class AtlasSliceView(QtCore.QObject):
    """A collection of user interface elements bound together:
    
    * One AtlasImageItems displaying an orthogonal view of the atlas
    * An ROI object that defines the slice to be extracted from the orthogonal
      view of the atlas
    * A second AtlasImageItem that displays the sliced view
    * A HistogramLUTItem used to control color/contrast in both images
    * An AtlasDisplayCtrl that sets options for how all elements are drawn
    * A LabelTree that is used to selectively color specific brain regions
    """
    
    sig_slice_changed = QtCore.Signal()  # slice plane changed
    sig_image_changed = QtCore.Signal()  # orthogonal image changed
    mouseHovered = QtCore.Signal(object)
    mouseClicked = QtCore.Signal(object)
    
    def __init__(self):
        QtCore.QObject.__init__(self)

        self.scale = None
        self.atlas = None
        self.label = None
        self.interpolate = True
        
        self.img1 = AtlasImageItem()
        self.img2 = AtlasImageItem()
        self.img1.mouseHovered.connect(self.mouseHovered)
        self.img2.mouseHovered.connect(self.mouseHovered)
        
        self.line_roi = RulerROI([.005, 0], [.008, 0], angle=90, pen=(0, 9), movable=False)
        self.line_roi.sigRegionChanged.connect(self.updateSlice)

        self.zslider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.zslider.valueChanged.connect(self.updateImage)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.sliderRotation)
        
        self.lut = pg.HistogramLUTWidget()
        self.lut.setImageItem(self.img1.atlasImg)
        self.lut.sigLookupTableChanged.connect(self.histlutChanged)
        self.lut.sigLevelsChanged.connect(self.histlutChanged)

        self.displayCtrl = AtlasDisplayCtrl()
        self.displayCtrl.params.sigTreeStateChanged.connect(self.displayCtrlChanged)

        self.labelTree = LabelTree()
        self.labelTree.labelsChanged.connect(self.labelsChanged)

    def set_data1(self, atlas_data):
        self.atlas_data = atlas_data
        self.atlas = None
        self.label = None
        self.display_atlas = None
        self.display_label = None
        self.setAtlas(atlas_data.image)
        self.setLabels(atlas_data.label, atlas_data.ontology)

    def setLabels(self, label, ontology):
        self.label = label
        self.ontology = ontology
        self.labelTree.set_ontology(ontology)
        self.updateImage1()
        self.labelsChanged()

    def setAtlas(self, atlas):
        self.atlas = atlas
        self.updateImage1()

    def updateImage1(self):
        if self.atlas is None or self.label is None:
            return
        axis = self.displayCtrl.params['Orientation']
        axes = {
            'right': ('right', 'anterior', 'dorsal'),
            'dorsal': ('dorsal', 'right', 'anterior'),
            'anterior': ('anterior', 'right', 'dorsal')
        }[axis]
        order = [self.atlas._interpretAxis(ax) for ax in axes]

        # transpose, flip, downsample images
        ds = self.displayCtrl.params['Downsample']
        self.display_atlas = self.atlas.view(np.ndarray).transpose(order)
        with pg.BusyCursor():
            for ax in (0, 1, 2):
                self.display_atlas = pg.downsample(self.display_atlas, ds, axis=ax)
        self.display_label = self.label.view(np.ndarray).transpose(order)[::ds, ::ds, ::ds]

        # make sure atlas/label have the same size after downsampling

        scale = self.atlas._info[-1]['vxsize']*ds
        self.scale = (scale, scale)

        self.zslider.setMaximum(self.display_atlas.shape[0])
        self.zslider.setValue(self.display_atlas.shape[0] // 2)
        self.slider.setRange(-45, 45)
        self.slider.setValue(0)
        self.updateImage()
        self.updateSlice()
        self.lut.setLevels(self.display_atlas.min(), self.display_atlas.max())

    def labelsChanged(self):
        lut = self.labelTree.lookupTable()
        self.setLabelLUT(lut)        
        
    def displayCtrlChanged(self, param, changes):
        update = False
        for param, change, value in changes:
            if param.name() == 'Composition':
                self.setOverlay(value)
            elif param.name() == 'Opacity':
                self.setLabelOpacity(value)
            elif param.name() == 'Interpolate':
                self.setInterpolation(value)
            else:
                update = True
        if update:
            self.updateImage1()

    def updateImage(self):
        z = self.zslider.value()
        self.img1.setData(self.display_atlas[z], self.display_label[z], scale=self.scale)
        self.sig_image_changed.emit()

    def updateSlice(self):
        rotation = self.slider.value()

        if self.display_atlas is None:
            return

        if rotation == 0:
            atlas = self.line_roi.getArrayRegion(self.display_atlas, self.img1.atlasImg, axes=(1, 2), order=int(self.interpolate))
            label = self.line_roi.getArrayRegion(self.display_label, self.img1.atlasImg, axes=(1, 2), order=0)
        else:
            atlas = self.line_roi.getArrayRegion(self.display_atlas, self.img1.atlasImg, rotation=rotation, axes=(1, 2, 0), order=int(self.interpolate))
            label = self.line_roi.getArrayRegion(self.display_label, self.img1.atlasImg, rotation=rotation, axes=(1, 2, 0), order=0)

        if atlas.size == 0:
            return
        
        self.img2.setData(atlas, label, scale=self.scale)
        self.sig_slice_changed.emit()
        
        w = self.img2.atlasImg.scene().views()
        if len(w) > 0:
            # repaint immediately to avoid processing more mouse events before next repaint
            w[0].viewport().repaint()
            #w[0].viewport().repaint()
        
    def sliderRotation(self):
        rotation = self.slider.value()
        self.set_rotation_roi(self.img1.atlasImg, rotation)
        self.updateSlice()

    def close(self):
        self.data = None

    def setOverlay(self, o):
        self.img1.setOverlay(o)
        self.img2.setOverlay(o)

    def setLabelOpacity(self, o):
        self.img1.setLabelOpacity(o)
        self.img2.setLabelOpacity(o)

    def setInterpolation(self, interp):
        assert isinstance(interp, bool)
        self.interpolate = interp

    def setLabelLUT(self, lut):
        self.img1.setLUT(lut)
        self.img2.setLUT(lut)

    def histlutChanged(self):
        # note: img1 is updated automatically; only bneed to update img2 to match
        self.img2.atlasImg.setLookupTable(self.lut.getLookupTable(n=256))
        self.img2.atlasImg.setLevels(self.lut.getLevels())

    def set_rotation_roi(self, img, rotation):

        h1, h2, h3, h4, h5 = self.line_roi.getHandles()

        d_angle = pg.Point(h2.pos() - h1.pos())  # This gives the length in ccf coordinate size 
        d = pg.Point(self.line_roi.mapToItem(img, h2.pos()) - self.line_roi.mapToItem(img, h1.pos()))

        origin_roi = self.line_roi.mapToItem(img, h1.pos())

        if rotation == 0:
            offset = 0
        else:
            offset = self.get_offset(rotation)
        
        # This calculates by how much the ROI needs to shift
        if d.angle(pg.Point(1, 0)) == 90.0:
            # when ROI is on a 90 degree angle, can't really calculate using a right-angle triangle, ugh
            hyp, opposite, adjacent = offset * self.scale[0], 0, offset * self.scale[0]
        else:
            hyp = (offset * self.scale[0])
            opposite = (np.sin(np.radians(-(90 - d.angle(pg.Point(1, 0)))))) * hyp
            adjacent = opposite / (np.tan(np.radians(-(90 - d.angle(pg.Point(1, 0))))))
        
        # This is kind of a hack to avoid recursion error. Using update=False doesn't move the handles.
        self.line_roi.sigRegionChanged.disconnect(self.updateSlice)  
        # increase size to denote rotation
        self.line_roi.setSize(pg.Point(d_angle.length(), hyp * 2))
        # Shift position in order to keep the cutting axis in the middle
        self.line_roi.setPos(pg.Point((origin_roi.x() * self.scale[-1]) + adjacent, (origin_roi.y() * self.scale[-1]) + opposite))
        self.line_roi.sigRegionChanged.connect(self.updateSlice)

    def get_offset(self, rotation):
        theta = np.radians(-rotation)

        # Figure out the unit vector with theta angle
        x, z = 0, 1
        dc, ds = np.cos(theta), np.sin(theta)
        xv = dc * x - ds * z
        zv = ds * x + dc * z

        # Figure out the slope of the unit vector
        m = zv / xv

        # y = mx + b
        # Calculate the x-intercept. using half the distance in the z-dimension as b. Since we want the axis of rotation in the middle
        offset = (-self.atlas.shape[0] / 2) / m

        return abs(offset)


class AtlasDisplayCtrl(pg.parametertree.ParameterTree):
    """UI for controlling how the atlas is displayed. 
    """
    def __init__(self, parent=None):
        pg.parametertree.ParameterTree.__init__(self, parent=parent)
        params = [
            {'name': 'Orientation', 'type': 'list', 'values': ['right', 'anterior', 'dorsal']},
            {'name': 'Opacity', 'type': 'float', 'limits': [0, 1], 'value': 0.5, 'step': 0.1},
            {'name': 'Composition', 'type': 'list', 'values': ['Multiply', 'Overlay', 'SourceOver']},
            {'name': 'Downsample', 'type': 'int', 'value': 1, 'limits': [1, None], 'step': 1},
            {'name': 'Interpolate', 'type': 'bool', 'value': True},
        ]
        self.params = pg.parametertree.Parameter(name='params', type='group', children=params)
        self.setParameters(self.params, showTop=False)
        self.setHeaderHidden(True)


class LabelTree(QtGui.QWidget):
    labelsChanged = QtCore.Signal()

    def __init__(self, parent=None):
        self._block_signals = False
        QtGui.QWidget.__init__(self, parent)
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0,0,0,0)

        self.tree = QtGui.QTreeWidget(self)
        self.layout.addWidget(self.tree, 0, 0)
        self.tree.header().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.tree.headerItem().setText(0, "id")
        self.tree.headerItem().setText(1, "name")
        self.tree.headerItem().setText(2, "color")
        self.labelsById = {}
        self.labelsByAcronym = {}
        self.checked = set()
        self.tree.itemChanged.connect(self.itemChange)

        self.layerBtn = QtGui.QPushButton('Color by cortical layer')
        self.layout.addWidget(self.layerBtn, 1, 0)
        self.layerBtn.clicked.connect(self.colorByLayer)

        self.resetBtn = QtGui.QPushButton('Reset colors')
        self.layout.addWidget(self.resetBtn, 2, 0)
        self.resetBtn.clicked.connect(self.resetColors)

    def set_ontology(self, ontology):
        # prevent emission of multiple signals during update
        self._block_signals = True
        try:
            for rec in ontology:
                self.addLabel(*rec)
        finally:
            self._block_signals = False
        
        self.labelsChanged.emit()

    def addLabel(self, id, parent, name, acronym, color):
        item = QtGui.QTreeWidgetItem([acronym, name, ''])
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(0, QtCore.Qt.Unchecked)

        if parent in self.labelsById:
            root = self.labelsById[parent]['item']
        else:
            root = self.tree.invisibleRootItem()

        root.addChild(item)

        btn = pg.ColorButton(color=pg.mkColor(color))
        btn.defaultColor = color
        self.tree.setItemWidget(item, 2, btn)

        self.labelsById[id] = {'item': item, 'btn': btn}
        item.id = id
        self.labelsByAcronym[acronym] = self.labelsById[id]

        btn.sigColorChanged.connect(self.itemColorChanged)

    def itemChange(self, item, col):
        checked = item.checkState(0) == QtCore.Qt.Checked
        with SignalBlock(self.tree.itemChanged, self.itemChange):
            self.checkRecursive(item, checked)
            
        if not self._block_signals:
            self.labelsChanged.emit()

    def checkRecursive(self, item, checked):
        if checked:
            self.checked.add(item.id)
            item.setCheckState(0, QtCore.Qt.Checked)
        else:
            if item.id in self.checked:
                self.checked.remove(item.id)
            item.setCheckState(0, QtCore.Qt.Unchecked)

        for i in range(item.childCount()):
            self.checkRecursive(item.child(i), checked)

    def itemColorChanged(self, *args):
        self.labelsChanged.emit()

    def lookupTable(self):
        lut = np.zeros((2**16, 4), dtype=np.ubyte)
        for id in self.checked:
            if id >= lut.shape[0]:
                continue
            lut[id] = self.labelsById[id]['btn'].color(mode='byte')
        return lut

    def colorByLayer(self, root=None):
        try:
            unblock = False
            if not isinstance(root, pg.QtGui.QTreeWidgetItem):
                self.blockSignals(True)
                unblock = True
                root = self.labelsByAcronym['Isocortex']['item']

            name = str(root.text(1))
            if ', layer' in name.lower():
                layer = name.split(' ')[-1]
                layer = {'1': 0, '2': 1, '2/3': 2, '4': 3, '5': 4, '6a': 5, '6b': 6}[layer]
                btn = self.labelsById[root.id]['btn']
                btn.setColor(pg.intColor(layer, 10))
                #root.setCheckState(0, QtCore.Qt.Checked)

            for i in range(root.childCount()):
                self.colorByLayer(root.child(i))
        finally:
            if unblock:
                self.blockSignals(False)
                self.labelsChanged.emit()

    def resetColors(self):
        try:
            self.blockSignals(True)
            for k,v in self.labelsById.items():
                v['btn'].setColor(pg.mkColor(v['btn'].defaultColor))
                #v['item'].setCheckState(0, QtCore.Qt.Unchecked)
        finally:
            self.blockSignals(False)
            self.labelsChanged.emit()

    def describe(self, id):
        if id not in self.labelsById:
            return "Unknown label: %d" % id
        descr = []
        item = self.labelsById[id]['item']
        name = str(item.text(1))
        while item is not self.labelsByAcronym['root']['item']:
            descr.insert(0, str(item.text(0)))
            item = item.parent()
        return ' > '.join(descr) + "  :  " + name


class AtlasImageItem(QtGui.QGraphicsItemGroup):
    class SignalProxy(QtCore.QObject):
        mouseHovered = QtCore.Signal(object)  # id
        mouseClicked = QtCore.Signal(object)  # id

    def __init__(self):
        self._sigprox = AtlasImageItem.SignalProxy()
        self.mouseHovered = self._sigprox.mouseHovered
        self.mouseClicked = self._sigprox.mouseClicked

        QtGui.QGraphicsItemGroup.__init__(self)
        self.atlasImg = pg.ImageItem(levels=[0,1])
        self.labelImg = pg.ImageItem()
        self.atlasImg.setParentItem(self)
        self.labelImg.setParentItem(self)
        self.labelImg.setZValue(10)
        self.labelImg.setOpacity(0.5)
        self.setOverlay('Multiply')

        self.labelColors = {}
        self.setAcceptHoverEvents(True)

    def setData(self, atlas, label, scale=None):
        self.labelData = label
        self.atlasData = atlas
        if scale is not None:
            self.resetTransform()
            self.scale(*scale)
        self.atlasImg.setImage(self.atlasData, autoLevels=False)
        self.labelImg.setImage(self.labelData, autoLevels=False)  

    def setLUT(self, lut):
        self.labelImg.setLookupTable(lut)

    def setOverlay(self, overlay):
        mode = getattr(QtGui.QPainter, 'CompositionMode_' + overlay)
        self.labelImg.setCompositionMode(mode)

    def setLabelOpacity(self, o):
        self.labelImg.setOpacity(o)

    def setLabelColors(self, colors):
        self.labelColors = colors

    def hoverEvent(self, event):
        if event.isExit():
            return

        try:
            id = self.labelData[int(event.pos().x()), int(event.pos().y())]
        except IndexError, AttributeError:
            return
        self.mouseHovered.emit(id)

    def mouseClickEvent(self, event):
        id = self.labelData[int(event.pos().x()), int(event.pos().y())]
        self.mouseClicked.emit([event, id])

    def boundingRect(self):
        return self.labelImg.boundingRect()

    def shape(self):
        return self.labelImg.shape()


class RulerROI(pg.ROI):
    """
    ROI subclass with one rotate handle, one scale-rotate handle and one translate handle. Rotate handles handles define a line. 
    
    ============== =============================================================
    **Arguments**
    positions      (list of two length-2 sequences) 
    \**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    """
    
    def __init__(self, pos, size, **args):
        pg.ROI.__init__(self, pos, size, **args)
        self.ab_vector = (0, 0, 0)  # This is the vector pointing up/down from the origin
        self.ac_vector = (0, 0, 0)  # This is the vector pointing across form the orign
        self.origin = (0, 0, 0)     # This is the origin
        self.ab_angle = 90  # angle on the ab_vector
        self.ac_angle = 0   # angle of the ac_vector 
        self.addRotateHandle([0, 0.5], [1, 1])
        self.addScaleRotateHandle([1, 0.5], [0.5, 0.5])
        self.addTranslateHandle([0.5, 0.5])
        self.addFreeHandle([0, 1], [0, 0])  
        self.addFreeHandle([0, 0], [0, 0])
        self.newRoi = pg.ROI((0, 0), [1, 5], parent=self, pen=pg.mkPen('w', style=QtCore.Qt.DotLine))

    def paint(self, p, *args):
        pg.ROI.paint(self, p, *args)
        h1 = self.handles[0]['item'].pos()
        h2 = self.handles[1]['item'].pos()
        h4 = self.handles[3]['item'] 
        h5 = self.handles[4]['item'] 
        h4.setVisible(False)
        h5.setVisible(False)
        p1 = p.transform().map(h1)
        p2 = p.transform().map(h2)

        vec = pg.Point(h2) - pg.Point(h1)
        length = vec.length()

        pvec = p2 - p1
        pvecT = pg.Point(pvec.y(), -pvec.x())
        pos = 0.5 * (p1 + p2) + pvecT * 40 / pvecT.length()

        angle = pg.Point(1, 0).angle(pg.Point(pvec)) 
        self.ab_angle = angle
        
        # Overlay a line to signal which side of the ROI is the back.
        if self.ac_angle > 0:
            self.newRoi.setVisible(True)
            self.newRoi.setPos(h5.pos())
        elif self.ac_angle < 0:
            self.newRoi.setVisible(True)
            self.newRoi.setPos(h4.pos())
        else:
            self.newRoi.setVisible(False)
            
        self.newRoi.setSize(pg.Point(self.size()[0], 0))
        
        p.resetTransform()

        txt = pg.siFormat(length, suffix='m') + '\n%0.1f deg' % angle + '\n%0.1f deg' % self.ac_angle
        p.drawText(QtCore.QRectF(pos.x() - 50, pos.y() - 50, 100, 100), QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter, txt)

    def boundingRect(self):
        r = pg.ROI.boundingRect(self)
        pxw = 50 * self.pixelLength(pg.Point([1, 0]))
        return r.adjusted(-50, -50, 50, 50)

    def getArrayRegion(self, data, img, axes=(0, 1), order=1, rotation=0, **kwds):

        imgPts = [self.mapToItem(img, h.pos()) for h in self.getHandles()]

        d = pg.Point(imgPts[1] - imgPts[0]) # This is the xy direction vector
        o = pg.Point(imgPts[0])
       
        if rotation != 0:
            ac_vector, ac_vector_length, origin = self.get_affine_slice_params(data, img, rotation)
            rgn = fn.affineSlice(data, shape=(int(ac_vector_length), int(d.length())), vectors=[ac_vector, (d.norm().x(), d.norm().y(), 0)],
                                 origin=origin, axes=axes, order=order, **kwds) 
            
            # Save vector and origin
            self.origin = origin
            self.ac_vector = ac_vector * ac_vector_length
        else:
            rgn = fn.affineSlice(data, shape=(int(d.length()),), vectors=[pg.Point(d.norm())], origin=o, axes=axes, order=order, **kwds)
            # Save vector and origin
            self.ac_vector = (0, 0, data.shape[0])
            self.origin = (o.x(), o.y(), 0.0) 
        
        # save this as well
        self.ab_vector = (d.x(), d.y(), 0)
        self.ac_angle = rotation
        
        return rgn

    def get_affine_slice_params(self, data, img, rotation):
        """
        Use the position of this ROI handles to get a new vector for the slice view's x-z direction.
        """
        counter_clockwise = rotation < 0
        
        h1, h2, h3, h4, h5 = self.getHandles()
        origin_roi = self.mapToItem(img, h5.pos())
        left_corner = self.mapToItem(img, h4.pos())
        
        if counter_clockwise:
            origin = np.array([origin_roi.x(), origin_roi.y(), 0])
            end_point = np.array([left_corner.x(), left_corner.y(), data.shape[0]])
        else:
            origin = np.array([left_corner.x(), left_corner.y(), 0])
            end_point = np.array([origin_roi.x(), origin_roi.y(), data.shape[0]])

        new_vector = end_point - origin
        ac_vector_length = np.sqrt(new_vector.dot(new_vector))
        
        return (new_vector[0], new_vector[1], new_vector[2]) / ac_vector_length, ac_vector_length, (origin[0], origin[1], origin[2])
    
    def get_roi_size(self, ab_vector, ac_vector):
        """
        Returns the size of the ROI expected from the given vectors.
        """
        # Find the width
        w = pg.Point(ab_vector[0], ab_vector[1]) 
    
        # Find the length
        l = pg.Point(ac_vector[0], ac_vector[1])
        
        return w.length(), l.length()
    
    def get_ab_angle(self, with_ab_vector=None):
        """
        Gets ROI.ab_angle. If with_ab_vector is given, then the angle returned is with respect to the given vector 
        """
        if with_ab_vector is not None:
            corner = pg.Point(with_ab_vector[0], with_ab_vector[1])
            return corner.angle(pg.Point(1, 0))
        else:
            return self.ab_angle
        
    def get_ac_angle(self, with_ac_vector=None):
        """
        Gets ROI.ac_angle. If with_ac_vector is given, then the angle returned is with respect to the given vector 
        """
        if with_ac_vector is not None:
            l = pg.Point(with_ac_vector[0], with_ac_vector[1])  # Explain this. 
            corner = pg.Point(l.length(), with_ac_vector[2])
            
            if with_ac_vector[0] < 0:  # Make sure this points to the correct direction 
                corner = pg.Point(-l.length(), with_ac_vector[2])
            
            return pg.Point(0, 1).angle(corner)
        else:
            return self.ac_angle

