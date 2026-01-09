import sys
import os
import time
import math
import json
import numpy as np
import cv2
import torch
from PIL import Image

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QListWidget, QSlider, QMessageBox, QProgressDialog,
                             QLineEdit, QColorDialog, QGroupBox, QListWidgetItem,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsItem,
                             QGraphicsEllipseItem, QSplitter, QShortcut, QSizePolicy,
                             QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QLineF, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QIcon, QPolygonF, QKeySequence

from transformers import Sam3Model, Sam3Processor

# ------------------------------------------------------
# Config
# ------------------------------------------------------
MODEL_PATH = "sam3_weights"
CONFIG_FILE = "config.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------
# Math Helper
# ------------------------------------------------------
def distance_point_to_segment(p, v, w):
    """
    Calculates the minimum distance from a point to a line segment and finds the projection point.

    This function computes the perpendicular distance if the projection falls within
    the segment, or the distance to the nearest endpoint if it falls outside.



[Image of vector projection on line segment]


    Args:
        p (QPointF): The query point.
        v (QPointF): The first endpoint of the line segment.
        w (QPointF): The second endpoint of the line segment.

    Returns:
        tuple: A tuple containing:
            - distance (float): The squared distance (or Euclidean distance approximation depending on implementation context).
            - projection (QPointF): The coordinates of the closest point on the segment.
    """
    l2 = (w.x() - v.x()) ** 2 + (w.y() - v.y()) ** 2
    if l2 == 0:
        return (p.x() - v.x()) ** 2 + (p.y() - v.y()) ** 2, v

    t = ((p.x() - v.x()) * (w.x() - v.x()) + (p.y() - v.y()) * (w.y() - v.y())) / l2
    t = max(0, min(1, t))

    proj_x = v.x() + t * (w.x() - v.x())
    proj_y = v.y() + t * (w.y() - v.y())

    return math.sqrt((p.x() - proj_x) ** 2 + (p.y() - proj_y) ** 2), QPointF(proj_x, proj_y)


# ------------------------------------------------------
# Graphics Items
# ------------------------------------------------------
class VertexHandle(QGraphicsRectItem):
    """
    A draggable square handle representing a vertex of a polygon.

    This item allows users to move vertices to reshape the polygon or right-click
    to delete specific vertices. Size automatically scales with zoom level.
    """

    def __init__(self, parent_polygon, index, point, base_size=10):
        """
        Initializes the VertexHandle.

        Args:
            parent_polygon (EditablePolygonItem): The polygon instance this handle modifies.
            index (int): The index of this vertex in the polygon's coordinate list.
            point (QPointF): The initial position of the handle in the scene.
            base_size (int, optional): The base width/height of the square handle. Defaults to 10.
        """
        super().__init__(-base_size / 2, -base_size / 2, base_size, base_size)
        self.setParentItem(parent_polygon)
        self.setPos(point)
        self.setBrush(QBrush(Qt.white))
        self.setPen(QPen(Qt.blue, 1))
        self.setFlags(
            QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemIgnoresTransformations)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.parent_poly = parent_polygon
        self.index = index
        self.base_size = base_size
        self.drag_start_pos = None  # Track drag start position for undo

    def update_size(self, size_multiplier=1.0):
        """
        Updates the handle size based on a multiplier.

        Args:
            size_multiplier (float): Multiplier for the base size (e.g., 0.5 to 2.0)
        """
        new_size = self.base_size * size_multiplier
        self.setRect(-new_size / 2, -new_size / 2, new_size, new_size)

    def itemChange(self, change, value):
        """
        Intercepts item state changes to update the parent polygon geometry.
        Constrains movement to image boundaries.

        Args:
            change (QGraphicsItem.GraphicsItemChange): The type of state change (e.g., position change).
            value (Any): The new value associated with the change.

        Returns:
            Any: The result of the superclass itemChange method.
        """
        if change == QGraphicsItem.ItemPositionChange and self.parent_poly:
            # Constrain position to image boundaries
            scene = self.scene()
            if scene and hasattr(scene, 'views') and scene.views():
                view = scene.views()[0]
                if hasattr(view, 'pixmap_item') and view.pixmap_item:
                    pixmap = view.pixmap_item.pixmap()
                    if not pixmap.isNull():
                        # Get image bounds
                        img_rect = view.pixmap_item.boundingRect()

                        # Clamp position to image boundaries
                        new_pos = value
                        clamped_x = max(img_rect.left(), min(new_pos.x(), img_rect.right()))
                        clamped_y = max(img_rect.top(), min(new_pos.y(), img_rect.bottom()))
                        value = QPointF(clamped_x, clamped_y)

            self.parent_poly.update_vertex_pos(self.index, value)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        """
        Handles mouse press events on the handle.

        Right-clicking triggers vertex deletion.
        Left-clicking saves start position for undo.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse event details.
        """
        if event.button() == Qt.RightButton:
            self.parent_poly.delete_vertex(self.index)
            event.accept()
        elif event.button() == Qt.LeftButton:
            # Save start position for undo when drag starts
            self.drag_start_pos = QPointF(self.pos())
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Handles mouse release events to record undo action after vertex drag.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse event details.
        """
        if event.button() == Qt.LeftButton and self.drag_start_pos is not None:
            drag_end_pos = QPointF(self.pos())

            # Only record if position actually changed
            if (drag_end_pos - self.drag_start_pos).manhattanLength() > 0.1:
                self.parent_poly.record_vertex_move(self.index, self.drag_start_pos, drag_end_pos)

            self.drag_start_pos = None

        super().mouseReleaseEvent(event)


class BboxCornerHandle(QGraphicsEllipseItem):
    """
    A draggable corner handle for resizing bounding boxes.
    Size automatically scales with zoom level.
    """
    def __init__(self, corner_index, parent_bbox, base_size=8):
        """
        Args:
            corner_index (int): 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
            parent_bbox (EditableBboxItem): The bbox item this handle belongs to
            base_size (int, optional): The base diameter of the handle. Defaults to 8.
        """
        super().__init__(-base_size / 2, -base_size / 2, base_size, base_size)
        self.corner_index = corner_index
        self.parent_bbox = parent_bbox
        self.base_size = base_size
        self.drag_start_bbox = None  # Track bbox at drag start for undo

        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setPen(QPen(QColor(0, 0, 0), 1))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(200)

    def update_size(self, size_multiplier=1.0):
        """
        Updates the handle size based on a multiplier.

        Args:
            size_multiplier (float): Multiplier for the base size (e.g., 0.5 to 2.0)
        """
        new_size = self.base_size * size_multiplier
        self.setRect(-new_size / 2, -new_size / 2, new_size, new_size)

    def itemChange(self, change, value):
        """Update bbox when handle is moved. Constrains to image boundaries."""
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            # Constrain position to image boundaries
            scene = self.scene()
            if scene and hasattr(scene, 'views') and scene.views():
                view = scene.views()[0]
                if hasattr(view, 'pixmap_item') and view.pixmap_item:
                    pixmap = view.pixmap_item.pixmap()
                    if not pixmap.isNull():
                        # Get image bounds
                        img_rect = view.pixmap_item.boundingRect()

                        # Clamp position to image boundaries
                        new_pos = value
                        clamped_x = max(img_rect.left(), min(new_pos.x(), img_rect.right()))
                        clamped_y = max(img_rect.top(), min(new_pos.y(), img_rect.bottom()))
                        value = QPointF(clamped_x, clamped_y)

            self.parent_bbox.update_from_handle(self.corner_index, value)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        """Save bbox state when starting drag."""
        if event.button() == Qt.LeftButton:
            # Save current bbox for undo
            self.drag_start_bbox = self.parent_bbox.bbox[:]  # Copy list
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Record undo action after bbox resize."""
        if event.button() == Qt.LeftButton and self.drag_start_bbox is not None:
            drag_end_bbox = self.parent_bbox.bbox[:]

            # Only record if bbox actually changed
            if self.drag_start_bbox != drag_end_bbox:
                self.parent_bbox.record_bbox_resize(self.drag_start_bbox, drag_end_bbox)

            self.drag_start_bbox = None

        super().mouseReleaseEvent(event)


class EditableBboxItem(QGraphicsRectItem):
    """
    A custom QGraphicsRectItem for editing bounding boxes.

    Supports resizing via corner handles while maintaining rectangular shape.
    """

    def __init__(self, bbox, label, color, score=None, instance_id=None, area=None):
        """
        Initializes the EditableBboxItem.

        Args:
            bbox (list): [x_min, y_min, x_max, y_max]
            label (str): The semantic class name
            color (QColor): The color associated with the class label
            score (float): Confidence score
            instance_id (int): Instance identifier
            area (float): Pixel area
        """
        x_min, y_min, x_max, y_max = bbox
        super().__init__(x_min, y_min, x_max - x_min, y_max - y_min)

        self.label = label
        self.base_color = color
        self.shape_type = "detection_bbox"
        self.bbox = bbox
        self.score = score
        self.instance_id = instance_id
        self.component_id = None  # Bboxes don't have component_id, but needed for compatibility
        self.area = area

        # Callback for notifying changes
        self.on_change_callback = None

        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))

        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
        self.setAcceptHoverEvents(True)

        self.is_editing = False
        self.handles = []  # Corner handles for editing
        self._updating_handles = False  # Flag to prevent recursion during handle updates

        # Set initial pen width (will be updated to be zoom-independent)
        self.update_pen_width()

    def notify_change(self):
        """Triggers the change callback if one is registered."""
        if self.on_change_callback:
            self.on_change_callback()

    def update_properties(self, label, color):
        """
        Updates the label and visual color of the bbox.

        Args:
            label (str): The new class label.
            color (QColor): The new color for the pen and brush.
        """
        self.label = label
        self.base_color = color
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
        self.set_editing(self.is_editing)
        self.notify_change()

    def set_editing(self, editing):
        """
        Toggles the editing mode for the bbox.

        Args:
            editing (bool): True to enable editing, False to disable.
        """
        self.is_editing = editing
        self.update_pen_width()
        if editing:
            self.create_handles()
        else:
            self.remove_handles()

    def update_pen_width(self):
        """Updates pen width to be zoom-independent and user-adjustable."""
        # Get zoom level from view
        zoom_factor = 1.0
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            zoom_factor = view.transform().m11()

        # Get border width multiplier from main window
        width_multiplier = 1.0
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'main_window') and view.main_window:
                width_multiplier = view.main_window.border_width_multiplier

        # Calculate width that stays constant in screen space
        base_width = (3 if self.is_editing else 2) * width_multiplier
        adjusted_width = base_width / zoom_factor

        pen = QPen(self.base_color)
        pen.setWidthF(adjusted_width)
        pen.setStyle(Qt.SolidLine)
        self.setPen(pen)

    def create_handles(self):
        """Creates four corner handles for bbox editing."""
        self.remove_handles()

        rect = self.rect()
        pos = self.pos()

        # Define corner positions: top-left, top-right, bottom-right, bottom-left
        corners = [
            QPointF(pos.x() + rect.left(), pos.y() + rect.top()),      # 0: top-left
            QPointF(pos.x() + rect.right(), pos.y() + rect.top()),     # 1: top-right
            QPointF(pos.x() + rect.right(), pos.y() + rect.bottom()),  # 2: bottom-right
            QPointF(pos.x() + rect.left(), pos.y() + rect.bottom())    # 3: bottom-left
        ]

        # Get vertex size multiplier from main window if available
        size_multiplier = 1.0
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'main_window') and view.main_window:
                size_multiplier = view.main_window.vertex_size_multiplier

        for i, corner_pos in enumerate(corners):
            handle = BboxCornerHandle(i, self)
            handle.update_size(size_multiplier)
            handle.setPos(corner_pos)
            self.scene().addItem(handle)
            self.handles.append(handle)

    def remove_handles(self):
        """Removes all corner handles."""
        for handle in self.handles:
            if handle.scene():
                self.scene().removeItem(handle)
        self.handles = []

    def update_from_handle(self, corner_index, new_pos):
        """
        Updates bbox geometry when a corner handle is dragged.

        Args:
            corner_index (int): Which corner is being dragged (0-3)
            new_pos (QPointF): New position of the handle
        """
        if self._updating_handles:
            return  # Prevent recursion

        self._updating_handles = True

        rect = self.rect()
        pos = self.pos()

        # Get current absolute corners
        x_min = pos.x() + rect.left()
        y_min = pos.y() + rect.top()
        x_max = pos.x() + rect.right()
        y_max = pos.y() + rect.bottom()

        # Update based on which corner is being dragged
        if corner_index == 0:  # top-left
            x_min = new_pos.x()
            y_min = new_pos.y()
        elif corner_index == 1:  # top-right
            x_max = new_pos.x()
            y_min = new_pos.y()
        elif corner_index == 2:  # bottom-right
            x_max = new_pos.x()
            y_max = new_pos.y()
        elif corner_index == 3:  # bottom-left
            x_min = new_pos.x()
            y_max = new_pos.y()

        # Ensure min < max
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        # Update rect and position
        self.setPos(0, 0)
        self.setRect(x_min, y_min, x_max - x_min, y_max - y_min)

        # Update other handles to match new corners
        self.update_handle_positions()

        # Update internal bbox
        self.update_bbox()
        self.notify_change()

        self._updating_handles = False

    def update_handle_positions(self):
        """Updates all handle positions to match current bbox geometry."""
        if not self.handles:
            return

        rect = self.rect()
        pos = self.pos()

        corners = [
            QPointF(pos.x() + rect.left(), pos.y() + rect.top()),
            QPointF(pos.x() + rect.right(), pos.y() + rect.top()),
            QPointF(pos.x() + rect.right(), pos.y() + rect.bottom()),
            QPointF(pos.x() + rect.left(), pos.y() + rect.bottom())
        ]

        # Temporarily block signals to prevent recursion
        for i, handle in enumerate(self.handles):
            handle.setFlag(QGraphicsItem.ItemSendsGeometryChanges, False)
            handle.setPos(corners[i])
            handle.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

    def polygon(self):
        """
        Returns the bbox as a QPolygonF for compatibility with save/load functions.

        Returns:
            QPolygonF: Rectangle as a 4-point polygon
        """
        rect = self.rect()
        return QPolygonF([
            rect.topLeft(),
            rect.topRight(),
            rect.bottomRight(),
            rect.bottomLeft()
        ])

    def itemChange(self, change, value):
        """Intercepts item state changes to update bbox and notify changes."""
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.update_bbox()
            self.update_handle_positions()
            self.notify_change()
        return super().itemChange(change, value)

    def update_bbox(self):
        """Updates the internal bbox representation after position/size changes."""
        rect = self.rect()
        pos = self.pos()
        self.bbox = [
            int(pos.x() + rect.x()),
            int(pos.y() + rect.y()),
            int(pos.x() + rect.x() + rect.width()),
            int(pos.y() + rect.y() + rect.height())
        ]

    def record_bbox_resize(self, old_bbox, new_bbox):
        """
        Records a bbox resize operation for undo.

        Args:
            old_bbox (list): Original bbox [x_min, y_min, x_max, y_max]
            new_bbox (list): New bbox after resize
        """
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'main_window') and view.main_window:
                view.main_window.record_single_action({
                    'type': 'bbox_resize',
                    'item': self,
                    'old_bbox': old_bbox[:],
                    'new_bbox': new_bbox[:]
                })

    def mousePressEvent(self, event):
        """
        Handles mouse press events on the bbox.

        Alt + Click is ignored to allow canvas panning.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse event details.
        """
        # Alt key pressed: ignore the event to allow canvas panning
        if event.modifiers() == Qt.AltModifier:
            event.ignore()
            return

        super().mousePressEvent(event)


class EditablePolygonItem(QGraphicsPolygonItem):
    """
    A custom QGraphicsPolygonItem that supports semantic labeling, editing, and visual styling.

    It manages a list of VertexHandles when in editing mode and provides methods to
    manipulate the underlying polygon geometry.

    Supports three annotation types:
    - detection_bbox: Bounding box for object detection
    - instance_polygon: Instance segmentation polygon
    - semantic_polygon: Semantic segmentation polygon (fused connected instances)
    """

    def __init__(self, polygon, label, color, shape_type="instance_polygon", bbox=None, score=None,
                 instance_id=None, component_id=None, area=None):
        """
        Initializes the EditablePolygonItem.

        Args:
            polygon (QPolygonF): The initial geometry of the polygon.
            label (str): The semantic class name (e.g., "car", "tree").
            color (QColor): The color associated with the class label.
            shape_type (str): Type of annotation - "detection_bbox", "instance_polygon", or "semantic_polygon"
            bbox (list): [x_min, y_min, x_max, y_max] bounding box
            score (float): Confidence score (for instance annotations)
            instance_id (int): Instance identifier (for instance annotations)
            component_id (int): Connected component ID (for semantic annotations)
            area (float): Pixel area of the annotation
        """
        super().__init__(polygon)
        self.label = label
        self.base_color = color
        self.shape_type = shape_type
        self.bbox = bbox
        self.score = score
        self.instance_id = instance_id
        self.component_id = component_id
        self.area = area

        # Callback for notifying changes (set by MainWindow)
        self.on_change_callback = None

        # Different opacity for different types
        opacity = 80 if shape_type == "instance_polygon" else 120 if shape_type == "semantic_polygon" else 50
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), opacity)))

        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)

        self.handles = []
        self.is_editing = False

        # Set initial pen width (will be updated to be zoom-independent)
        self.update_pen_width()

    def notify_change(self):
        """
        Triggers the change callback if one is registered.
        Used to notify the main window that the annotation has been modified (dirty state).
        """
        if self.on_change_callback:
            self.on_change_callback()

    def update_properties(self, label, color):
        """
        Updates the label and visual color of the polygon.

        Args:
            label (str): The new class label.
            color (QColor): The new color for the pen and brush.
        """
        self.label = label
        self.base_color = color
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
        self.set_editing(self.is_editing)
        self.notify_change()  # Label/Color changed

    def set_editing(self, editing):
        """
        Toggles the editing mode for the polygon.

        When enabled, vertex handles are created and the border becomes dashed.
        When disabled, handles are removed and the border becomes solid.

        Args:
            editing (bool): True to enable editing, False to disable.
        """
        self.is_editing = editing
        self.update_pen_width()
        if editing:
            self.create_handles()
        else:
            self.remove_handles()

    def update_pen_width(self):
        """Updates pen width to be zoom-independent and user-adjustable."""
        # Get zoom level from view
        zoom_factor = 1.0
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            zoom_factor = view.transform().m11()

        # Get border width multiplier from main window
        width_multiplier = 1.0
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'main_window') and view.main_window:
                width_multiplier = view.main_window.border_width_multiplier

        # Calculate width that stays constant in screen space
        base_width = (3 if self.is_editing else 2) * width_multiplier
        adjusted_width = base_width / zoom_factor

        pen = QPen(self.base_color)
        pen.setWidthF(adjusted_width)
        pen.setStyle(Qt.DashLine if self.is_editing else Qt.SolidLine)
        self.setPen(pen)

    def create_handles(self):
        """Generates VertexHandle items for every point in the polygon."""
        self.remove_handles()
        poly = self.polygon()
        # Get vertex size multiplier from main window if available
        size_multiplier = 1.0
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'main_window') and view.main_window:
                size_multiplier = view.main_window.vertex_size_multiplier

        for i in range(poly.count()):
            handle = VertexHandle(self, i, poly.at(i))
            handle.update_size(size_multiplier)
            self.handles.append(handle)

    def remove_handles(self):
        """Removes all VertexHandle items from the scene."""
        scene = self.scene()
        for h in self.handles:
            if scene:
                scene.removeItem(h)
            else:
                h.setParentItem(None)
        self.handles = []

    def update_vertex_pos(self, index, new_pos):
        """
        Updates the position of a specific vertex.

        Args:
            index (int): The index of the vertex to move.
            new_pos (QPointF): The new coordinates for the vertex.
        """
        poly = self.polygon()
        if 0 <= index < poly.count():
            poly.replace(index, new_pos)
            self.setPolygon(poly)
            self.notify_change()  # Geometry changed (drag)

    def delete_vertex(self, index):
        """
        Deletes a vertex at the specified index.
        The polygon must retain at least 3 vertices.

        Args:
            index (int): The index of the vertex to remove.
        """
        poly = self.polygon()
        if poly.count() > 3:
            # Record the deleted vertex for undo
            deleted_point = QPointF(poly.at(index))
            self.record_vertex_delete(index, deleted_point)

            poly.remove(index)
            self.setPolygon(poly)
            self.create_handles()
            self.notify_change()  # Geometry changed (delete point)

    def insert_vertex(self, index, pos):
        """
        Inserts a new vertex into the polygon.

        Args:
            index (int): The index at which to insert the new point.
            pos (QPointF): The coordinates of the new vertex.
        """
        # Record the insertion for undo
        self.record_vertex_insert(index, QPointF(pos))

        poly = self.polygon()
        poly.insert(index, pos)
        self.setPolygon(poly)
        self.create_handles()
        self.notify_change()  # Geometry changed (add point)

    def record_vertex_move(self, index, old_pos, new_pos):
        """
        Records a vertex move operation for undo.

        Args:
            index (int): Vertex index
            old_pos (QPointF): Original position
            new_pos (QPointF): New position after drag
        """
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'main_window') and view.main_window:
                view.main_window.record_single_action({
                    'type': 'vertex_move',
                    'item': self,
                    'index': index,
                    'old_pos': QPointF(old_pos),
                    'new_pos': QPointF(new_pos)
                })

    def record_vertex_delete(self, index, deleted_point):
        """Records vertex deletion for undo."""
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'main_window') and view.main_window:
                view.main_window.record_single_action({
                    'type': 'vertex_delete',
                    'item': self,
                    'index': index,
                    'deleted_point': QPointF(deleted_point)
                })

    def record_vertex_insert(self, index, inserted_point):
        """Records vertex insertion for undo."""
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'main_window') and view.main_window:
                view.main_window.record_single_action({
                    'type': 'vertex_insert',
                    'item': self,
                    'index': index,
                    'inserted_point': QPointF(inserted_point)
                })

    def mousePressEvent(self, event):
        """
        Handles mouse press events on the polygon.

        Supports adding new vertices by holding Ctrl + Left Click near an edge.
        Alt + Click is ignored to allow canvas panning.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse event details.
        """
        # Alt key pressed: ignore the event to allow canvas panning
        if event.modifiers() == Qt.AltModifier:
            event.ignore()
            return

        if self.is_editing and event.modifiers() == Qt.ControlModifier and event.button() == Qt.LeftButton:
            click_pos = event.pos()
            poly = self.polygon()
            best_dist = float('inf')  # 初始化为无穷大，确保能找到最近的线段
            best_idx = -1
            insert_pos = QPointF()

            # 遍历所有线段，找到距离点击位置最近的那条线段
            for i in range(poly.count()):
                p1 = poly.at(i)
                p2 = poly.at((i + 1) % poly.count())
                dist, proj = distance_point_to_segment(click_pos, p1, p2)

                # 更新最近的线段（移除了距离阈值限制）
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i + 1
                    insert_pos = proj

            # 总是能找到最近的线段，插入投影点
            if best_idx != -1:
                self.insert_vertex(best_idx, insert_pos)
                event.accept()
                return

        super().mousePressEvent(event)


# ------------------------------------------------------
# Canvas View
# ------------------------------------------------------
class CanvasView(QGraphicsView):
    """
    The main viewing widget for displaying images and annotations.

    It handles two primary modes:
    1. 'view': Standard interaction, selecting polygons, zooming, panning.
    2. 'create': Drawing new polygons point-by-point.
    """
    item_clicked = pyqtSignal(object)
    item_double_clicked = pyqtSignal(object)
    scene_clicked = pyqtSignal()

    def __init__(self, parent=None):
        """
        Initializes the CanvasView settings.

        Args:
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.pixmap_item = None
        self.mode = "view"
        self.temp_points = []
        self.temp_visuals = []
        self.main_window = None

        # For bbox creation (click and drag)
        self.bbox_start_pos = None
        self.temp_bbox_rect = None

    def set_image(self, pixmap):
        """
        Loads and displays a new image in the scene.
        Adds padding around the image for better panning freedom.

        Args:
            pixmap (QPixmap): The image data to display.
        """
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setZValue(-100)
        self.scene.addItem(self.pixmap_item)

        # Add padding around image for panning freedom
        # This allows edges and corners to be moved to viewport center
        img_rect = QRectF(pixmap.rect())
        padding = max(img_rect.width(), img_rect.height()) * 0.5  # 50% of larger dimension
        padded_rect = img_rect.adjusted(-padding, -padding, padding, padding)
        self.setSceneRect(padded_rect)

        self.reset_creation()

    def clamp_to_image(self, pos):
        """
        Clamps a scene position to image boundaries.

        Args:
            pos (QPointF): The position to clamp.

        Returns:
            QPointF: The clamped position within image bounds.
        """
        if self.pixmap_item and not self.pixmap_item.pixmap().isNull():
            img_rect = self.pixmap_item.boundingRect()
            clamped_x = max(img_rect.left(), min(pos.x(), img_rect.right()))
            clamped_y = max(img_rect.top(), min(pos.y(), img_rect.bottom()))
            return QPointF(clamped_x, clamped_y)
        return pos

    def reset_creation(self):
        """Clears all temporary visual elements (points/lines/bbox) used during creation."""
        self.temp_points = []
        for item in self.temp_visuals:
            self.scene.removeItem(item)
        self.temp_visuals = []

        # Clear bbox creation state
        if self.temp_bbox_rect and self.temp_bbox_rect.scene():
            self.scene.removeItem(self.temp_bbox_rect)
        self.temp_bbox_rect = None
        self.bbox_start_pos = None

    def resizeEvent(self, event):
        """
        Handles window resize events to fit the image in view.

        Args:
            event (QResizeEvent): The resize event details.
        """
        super().resizeEvent(event)
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        """
        Handles mouse wheel events to zoom in and out of the scene.
        Updates pen widths after zoom to maintain consistent visual appearance.

        Args:
            event (QWheelEvent): The wheel event details.
        """
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(zoom_factor, zoom_factor)

        # Update pen widths for all items to stay zoom-independent
        if self.main_window:
            for item in self.main_window.polygon_items:
                if isinstance(item, (EditablePolygonItem, EditableBboxItem)):
                    if hasattr(item, 'update_pen_width'):
                        item.update_pen_width()

    def mousePressEvent(self, event):
        """
        Handles mouse clicks based on the current mode.

        - 'create' mode:
            - Detection mode: Click and drag to create bbox
            - Instance/Semantic mode: Left click adds points, Right click undoes
        - 'view' mode: Left click selects items or background.
            - Alt+drag: Pan the canvas even when clicking on masks

        Args:
            event (QMouseEvent): The mouse event details.
        """
        if self.mode == "create":
            pos_scene = self.mapToScene(event.pos())
            # Clamp position to image boundaries
            pos_scene = self.clamp_to_image(pos_scene)

            # Check if we're in detection mode (bbox creation)
            if self.main_window and self.main_window.annotation_mode == "detection":
                if event.button() == Qt.LeftButton:
                    # Start bbox creation
                    self.bbox_start_pos = pos_scene
                    self.temp_bbox_rect = QGraphicsRectItem(pos_scene.x(), pos_scene.y(), 0, 0)
                    self.temp_bbox_rect.setPen(QPen(Qt.yellow, 2 / self.transform().m11()))
                    self.temp_bbox_rect.setBrush(QBrush(QColor(255, 255, 0, 50)))
                    self.temp_bbox_rect.setZValue(100)
                    self.scene.addItem(self.temp_bbox_rect)
                return

            # Polygon creation mode (instance/semantic)
            if event.button() == Qt.LeftButton:
                if len(self.temp_points) > 2:
                    start_p = self.temp_points[0]
                    dist = math.sqrt((pos_scene.x() - start_p.x()) ** 2 + (pos_scene.y() - start_p.y()) ** 2)
                    if dist < 10 / self.transform().m11():
                        self.finish_polygon()
                        return
                self.add_temp_point(pos_scene)
            elif event.button() == Qt.RightButton:
                self.undo_last_point()
            return

        if self.mode == "view":
            if event.button() == Qt.LeftButton:
                # Alt key pressed: enable canvas panning even on masks
                if event.modifiers() == Qt.AltModifier:
                    super().mousePressEvent(event)
                    return

                item = self.itemAt(event.pos())
                if isinstance(item, (VertexHandle, BboxCornerHandle)):
                    super().mousePressEvent(event)
                    return
                if isinstance(item, (EditablePolygonItem, EditableBboxItem)):
                    self.item_clicked.emit(item)
                    if event.modifiers() == Qt.ControlModifier:
                        super().mousePressEvent(event)
                    return
                if item == self.pixmap_item or item is None:
                    self.scene_clicked.emit()
                    super().mousePressEvent(event)
                    return

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """
        Handles double-click events on items.

        Double-clicking a polygon enters editing mode.

        Args:
            event (QMouseEvent): The mouse event details.
        """
        if self.mode == "view" and event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, (EditablePolygonItem, EditableBboxItem)):
                self.item_double_clicked.emit(item)
                event.accept()
                return

        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        """
        Handles mouse move events for bbox dragging.

        Args:
            event (QMouseEvent): The mouse event details.
        """
        if self.mode == "create" and self.bbox_start_pos is not None and self.temp_bbox_rect is not None:
            # Update bbox size during drag
            pos_scene = self.mapToScene(event.pos())
            # Clamp position to image boundaries
            pos_scene = self.clamp_to_image(pos_scene)

            x1, y1 = self.bbox_start_pos.x(), self.bbox_start_pos.y()
            x2, y2 = pos_scene.x(), pos_scene.y()

            # Create rect from top-left to bottom-right
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            self.temp_bbox_rect.setRect(x, y, w, h)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Handles mouse release events to finish bbox creation.

        Args:
            event (QMouseEvent): The mouse event details.
        """
        if self.mode == "create" and event.button() == Qt.LeftButton:
            if self.bbox_start_pos is not None and self.temp_bbox_rect is not None:
                # Finish bbox creation
                rect = self.temp_bbox_rect.rect()

                # Only create bbox if it has some size (at least 5x5 pixels)
                if rect.width() > 5 and rect.height() > 5:
                    self.finish_bbox(rect)
                else:
                    # Too small, cancel
                    if self.temp_bbox_rect.scene():
                        self.scene.removeItem(self.temp_bbox_rect)

                self.bbox_start_pos = None
                self.temp_bbox_rect = None
                return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """
        Handles keyboard input.

        - Enter: Finishes polygon creation.
        - Escape: Cancels polygon creation.

        Args:
            event (QKeyEvent): The key event details.
        """
        if self.mode == "create":
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self.finish_polygon()
                return
            if event.key() == Qt.Key_Escape:
                self.reset_creation()
                return
        super().keyPressEvent(event)

    def add_temp_point(self, pos):
        """
        Adds a temporary point and a line segment to the current drawing path.

        Args:
            pos (QPointF): The scene coordinate of the new point.
        """
        self.temp_points.append(pos)
        r = 4 / self.transform().m11()
        dot = self.scene.addEllipse(pos.x() - r, pos.y() - r, r * 2, r * 2, QPen(Qt.yellow), QBrush(Qt.yellow))
        dot.setZValue(100)
        self.temp_visuals.append(dot)
        if len(self.temp_points) > 1:
            line = QLineF(self.temp_points[-2], self.temp_points[-1])
            l_item = self.scene.addLine(line, QPen(Qt.yellow, 2 / self.transform().m11()))
            l_item.setZValue(99)
            self.temp_visuals.append(l_item)

    def undo_last_point(self):
        """Removes the most recently added temporary point and line segment."""
        if not self.temp_points: return
        self.temp_points.pop()
        if self.temp_visuals:
            item = self.temp_visuals.pop()
            self.scene.removeItem(item)
            if isinstance(item, type(self.scene.addLine(QLineF(), QPen()))):
                if self.temp_visuals:
                    dot = self.temp_visuals.pop()
                    self.scene.removeItem(dot)

    def finish_polygon(self):
        """
        Finalizes the polygon creation, converts temp points to a polygon,
        and notifies the main window.
        """
        if len(self.temp_points) > 2:
            poly = QPolygonF(self.temp_points)
            if self.main_window:
                self.main_window.add_polygon_from_create(poly)
        self.reset_creation()

    def finish_bbox(self, rect):
        """
        Finalizes the bbox creation and notifies the main window.

        Args:
            rect (QRectF): The created bounding box rectangle.
        """
        # Remove temporary visual
        if self.temp_bbox_rect and self.temp_bbox_rect.scene():
            self.scene.removeItem(self.temp_bbox_rect)

        # Create bbox as a polygon (rectangle with 4 points) for compatibility
        bbox_poly = QPolygonF([
            rect.topLeft(),
            rect.topRight(),
            rect.bottomRight(),
            rect.bottomLeft()
        ])

        if self.main_window:
            self.main_window.add_bbox_from_create(rect, bbox_poly)


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Sam3Worker(QThread):
    """
    A QThread worker that runs the SAM3 model inference in the background
    to prevent freezing the UI.
    """
    finished_signal = pyqtSignal(list, float)
    error_signal = pyqtSignal(str)

    def __init__(self, model, processor, image_path, labels):
        """
        Initializes the worker.

        Args:
            model (Sam3Model): The loaded SAM3 model instance.
            processor (Sam3Processor): The loaded SAM3 processor.
            image_path (str): The file path to the image to process.
            labels (list): A list of string prompts (class names) to detect.
        """
        super().__init__()
        self.model = model;
        self.processor = processor;
        self.image_path = image_path;
        self.labels = labels

    def run(self):
        """
        Executes the inference logic:
        1. Loads image.
        2. Preprocesses text prompts.
        3. Runs model inference.
        4. Post-processes masks.
        5. Emits results via signals.
        """
        start_t = time.time();
        results_data = []
        try:
            image = Image.open(self.image_path).convert("RGB")
            for cls_name in self.labels:
                inputs = self.processor(images=image, text=[cls_name], return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                res = self.processor.post_process_instance_segmentation(outputs, threshold=0.4, mask_threshold=0.5,
                                                                        target_sizes=[image.size[::-1]])[0]
                masks = res["masks"].cpu().numpy();
                scores = res["scores"].cpu().numpy()
                for m, s in zip(masks, scores):
                    if not m.any(): continue
                    results_data.append({"label": cls_name, "score": float(s), "mask": m.astype(bool)})
            self.finished_signal.emit(results_data, time.time() - start_t)
        except Exception as e:
            self.error_signal.emit(str(e))


def mask_to_polygon(mask_np):
    """
    Converts a binary numpy mask into a simplified QPolygonF with bounding box and area.

    Uses OpenCV's findContours and approxPolyDP.

    Args:
        mask_np (numpy.ndarray): Boolean or binary mask array.

    Returns:
        tuple: (QPolygonF or None, bbox, area)
            - polygon: The resulting polygon, or None if no valid contour is found.
            - bbox: [x_min, y_min, x_max, y_max] or None
            - area: Pixel area of the mask or 0
    """
    mask_u8 = (mask_np.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0

    c = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    poly = QPolygonF()
    for point in approx:
        poly.append(QPointF(point[0][0], point[0][1]))

    # Calculate bbox from mask
    ys, xs = np.where(mask_np)
    if len(xs) > 0 and len(ys) > 0:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    else:
        bbox = None

    # Calculate area
    area = float(cv2.contourArea(c))

    return poly, bbox, area


def fuse_connected_instances(instances_by_class, img_height, img_width):
    """
    Fuses connected instances of the same class while resolving cross-class conflicts by confidence score.

    Only spatially connected instances (same class) are merged together.
    Different classes compete based on confidence scores at overlapping pixels.

    Args:
        instances_by_class (dict): {
            'Car': [(mask1, score1, bbox1), (mask2, score2, bbox2), ...],
            'Person': [(mask3, score3, bbox3), ...]
        }
        img_height (int): Image height (not used, inferred from masks)
        img_width (int): Image width (not used, inferred from masks)

    Returns:
        dict: {
            'Car': [
                {'polygon': QPolygonF, 'component_id': 1, 'area': float},
                {'polygon': QPolygonF, 'component_id': 2, 'area': float}  # Disconnected region
            ],
            'Person': [...]
        }
    """
    # Get actual dimensions from first available mask
    first_mask = None
    for instances in instances_by_class.values():
        if instances:
            first_mask = instances[0][0]
            break

    if first_mask is None:
        return {}

    actual_height, actual_width = first_mask.shape

    # Step 1: Resolve cross-class conflicts based on confidence scores
    priority_map = np.zeros((actual_height, actual_width), dtype=np.float32)
    class_assignment = np.zeros((actual_height, actual_width), dtype=np.int32)  # 0=background

    class_to_id = {}
    id_to_class = {0: 'background'}
    class_id_counter = 1

    for class_name in instances_by_class.keys():
        class_to_id[class_name] = class_id_counter
        id_to_class[class_id_counter] = class_name
        class_id_counter += 1

    # Assign each pixel to the class with highest confidence
    for class_name, instances in instances_by_class.items():
        class_id = class_to_id[class_name]
        for mask, score, bbox in instances:
            # Only update pixels where this instance has higher confidence
            update_region = (mask > 0) & (score > priority_map)
            class_assignment[update_region] = class_id
            priority_map[update_region] = score

    # Step 2: For each class, extract connected components
    semantic_results = {}

    for class_name, class_id in class_to_id.items():
        # Extract all pixels assigned to this class
        class_mask = (class_assignment == class_id).astype(np.uint8)

        if not class_mask.any():
            continue

        # Find connected components
        num_components, labels = cv2.connectedComponents(class_mask)

        # Generate polygon for each connected component
        components = []
        for comp_id in range(1, num_components):  # 0 is background
            component_mask = (labels == comp_id)
            polygon, bbox, area = mask_to_polygon(component_mask)

            if polygon and polygon.count() >= 3:
                components.append({
                    'polygon': polygon,
                    'component_id': comp_id,
                    'area': area,
                    'bbox': bbox
                })

        if components:
            semantic_results[class_name] = components

    return semantic_results


# ------------------------------------------------------
# Main Window
# ------------------------------------------------------
class MainWindow(QMainWindow):
    """
    The main application window managing the UI, application state, file I/O,
    and coordination between the Canvas, Model, and Lists.
    """

    def __init__(self):
        """Initializes the MainWindow, sets up UI, shortcuts, config, and starts model loading."""
        super().__init__()
        self.setWindowTitle("SAM3 Auto-annotator")
        self.resize(1400, 900)

        self.input_dir = "";
        self.output_dir = "";
        self.image_files = [];
        self.current_img_idx = 0
        self.model = None;
        self.processor = None

        self.label_colors = {}
        self.current_color = QColor("red")
        self.polygon_items = []

        # --- [Flag] Dirty Check ---
        self.is_dirty = False
        # --------------------------

        # --- [Annotation Mode] ---
        self.annotation_mode = "instance"  # "detection", "instance", or "semantic"
        # -------------------------

        # --- [Batch Processing Mode] ---
        self.batch_mode = False  # False: single image, True: batch processing
        self.batch_cancelled = False  # Flag to cancel batch processing
        # -------------------------------

        # --- [Undo History] ---
        self.undo_history = []  # Stack to store last 3 single operations (vertex moves, etc.)
        self.max_undo_steps = 3
        # ----------------------

        # --- [Mask Filtering and Visualization] ---
        self.min_mask_area_ratio = 0.0005  # Minimum mask area as ratio of image area (0.05% default)
        self.masks_visible = True  # Toggle to show/hide mask visualization
        # ------------------------------------------

        # --- [Vertex Size Control] ---
        self.vertex_size_multiplier = 1.0  # Default 1.0x, range 0.5x to 2.0x
        # -----------------------------

        # --- [Border Width Control] ---
        self.border_width_multiplier = 1.0  # Default 1.0x, range 0.5x to 2.0x
        # ------------------------------

        self.init_ui()
        self.init_shortcuts()
        self.load_config()

        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self.load_model)

    def init_shortcuts(self):
        """Sets up keyboard shortcuts for navigation (A/D), saving (S), delete, and undo."""
        self.shortcut_prev = QShortcut(QKeySequence("A"), self)
        self.shortcut_prev.activated.connect(self.prev_img)
        self.shortcut_next = QShortcut(QKeySequence("D"), self)
        self.shortcut_next.activated.connect(self.next_img)
        self.shortcut_save = QShortcut(QKeySequence("S"), self)
        self.shortcut_save.activated.connect(self.save_and_exit_mode)
        self.shortcut_delete = QShortcut(QKeySequence("Delete"), self)
        self.shortcut_delete.activated.connect(self.delete_polygon)
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo_action)

    def init_ui(self):
        """Constructs the GUI layout, widgets, and connections."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- Left: View ---
        self.view = CanvasView(self)
        self.view.main_window = self
        self.view.item_clicked.connect(self.on_canvas_item_clicked)
        self.view.item_double_clicked.connect(self.on_canvas_item_double_clicked)
        self.view.scene_clicked.connect(self.clear_selection)

        self.view.setStyleSheet("background-color: #333;")
        self.splitter.addWidget(self.view)

        # --- Right: Panel with Tabs ---
        self.right_panel = QWidget()
        self.right_panel.setMinimumWidth(100)
        p_layout = QVBoxLayout(self.right_panel)
        p_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(self.right_panel)

        # Create Tab Widget (no help button at top anymore)
        from PyQt5.QtWidgets import QTabWidget
        self.tab_widget = QTabWidget()
        p_layout.addWidget(self.tab_widget)

        # --- Bottom: Console (in right panel) ---
        console_widget = QWidget()
        console_widget.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        console_widget.setFixedHeight(26)
        console_layout = QHBoxLayout(console_widget)
        console_layout.setContentsMargins(5, 2, 5, 2)

        console_label = QLabel("Status:")
        console_label.setStyleSheet("font-weight: bold;")
        console_layout.addWidget(console_label)

        self.lbl_info = QLabel("Ready")
        console_layout.addWidget(self.lbl_info, 1)

        p_layout.addWidget(console_widget)

        # ===== MAIN TAB =====
        self.tab_main = QWidget()
        main_tab_layout = QVBoxLayout(self.tab_main)
        self.tab_widget.addTab(self.tab_main, "Main")

        # Create vertical splitter for resizable lists
        self.lists_splitter = QSplitter(Qt.Vertical)
        main_tab_layout.addWidget(self.lists_splitter)

        # 1. Image list in Main Tab
        images_widget = QWidget()
        l_images = QVBoxLayout(images_widget)
        l_images.setContentsMargins(0, 0, 0, 0)

        images_label = QLabel("Images")
        images_label.setStyleSheet("font-weight: bold; padding: 5px;")
        l_images.addWidget(images_label)

        self.list_images = QListWidget()
        self.list_images.currentRowChanged.connect(self.on_image_selected)
        l_images.addWidget(self.list_images)

        self.lists_splitter.addWidget(images_widget)

        # ===== SETTINGS TAB =====
        self.tab_settings = QWidget()
        settings_tab_layout = QVBoxLayout(self.tab_settings)
        self.tab_widget.addTab(self.tab_settings, "Settings")

        # 1. Files in Settings Tab
        g_file = QGroupBox("Files")
        l_f = QVBoxLayout()
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Input Folder...")
        self.btn_in_sel = QPushButton("...")
        self.btn_in_sel.setFixedWidth(30)
        self.btn_in_sel.clicked.connect(self.select_input)
        h_in = QHBoxLayout()
        h_in.addWidget(self.txt_input)
        h_in.addWidget(self.btn_in_sel)
        l_f.addLayout(h_in)

        self.txt_output = QLineEdit()
        self.txt_output.setPlaceholderText("Output Folder...")
        self.btn_out_sel = QPushButton("...")
        self.btn_out_sel.setFixedWidth(30)
        self.btn_out_sel.clicked.connect(self.select_output)
        h_out = QHBoxLayout()
        h_out.addWidget(self.txt_output)
        h_out.addWidget(self.btn_out_sel)
        l_f.addLayout(h_out)

        g_file.setLayout(l_f)
        settings_tab_layout.addWidget(g_file, 0)

        # 2. Annotation Mode Selection in Settings Tab
        g_mode = QGroupBox("Annotation Mode")
        l_mode = QHBoxLayout()
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup
        self.radio_detection = QRadioButton("Detection")
        self.radio_instance = QRadioButton("Instance")
        self.radio_semantic = QRadioButton("Semantic")
        self.radio_instance.setChecked(True)  # Default mode

        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.radio_detection)
        self.mode_button_group.addButton(self.radio_instance)
        self.mode_button_group.addButton(self.radio_semantic)

        self.radio_detection.toggled.connect(lambda checked: checked and self.set_annotation_mode("detection"))
        self.radio_instance.toggled.connect(lambda checked: checked and self.set_annotation_mode("instance"))
        self.radio_semantic.toggled.connect(lambda checked: checked and self.set_annotation_mode("semantic"))

        l_mode.addWidget(self.radio_detection)
        l_mode.addWidget(self.radio_instance)
        l_mode.addWidget(self.radio_semantic)
        g_mode.setLayout(l_mode)
        settings_tab_layout.addWidget(g_mode, 0)

        # 2.5. Vertex Size Control in Settings Tab
        g_vertex_size = QGroupBox("Vertex Size")
        l_vertex_size = QVBoxLayout()

        h_vertex_slider = QHBoxLayout()
        lbl_vertex_size = QLabel("Size:")
        self.slider_vertex_size = QSlider(Qt.Horizontal)
        self.slider_vertex_size.setMinimum(5)  # 0.5x
        self.slider_vertex_size.setMaximum(20)  # 2.0x
        self.slider_vertex_size.setValue(10)  # 1.0x default
        self.slider_vertex_size.setTickPosition(QSlider.TicksBelow)
        self.slider_vertex_size.setTickInterval(5)
        self.slider_vertex_size.valueChanged.connect(self.on_vertex_size_changed)

        self.lbl_vertex_size_value = QLabel("1.0x")
        self.lbl_vertex_size_value.setFixedWidth(40)

        h_vertex_slider.addWidget(lbl_vertex_size)
        h_vertex_slider.addWidget(self.slider_vertex_size)
        h_vertex_slider.addWidget(self.lbl_vertex_size_value)

        l_vertex_size.addLayout(h_vertex_slider)
        g_vertex_size.setLayout(l_vertex_size)
        settings_tab_layout.addWidget(g_vertex_size, 0)

        # 2.6. Border Width Control in Settings Tab
        g_border_width = QGroupBox("Border Width")
        l_border_width = QVBoxLayout()

        h_border_slider = QHBoxLayout()
        lbl_border_width = QLabel("Width:")
        self.slider_border_width = QSlider(Qt.Horizontal)
        self.slider_border_width.setMinimum(5)  # 0.5x
        self.slider_border_width.setMaximum(20)  # 2.0x
        self.slider_border_width.setValue(10)  # 1.0x default
        self.slider_border_width.setTickPosition(QSlider.TicksBelow)
        self.slider_border_width.setTickInterval(5)
        self.slider_border_width.valueChanged.connect(self.on_border_width_changed)

        self.lbl_border_width_value = QLabel("1.0x")
        self.lbl_border_width_value.setFixedWidth(40)

        h_border_slider.addWidget(lbl_border_width)
        h_border_slider.addWidget(self.slider_border_width)
        h_border_slider.addWidget(self.lbl_border_width_value)

        l_border_width.addLayout(h_border_slider)
        g_border_width.setLayout(l_border_width)
        settings_tab_layout.addWidget(g_border_width, 0)

        # 2.7. Model Management in Settings Tab
        g_model = QGroupBox("Model Management")
        l_model = QVBoxLayout()

        self.btn_unload_model = QPushButton("Unload SAM3 Model")
        self.btn_unload_model.clicked.connect(self.unload_model)
        self.btn_unload_model.setStyleSheet("background-color: #ff9800; color: white; padding: 8px;")
        self.btn_unload_model.setToolTip("Free GPU/CPU memory by unloading the model")
        l_model.addWidget(self.btn_unload_model)

        self.btn_load_model = QPushButton("Load SAM3 Model")
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_model.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.btn_load_model.setVisible(False)  # Hidden until model is unloaded
        l_model.addWidget(self.btn_load_model)

        g_model.setLayout(l_model)
        settings_tab_layout.addWidget(g_model, 0)

        # 2.8. Help Section (Collapsible) in Settings Tab
        from PyQt5.QtWidgets import QTextBrowser, QToolButton

        # Toggle button with arrow - add directly to settings layout
        self.help_toggle = QToolButton()
        self.help_toggle.setText("▶ Help & Shortcuts")
        self.help_toggle.setCheckable(True)
        self.help_toggle.setChecked(False)
        self.help_toggle.setStyleSheet("""
            QToolButton {
                border: 1px solid #ccc;
                background-color: #f0f0f0;
                padding: 5px;
                text-align: left;
            }
            QToolButton:hover {
                background-color: #e0e0e0;
            }
        """)
        settings_tab_layout.addWidget(self.help_toggle, 0)

        # Help content (always exists in layout, just hidden)
        self.help_content = QTextBrowser()
        self.help_content.setReadOnly(True)
        self.help_content.setOpenExternalLinks(False)
        self.help_content.setMaximumHeight(400)
        self.help_content.setVisible(False)
        self.help_content.setHtml(
            "<b>Keyboard Shortcuts:</b><br>"
            "A: Previous Image<br>"
            "D: Next Image<br>"
            "S: Save JSON & Exit Edit Mode<br>"
            "Delete: Delete Selected Annotation<br>"
            "Ctrl+Z: Undo Last Operation (up to 3 steps)<br><br>"

            "<b>Annotation Modes:</b><br>"
            "Detection: Bounding boxes only<br>"
            "Instance: Polygons with bboxes<br>"
            "Semantic: Fused connected instances<br><br>"

            "<b>Auto-Annotation Modes:</b><br>"
            "Single: Process current image only<br>"
            "Batch: Process all images in folder with progress bar<br>"
            "- Batch mode has option to delete all annotations<br><br>"

            "<b>Create Mode:</b><br>"
            "- Left Click: Add Point<br>"
            "- Right Click: Undo Last Point<br>"
            "- Click near Start Point: Finish Polygon<br>"
            "- Creates annotation matching current mode (Detection/Instance/Semantic)<br>"
            "- Detection mode: polygon converted to bounding box<br><br>"

            "<b>View Mode:</b><br>"
            "- Double Click Annotation: Enter Edit Mode<br>"
            "- Single Click: Select Annotation<br>"
            "- Click Labels List: Change Class of Selected<br>"
            "- Alt+Drag: Pan Canvas (ignores masks)<br><br>"

            "<b>Edit Mode:</b><br>"
            "- Drag Vertex\\Corner: Move Point<br>"
            "- Right Click Vertex: Delete Point (min 3 points)<br>"
            "- Ctrl+Left Click Edge: Insert New Point<br>"
            "- Click Outside: Exit Edit Mode<br><br>"

            "<b>Undo Operations:</b><br>"
            "- Vertex move/delete/insert<br>"
            "- Bbox resize<br>"
            "- Polygon create/delete<br>"
            "- Label change"
        )
        settings_tab_layout.addWidget(self.help_content, 0)

        # Toggle function
        def toggle_help():
            is_checked = self.help_toggle.isChecked()
            self.help_content.setVisible(is_checked)
            self.help_toggle.setText("▼ Help & Shortcuts" if is_checked else "▶ Help & Shortcuts")

        self.help_toggle.toggled.connect(toggle_help)

        # Add stretch to settings tab to push everything to the top
        settings_tab_layout.addStretch()

        # 2. Classes in Main Tab (in splitter)
        classes_widget = QWidget()
        l_classes = QVBoxLayout(classes_widget)
        l_classes.setContentsMargins(0, 0, 0, 0)

        classes_label = QLabel("Classes")
        classes_label.setStyleSheet("font-weight: bold; padding: 5px;")
        l_classes.addWidget(classes_label)

        h_l = QHBoxLayout()
        self.txt_lbl = QLineEdit()

        self.btn_col = QPushButton()
        self.btn_col.setStyleSheet("background-color: red")
        self.btn_col.setFixedSize(24, 24)
        self.btn_col.clicked.connect(self.pick_color)

        self.btn_add = QPushButton("Add")
        self.btn_add.setFixedWidth(50)
        self.btn_add.clicked.connect(self.add_label)

        self.btn_del_lbl = QPushButton("Del")
        self.btn_del_lbl.setFixedWidth(50)
        self.btn_del_lbl.clicked.connect(self.delete_label)
        self.btn_del_lbl.setToolTip("Delete selected class")

        h_l.addWidget(self.txt_lbl)
        h_l.addWidget(self.btn_col)
        h_l.addWidget(self.btn_add)
        h_l.addWidget(self.btn_del_lbl)
        l_classes.addLayout(h_l)

        self.list_labels = QListWidget()
        self.list_labels.itemClicked.connect(self.change_category_of_selected_mask)
        l_classes.addWidget(self.list_labels)

        self.lists_splitter.addWidget(classes_widget)

        # 3. SAM - Batch/Single Mode Selection
        g_sam = QGroupBox("Auto Annotation")
        l_sam = QVBoxLayout()

        # Batch/Single mode radio buttons
        h_sam_mode = QHBoxLayout()
        self.radio_single = QRadioButton("Single")
        self.radio_batch = QRadioButton("Batch")
        self.radio_single.setChecked(True)  # Default to single mode

        self.sam_mode_group = QButtonGroup()
        self.sam_mode_group.addButton(self.radio_single)
        self.sam_mode_group.addButton(self.radio_batch)

        self.radio_single.toggled.connect(lambda checked: checked and self.set_batch_mode(False))
        self.radio_batch.toggled.connect(lambda checked: checked and self.set_batch_mode(True))

        h_sam_mode.addWidget(self.radio_single)
        h_sam_mode.addWidget(self.radio_batch)
        l_sam.addLayout(h_sam_mode)

        # RUN button
        self.btn_sam = QPushButton("RUN");
        self.btn_sam.clicked.connect(self.run_sam);
        self.btn_sam.setEnabled(False)
        self.btn_sam.setStyleSheet("background-color: green; color: white; padding: 10px; font-weight: bold;")
        l_sam.addWidget(self.btn_sam)

        # Progress bar and cancel button (only visible during batch processing)
        h_progress = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v/%m - %p%")
        h_progress.addWidget(self.progress_bar, 1)

        self.btn_cancel_batch = QPushButton("Cancel")
        self.btn_cancel_batch.setVisible(False)
        self.btn_cancel_batch.setFixedWidth(60)
        self.btn_cancel_batch.clicked.connect(self.cancel_batch_processing)
        self.btn_cancel_batch.setStyleSheet("background-color: #ff9800; color: white;")
        h_progress.addWidget(self.btn_cancel_batch)

        l_sam.addLayout(h_progress)

        # Progress label (shows current file being processed)
        self.lbl_progress = QLabel("")
        self.lbl_progress.setVisible(False)
        self.lbl_progress.setWordWrap(True)
        self.lbl_progress.setStyleSheet("color: #666; font-size: 9pt;")
        l_sam.addWidget(self.lbl_progress)

        # Batch delete button (only visible in batch mode)
        self.btn_batch_delete = QPushButton("Batch Delete Annotations")
        self.btn_batch_delete.clicked.connect(self.batch_delete_annotations)
        self.btn_batch_delete.setStyleSheet("background-color: #f44336; color: white; padding: 5px;")
        self.btn_batch_delete.setVisible(False)  # Hidden by default
        l_sam.addWidget(self.btn_batch_delete)

        g_sam.setLayout(l_sam)
        main_tab_layout.addWidget(g_sam, 0)

        # 3. Annotations List in Main Tab (in splitter)
        annotations_widget = QWidget()
        l_annotations = QVBoxLayout(annotations_widget)
        l_annotations.setContentsMargins(0, 0, 0, 0)

        annotations_label = QLabel("Annotations")
        annotations_label.setStyleSheet("font-weight: bold; padding: 5px;")
        l_annotations.addWidget(annotations_label)

        self.list_masks = QListWidget()
        self.list_masks.currentRowChanged.connect(self.on_list_selected)
        l_annotations.addWidget(self.list_masks)

        self.lists_splitter.addWidget(annotations_widget)

        # ===== CONTROLS BELOW SPLITTER =====
        # Mask Visibility Toggle
        from PyQt5.QtWidgets import QCheckBox
        h_visibility = QHBoxLayout()
        self.chk_show_masks = QCheckBox("Show Masks")
        self.chk_show_masks.setChecked(True)  # Default to visible
        self.chk_show_masks.toggled.connect(self.toggle_mask_visibility)
        h_visibility.addWidget(self.chk_show_masks)
        h_visibility.addStretch()
        main_tab_layout.addLayout(h_visibility)

        # 5. Tools in Main Tab
        g_t = QGroupBox("Mode & Tools")
        l_t = QVBoxLayout()
        h_m = QHBoxLayout()
        self.btn_v = QPushButton("Edit")
        self.btn_v.setCheckable(True)
        self.btn_v.setChecked(True)
        self.btn_c = QPushButton("Create")
        self.btn_c.setCheckable(True)
        self.btn_v.clicked.connect(lambda: self.set_mode("view"))
        self.btn_c.clicked.connect(lambda: self.set_mode("create"))
        h_m.addWidget(self.btn_v)
        h_m.addWidget(self.btn_c)
        l_t.addLayout(h_m)

        # Delete and Clear All buttons in same row
        h_del = QHBoxLayout()
        self.btn_del = QPushButton("Delete")
        self.btn_del.clicked.connect(self.delete_polygon)
        h_del.addWidget(self.btn_del)
        self.btn_clear_all = QPushButton("Clear All")
        self.btn_clear_all.clicked.connect(self.clear_all_polygons)
        h_del.addWidget(self.btn_clear_all)
        l_t.addLayout(h_del)

        g_t.setLayout(l_t)
        main_tab_layout.addWidget(g_t, 0)

        # 6. Save & Nav in Main Tab
        self.btn_save = QPushButton("Save (S)")
        self.btn_save.clicked.connect(self.save_and_exit_mode)
        self.btn_save.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        h_n = QHBoxLayout()
        self.btn_p = QPushButton("<< (A)")
        self.btn_n = QPushButton(">> (D)")
        self.btn_p.clicked.connect(self.prev_img)
        self.btn_n.clicked.connect(self.next_img)
        h_n.addWidget(self.btn_p)
        h_n.addWidget(self.btn_n)
        main_tab_layout.addWidget(self.btn_save, 0)
        main_tab_layout.addLayout(h_n, 0)

        # Set initial sizes for horizontal splitter (canvas vs right panel)
        current_w = self.width()
        self.splitter.setSizes([int(current_w * 0.8), int(current_w * 0.2)])

        # Set initial sizes for vertical splitter (three lists)
        # Images: 30%, Classes: 25%, Annotations: 45%
        self.lists_splitter.setSizes([300, 250, 450])

    # --- Dirty State Management ---
    def mark_dirty(self):
        """Marks the current application state as modified (unsaved), adding an asterisk to the title."""
        if not self.is_dirty:
            self.is_dirty = True
            self.setWindowTitle("SAM3 Auto-annotator * (Unsaved)")

    def mark_clean(self):
        """Marks the current application state as saved (clean), removing the asterisk from the title."""
        self.is_dirty = False
        self.setWindowTitle("SAM3 Auto-annotator")

    def check_unsaved_changes(self):
        """
        Checks if there are unsaved changes before performing a navigation action.

        If changes exist, a popup dialog asks the user to Save, Discard, or Cancel.

        Returns:
            bool: True if it is safe to proceed (saved or discarded), False if cancelled.
        """
        if not self.is_dirty:
            return True

        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "Current image has unsaved changes. Save now?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            self.save_result()
            return True
        elif reply == QMessageBox.No:
            return True  # Proceed without saving
        else:
            return False  # Cancel navigation

    def show_help_dialog(self):
        """Displays a message box with usage instructions and keyboard shortcuts."""
        msg = ("<b>Keyboard Shortcuts:</b><br>"
               "A: Previous Image<br>"
               "D: Next Image<br>"
               "S: Save JSON & Exit Edit Mode<br>"
               "Delete: Delete Selected Annotation<br>"
               "Ctrl+Z: Undo Last Operation (up to 3 steps)<br><br>"

               "<b>Annotation Modes:</b><br>"
               "Detection: Bounding boxes only<br>"
               "Instance: Polygons with bboxes<br>"
               "Semantic: Fused connected instances<br><br>"

               "<b>Auto-Annotation Modes:</b><br>"
               "Single: Process current image only<br>"
               "Batch: Process all images in folder with progress bar<br>"
               "- Batch mode has option to delete all annotations<br><br>"

               "<b>Create Mode:</b><br>"
               "- Left Click: Add Point<br>"
               "- Right Click: Undo Last Point<br>"
               "- Click near Start Point: Finish Polygon<br>"
               "- Creates annotation matching current mode (Detection/Instance/Semantic)<br>"
               "- Detection mode: polygon converted to bounding box<br><br>"

               "<b>View Mode:</b><br>"
               "- Double Click Annotation: Enter Edit Mode<br>"
               "- Single Click: Select Annotation<br>"
               "- Click Labels List: Change Class of Selected<br>"
               "- Alt+Drag: Pan Canvas (ignores masks)<br><br>"

               "<b>Edit Mode:</b><br>"
               "- Drag Vertex/Corner: Move Point<br>"
               "- Right Click Vertex: Delete Point (min 3 points)<br>"
               "- Ctrl+Left Click Edge: Insert New Point<br>"
               "- Click Outside: Exit Edit Mode<br><br>"

               "<b>Undo Operations:</b><br>"
               "- Vertex move/delete/insert<br>"
               "- Bbox resize<br>"
               "- Polygon create/delete<br>"
               "- Label change")
        QMessageBox.information(self, "Help", msg)

    # --- Config Persistence ---
    def save_config(self):
        """Saves current input/output paths, label definitions, annotation mode, and vertex size to `config.json`."""
        labels_data = [{"name": n, "color": c.name()} for n, c in self.label_colors.items()]
        data = {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "labels": labels_data,
            "annotation_mode": self.annotation_mode,  # Save current annotation mode
            "vertex_size_multiplier": self.vertex_size_multiplier,  # Save vertex size setting
            "border_width_multiplier": self.border_width_multiplier  # Save border width setting
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except:
            pass

    def load_config(self):
        """
        Loads configuration from `config.json`.
        If successful, sets paths, loads labels, restores annotation mode, and triggers image loading.
        """
        if not os.path.exists(CONFIG_FILE):
            self.add_label_item("car", QColor("red"));
            self.add_label_item("tree", QColor("green"))
            return
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
            self.input_dir = data.get("input_dir", "");
            self.output_dir = data.get("output_dir", "")
            self.txt_input.setText(self.input_dir);
            self.txt_output.setText(self.output_dir)
            if data.get("labels"):
                self.list_labels.clear();
                self.label_colors = {}
                for l in data["labels"]: self.add_label_item(l["name"], QColor(l["color"]))

            # Restore annotation mode
            saved_mode = data.get("annotation_mode", "instance")
            self.annotation_mode = saved_mode
            if saved_mode == "detection":
                self.radio_detection.setChecked(True)
            elif saved_mode == "semantic":
                self.radio_semantic.setChecked(True)
            else:
                self.radio_instance.setChecked(True)

            # Restore vertex size setting
            saved_vertex_size = data.get("vertex_size_multiplier", 1.0)
            self.vertex_size_multiplier = saved_vertex_size
            # Update slider to match loaded value
            slider_value = int(saved_vertex_size * 10)
            self.slider_vertex_size.setValue(slider_value)
            self.lbl_vertex_size_value.setText(f"{saved_vertex_size:.1f}x")

            # Restore border width setting
            saved_border_width = data.get("border_width_multiplier", 1.0)
            self.border_width_multiplier = saved_border_width
            # Update slider to match loaded value
            border_slider_value = int(saved_border_width * 10)
            self.slider_border_width.setValue(border_slider_value)
            self.lbl_border_width_value.setText(f"{saved_border_width:.1f}x")

            if self.input_dir and os.path.exists(self.input_dir):
                self.image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.png'))]
                if self.image_files:
                    self.current_img_idx = 0
                    self.refresh_image_list()
                    self.load_image()
        except:
            pass

    # --- Model/File Logic ---
    def load_model(self):
        """Initializes the SAM3 model and processor."""
        self.lbl_info.setText("Loading Model...")
        QApplication.processEvents()
        try:
            self.model = Sam3Model.from_pretrained(MODEL_PATH).to(DEVICE)
            self.processor = Sam3Processor.from_pretrained(MODEL_PATH)
            self.lbl_info.setText("Model Loaded.")
            self.btn_sam.setEnabled(True)

            # Update model management buttons
            self.btn_unload_model.setVisible(True)
            self.btn_load_model.setVisible(False)
        except Exception as e:
            self.lbl_info.setText(f"Error: {e}")

    def unload_model(self):
        """Unloads the SAM3 model to free memory."""
        if self.model is None:
            self.lbl_info.setText("Model already unloaded.")
            return

        try:
            # Delete model and processor
            del self.model
            del self.processor
            self.model = None
            self.processor = None

            # Clear GPU cache if using CUDA
            if DEVICE == "cuda":
                import torch
                torch.cuda.empty_cache()

            self.lbl_info.setText("Model unloaded. Memory freed.")
            self.btn_sam.setEnabled(False)  # Disable RUN button

            # Update model management buttons
            self.btn_unload_model.setVisible(False)
            self.btn_load_model.setVisible(True)
        except Exception as e:
            self.lbl_info.setText(f"Error unloading: {e}")

    def select_input(self):
        """Opens a directory dialog to select the input image folder."""
        d = QFileDialog.getExistingDirectory(self, "Input")
        if d:
            self.input_dir = d;
            self.txt_input.setText(d)
            self.image_files = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png'))]
            if self.image_files:
                self.current_img_idx = 0
                self.refresh_image_list()
                self.load_image()
            self.save_config()

    def refresh_image_list(self):
        """Refreshes the image list widget with current image files."""
        self.list_images.clear()
        for filename in self.image_files:
            item = QListWidgetItem(filename)
            self.list_images.addItem(item)

        # Select current image
        if 0 <= self.current_img_idx < len(self.image_files):
            self.list_images.setCurrentRow(self.current_img_idx)

    def on_image_selected(self, row):
        """
        Handles image selection from the list.

        Args:
            row (int): The index of the selected image in the list.
        """
        if row < 0 or row >= len(self.image_files):
            return

        # Avoid re-triggering when we programmatically set the selection
        if row == self.current_img_idx:
            return

        # Check for unsaved changes before switching
        if not self.check_unsaved_changes():
            # Restore previous selection if user cancels
            self.list_images.blockSignals(True)
            self.list_images.setCurrentRow(self.current_img_idx)
            self.list_images.blockSignals(False)
            return

        # Switch to selected image
        self.current_img_idx = row
        self.load_image()

    def select_output(self):
        """Opens a directory dialog to select the output JSON folder."""
        d = QFileDialog.getExistingDirectory(self, "Output");
        if d: self.output_dir = d; self.txt_output.setText(d); self.save_config()

    def pick_color(self):
        """Opens a color picker dialog to select a color for new labels."""
        c = QColorDialog.getColor(self.current_color, self)
        if c.isValid(): self.current_color = c; self.btn_col.setStyleSheet(f"background-color: {c.name()}")

    def add_label(self):
        """Adds a new label from the text input field."""
        t = self.txt_lbl.text();
        if t: self.add_label_item(t, self.current_color); self.txt_lbl.clear(); self.save_config()

    def delete_label(self):
        """
        Deletes the currently selected label from the list.
        Prompts for confirmation before deleting.
        """
        row = self.list_labels.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Warning", "Please select a label to delete.")
            return

        item = self.list_labels.item(row)
        label_name = item.text()

        reply = QMessageBox.question(self, 'Delete Label',
                                     f"Are you sure you want to delete the category '{label_name}'?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            if label_name in self.label_colors:
                del self.label_colors[label_name]
            self.list_labels.takeItem(row)
            self.save_config()
            self.lbl_info.setText(f"Deleted label: {label_name}")

    def add_label_item(self, t, c):
        """
        Helper method to create and add a QListWidgetItem for a label.

        Args:
            t (str): Label name.
            c (QColor): Label color.
        """
        if t in self.label_colors: return
        self.label_colors[t] = c;
        it = QListWidgetItem(t)
        p = QPixmap(16, 16);
        p.fill(c);
        it.setIcon(QIcon(p));
        self.list_labels.addItem(it)

    def load_image(self):
        """
        Loads the image at the current index.
        Loads associated JSON annotations if they exist in the output directory.
        """
        if not self.image_files: return

        # Reset dirty state when loading a new image fresh
        self.mark_clean()

        fname = self.image_files[self.current_img_idx]
        path = os.path.join(self.input_dir, fname)
        self.view.set_image(QPixmap(path))

        QTimer.singleShot(50, lambda: self.view.fitInView(self.view.pixmap_item, Qt.KeepAspectRatio))

        self.polygon_items = [];
        self.list_masks.clear()

        loaded = False
        if self.output_dir:
            jp = os.path.join(self.output_dir, os.path.splitext(fname)[0] + ".json")
            if os.path.exists(jp):
                self.load_annotations_from_json(jp);
                loaded = True;
                self.lbl_info.setText(f"Loaded JSON: {fname}")
        if not loaded: self.lbl_info.setText(f"Loaded Image: {fname}")

        # Update image list selection
        self.list_images.blockSignals(True)
        self.list_images.setCurrentRow(self.current_img_idx)
        self.list_images.blockSignals(False)

        # Ensure loading doesn't trigger dirty flag
        self.mark_clean()

    def load_annotations_from_json(self, jp):
        """
        Parses a JSON file with multi-type annotations and recreates polygon items on the canvas.

        Supports loading:
        - Old format: simple polygons (backward compatible)
        - New format: detection_bbox, instance_polygon, semantic_polygon with metadata

        Args:
            jp (str): File path to the JSON annotation file.
        """
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for shape in data.get("shapes", []):
                poly = QPolygonF([QPointF(pt[0], pt[1]) for pt in shape.get("points", [])])

                if poly.count() > 2:
                    # Extract shape type (default to instance_polygon for backward compatibility)
                    shape_type = shape.get("shape_type", "instance_polygon")
                    if shape_type == "polygon":  # Old format compatibility
                        shape_type = "instance_polygon"

                    # Extract metadata
                    bbox = shape.get("bbox", None)
                    score = shape.get("score", None)
                    instance_id = shape.get("instance_id", None)
                    component_id = shape.get("component_id", None)
                    area = shape.get("area", None)

                    self.add_polygon_item(
                        poly,
                        shape.get("label", "obj"),
                        shape_type=shape_type,
                        bbox=bbox,
                        score=score,
                        instance_id=instance_id,
                        component_id=component_id,
                        area=area
                    )

            # Apply visibility filter based on current annotation mode
            self.refresh_canvas_visibility()
        except Exception as e:
            print(f"Error loading annotations: {e}")
            pass

    # --- Interaction Logic ---
    def on_canvas_item_clicked(self, item):
        """
        Syncs canvas selection with the list widget selection.

        Args:
            item (EditablePolygonItem): The item clicked on the canvas.
        """
        # Single click just selects, doesn't enter edit mode
        pass

    def on_canvas_item_double_clicked(self, item):
        """
        Handles double-click on canvas items to enter editing mode.

        Args:
            item (EditablePolygonItem): The item double-clicked on the canvas.
        """
        if item not in self.polygon_items:
            return

        # Check if item matches current annotation mode
        mode_type_map = {
            "detection": "detection_bbox",
            "instance": "instance_polygon",
            "semantic": "semantic_polygon"
        }
        target_type = mode_type_map.get(self.annotation_mode, "instance_polygon")

        if item.shape_type != target_type:
            return

        # Find the index in the filtered list
        filtered_items = [it for it in self.polygon_items if it.shape_type == target_type]
        if item in filtered_items:
            filtered_index = filtered_items.index(item)

            # Save state before entering edit mode
            self.save_undo_state("edit", self.polygon_items.index(item))

            # Clear all editing states
            for it in self.polygon_items:
                it.set_editing(False)
                it.setZValue(0)

            # Set this item to editing
            item.set_editing(True)
            item.setZValue(100)

            # Update list selection
            self.list_masks.setCurrentRow(filtered_index)

            # Update info label
            info_text = f"Editing: {item.label}"
            if item.score is not None:
                info_text += f" (score: {item.score:.2f})"
            if item.instance_id is not None:
                info_text += f" [id: {item.instance_id}]"
            if item.component_id is not None:
                info_text += f" [comp: {item.component_id}]"

            self.lbl_info.setText(info_text)

    def clear_selection(self):
        """Deselects all items in the list and exits editing mode on the canvas."""
        self.list_masks.clearSelection()
        for item in self.polygon_items:
            item.set_editing(False)
            item.setZValue(0)

    def on_list_selected(self, row):
        """
        Handles selection changes in the mask list widget.
        Highlights the corresponding polygon on the canvas.

        Args:
            row (int): The index of the selected row in the filtered list.
        """
        # Clear all editing states
        for item in self.polygon_items:
            item.set_editing(False)
            item.setZValue(0)

        if row < 0:
            return

        # Find the actual polygon item from the filtered list
        mode_type_map = {
            "detection": "detection_bbox",
            "instance": "instance_polygon",
            "semantic": "semantic_polygon"
        }
        target_type = mode_type_map.get(self.annotation_mode, "instance_polygon")

        # Get the nth item matching current mode
        filtered_items = [item for item in self.polygon_items if item.shape_type == target_type]

        if 0 <= row < len(filtered_items):
            target = filtered_items[row]
            target.set_editing(True)
            target.setZValue(100)

            info_text = f"Editing: {target.label}"
            if target.score is not None:
                info_text += f" (score: {target.score:.2f})"
            if target.instance_id is not None:
                info_text += f" [id: {target.instance_id}]"
            if target.component_id is not None:
                info_text += f" [comp: {target.component_id}]"

            self.lbl_info.setText(info_text)

    def change_category_of_selected_mask(self, item):
        """
        Updates the label/category of the currently selected mask based on a click in the labels list.

        Args:
            item (QListWidgetItem): The clicked label item containing the new class name.
        """
        row = self.list_masks.currentRow()
        if row < 0:
            return

        new_label = item.text()
        new_color = self.label_colors.get(new_label)
        if not new_color:
            return

        # Get the filtered list matching current annotation mode
        mode_type_map = {
            "detection": "detection_bbox",
            "instance": "instance_polygon",
            "semantic": "semantic_polygon"
        }
        target_type = mode_type_map.get(self.annotation_mode, "instance_polygon")

        # Find the actual item from the filtered list
        filtered_items = [it for it in self.polygon_items if it.shape_type == target_type]

        if row >= len(filtered_items):
            return

        target_item = filtered_items[row]
        if target_item.label == new_label:
            return

        # Save state before modifying
        actual_index = self.polygon_items.index(target_item)
        self.save_undo_state("modify", actual_index)

        target_item.update_properties(new_label, new_color)

        # Update the list display
        self.refresh_mask_list()
        self.list_masks.setCurrentRow(row)  # Restore selection

        self.lbl_info.setText(f"Modified {target_type.replace('_', ' ')} -> {new_label}")

        self.mark_dirty()  # Modify

        self.list_labels.clearSelection()

    def set_annotation_mode(self, mode):
        """
        Switches the annotation mode (detection/instance/semantic).

        Args:
            mode (str): "detection", "instance", or "semantic"
        """
        self.annotation_mode = mode
        self.refresh_canvas_visibility()
        self.lbl_info.setText(f"Mode: {mode.capitalize()}")
        self.save_config()  # Save mode preference to config

    def toggle_mask_visibility(self, checked):
        """
        Toggles the visibility of all mask annotations on the canvas.

        Args:
            checked (bool): True to show masks, False to hide them
        """
        self.masks_visible = checked

        # When toggling global visibility, reset individual visibility states
        mode_type_map = {
            "detection": "detection_bbox",
            "instance": "instance_polygon",
            "semantic": "semantic_polygon"
        }
        target_type = mode_type_map.get(self.annotation_mode, "instance_polygon")

        for item in self.polygon_items:
            if item.shape_type == target_type:
                # Reset individual visibility to match global setting
                if not hasattr(item, '_individual_hidden'):
                    item._individual_hidden = False
                item._individual_hidden = not checked

        self.refresh_canvas_visibility()

    def on_vertex_size_changed(self, value):
        """
        Handles vertex size slider changes and updates all vertex handles.

        Args:
            value (int): Slider value from 5 to 20 (maps to 0.5x to 2.0x)
        """
        self.vertex_size_multiplier = value / 10.0  # Convert to multiplier
        self.lbl_vertex_size_value.setText(f"{self.vertex_size_multiplier:.1f}x")

        # Update all visible handles
        self.update_all_handle_sizes()
        self.save_config()  # Save to config

    def update_all_handle_sizes(self):
        """Updates the size of all vertex and bbox corner handles based on current multiplier."""
        for item in self.polygon_items:
            # Update polygon vertex handles
            if isinstance(item, EditablePolygonItem) and hasattr(item, 'handles'):
                for handle in item.handles:
                    if isinstance(handle, VertexHandle):
                        handle.update_size(self.vertex_size_multiplier)

            # Update bbox corner handles
            if isinstance(item, EditableBboxItem) and hasattr(item, 'handles'):
                for handle in item.handles:
                    if isinstance(handle, BboxCornerHandle):
                        handle.update_size(self.vertex_size_multiplier)

    def on_border_width_changed(self, value):
        """
        Handles border width slider changes and updates all border widths.

        Args:
            value (int): Slider value from 5 to 20 (maps to 0.5x to 2.0x)
        """
        self.border_width_multiplier = value / 10.0  # Convert to multiplier
        self.lbl_border_width_value.setText(f"{self.border_width_multiplier:.1f}x")

        # Update all border widths
        self.update_all_border_widths()
        self.save_config()  # Save to config

    def update_all_border_widths(self):
        """Updates the border width of all polygons and bboxes."""
        for item in self.polygon_items:
            if isinstance(item, (EditablePolygonItem, EditableBboxItem)):
                if hasattr(item, 'update_pen_width'):
                    item.update_pen_width()

    def refresh_canvas_visibility(self):
        """
        Refreshes the visibility of polygon items based on current annotation mode.

        Only items matching the current mode are visible and editable.
        Items from other modes are completely hidden.
        Respects individual item visibility settings.
        """
        mode_type_map = {
            "detection": "detection_bbox",
            "instance": "instance_polygon",
            "semantic": "semantic_polygon"
        }
        target_type = mode_type_map.get(self.annotation_mode, "instance_polygon")

        for item in self.polygon_items:
            # Initialize individual hidden state if not present
            if not hasattr(item, '_individual_hidden'):
                item._individual_hidden = False

            if item.shape_type == target_type:
                # Check both global and individual visibility
                should_be_visible = self.masks_visible and not item._individual_hidden

                item.setVisible(should_be_visible)
                item.setOpacity(1.0)
                item.setEnabled(should_be_visible)
                item.setFlag(QGraphicsItem.ItemIsSelectable, should_be_visible)

                if not should_be_visible:
                    item.set_editing(False)
            else:
                # Completely hide items from other modes
                item.setVisible(False)
                item.setEnabled(False)
                item.setFlag(QGraphicsItem.ItemIsSelectable, False)
                item.set_editing(False)

        # Refresh the mask list to show only relevant items
        self.refresh_mask_list()

    def refresh_mask_list(self):
        """
        Refreshes the annotation list widget to show items matching the current annotation mode.
        """
        self.list_masks.clear()

        mode_type_map = {
            "detection": "detection_bbox",
            "instance": "instance_polygon",
            "semantic": "semantic_polygon"
        }
        target_type = mode_type_map.get(self.annotation_mode, "instance_polygon")

        for i, item in enumerate(self.polygon_items):
            if item.shape_type == target_type:
                # Format display text based on type
                if item.shape_type == "instance_polygon":
                    text = f"{i}: {item.label}"
                    if item.score is not None:
                        text += f" ({item.score:.2f})"
                elif item.shape_type == "semantic_polygon":
                    text = f"{i}: {item.label}"
                    if item.component_id is not None:
                        text += f" [comp_{item.component_id}]"
                else:  # detection_bbox
                    text = f"{i}: {item.label}"
                    if item.score is not None:
                        text += f" ({item.score:.2f})"

                # Create custom widget with visibility toggle
                list_item = QListWidgetItem(self.list_masks)
                item_widget = self.create_list_item_widget(text, item)
                self.list_masks.addItem(list_item)
                self.list_masks.setItemWidget(list_item, item_widget)
                list_item.setSizeHint(item_widget.sizeHint())

    def create_list_item_widget(self, text, polygon_item):
        """
        Creates a custom widget for a list item with text and visibility toggle button.

        Args:
            text (str): The display text for the annotation
            polygon_item: The polygon/bbox item to control

        Returns:
            QWidget: Custom widget with label and toggle button
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Text label
        label = QLabel(text)
        label.setStyleSheet("padding: 2px;")
        layout.addWidget(label, 1)  # Stretch factor 1 to take available space

        # Visibility toggle button
        btn_visibility = QPushButton()
        btn_visibility.setFixedSize(20, 20)
        btn_visibility.setCheckable(True)
        btn_visibility.setChecked(polygon_item.isVisible())
        btn_visibility.setStyleSheet("""
            QPushButton {
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #d4edda;
                border-color: #28a745;
                color: #28a745;
            }
            QPushButton:hover {
                border: 1px solid #999;
                background-color: #f0f0f0;
            }
        """)

        # Use checkmark for visibility
        btn_visibility.setText("✓" if polygon_item.isVisible() else "")
        btn_visibility.setToolTip("View" if polygon_item.isVisible() else "Hidden")

        # Connect to toggle function
        btn_visibility.clicked.connect(lambda checked: self.toggle_individual_mask_visibility(polygon_item, btn_visibility))

        layout.addWidget(btn_visibility)

        return widget

    def toggle_individual_mask_visibility(self, polygon_item, button):
        """
        Toggles the visibility of an individual mask/annotation.

        Args:
            polygon_item: The polygon or bbox item to toggle
            button: The button widget to update
        """
        # Initialize if not present
        if not hasattr(polygon_item, '_individual_hidden'):
            polygon_item._individual_hidden = False

        # Toggle the individual hidden state
        polygon_item._individual_hidden = not polygon_item._individual_hidden
        is_visible = not polygon_item._individual_hidden

        # Update visibility (only if global masks_visible is True)
        should_be_visible = self.masks_visible and is_visible
        polygon_item.setVisible(should_be_visible)

        # Update button appearance and tooltip
        button.setText("✓" if is_visible else "")
        button.setToolTip("View" if is_visible else "Hidden")
        button.setChecked(is_visible)

        # Update enable state
        if should_be_visible:
            polygon_item.setEnabled(True)
            polygon_item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        else:
            # If hiding, exit edit mode
            polygon_item.set_editing(False)
            polygon_item.setEnabled(False)
            polygon_item.setFlag(QGraphicsItem.ItemIsSelectable, False)

    def set_mode(self, mode):
        """
        Switches the application mode.

        Args:
            mode (str): "view" for navigation/editing, "create" for drawing.
        """
        self.view.mode = mode
        self.btn_v.setChecked(mode == "view");
        self.btn_c.setChecked(mode == "create")
        if mode == "create":
            self.view.setDragMode(QGraphicsView.NoDrag)
            self.list_masks.clearSelection()
            for item in self.polygon_items: item.set_editing(False)
        else:
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.view.reset_creation()

    def save_and_exit_mode(self):
        """Saves current annotations and switches back to 'view' mode."""
        self.save_result()
        self.set_mode("view")
        self.clear_selection()
        self.view.scene.update()

    def on_polygon_geometry_changed(self):
        """
        Callback when polygon geometry is changed (vertex dragging, etc.).
        Saves undo state and marks dirty.
        """
        # Note: We save undo state when entering edit mode, not on every vertex drag
        # to avoid filling undo history with intermediate states
        self.mark_dirty()

    def add_polygon_from_create(self, poly):
        """
        Callback triggered when a user finishes drawing a polygon in Create mode.

        Args:
            poly (QPolygonF): The geometry of the newly created polygon.
        """
        # Save state before adding new polygon
        self.save_undo_state("add")

        row = self.list_labels.currentRow()
        if row >= 0:
            label = self.list_labels.item(row).text()
        elif self.list_labels.count() > 0:
            label = self.list_labels.item(0).text()
        else:
            label = "obj"

        # Determine shape_type based on current annotation mode
        mode_type_map = {
            "detection": "detection_bbox",
            "instance": "instance_polygon",
            "semantic": "semantic_polygon"
        }
        shape_type = mode_type_map.get(self.annotation_mode, "instance_polygon")

        # For detection mode, this shouldn't be called (use add_bbox_from_create instead)
        # But keep for backward compatibility
        bbox = None
        if shape_type == "detection_bbox":
            rect = poly.boundingRect()
            bbox = [int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())]

        self.set_mode("view")
        self.add_polygon_item(poly, label, shape_type=shape_type, bbox=bbox)
        self.list_masks.setCurrentRow(self.list_masks.count() - 1)
        self.mark_dirty()  # Creation

    def add_bbox_from_create(self, rect, bbox_poly):
        """
        Callback triggered when a user finishes creating a bbox in Detection Create mode.

        Args:
            rect (QRectF): The bounding box rectangle.
            bbox_poly (QPolygonF): The bbox as a 4-point polygon.
        """
        # Save state before adding new bbox
        self.save_undo_state("add")

        row = self.list_labels.currentRow()
        if row >= 0:
            label = self.list_labels.item(row).text()
        elif self.list_labels.count() > 0:
            label = self.list_labels.item(0).text()
        else:
            label = "obj"

        # Create bbox
        bbox = [int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())]

        self.set_mode("view")
        self.add_polygon_item(bbox_poly, label, shape_type="detection_bbox", bbox=bbox)
        self.list_masks.setCurrentRow(self.list_masks.count() - 1)
        self.mark_dirty()  # Creation

    def add_polygon_item(self, poly, label, shape_type="instance_polygon", bbox=None, score=None,
                         instance_id=None, component_id=None, area=None):
        """
        Creates an EditablePolygonItem or EditableBboxItem and adds it to the scene.

        Args:
            poly (QPolygonF): Geometry.
            label (str): Class label.
            shape_type (str): "detection_bbox", "instance_polygon", or "semantic_polygon"
            bbox (list): [x_min, y_min, x_max, y_max]
            score (float): Confidence score
            instance_id (int): Instance identifier
            component_id (int): Connected component ID
            area (float): Pixel area
        """
        c = self.label_colors.get(label, QColor("gray"))

        # Create appropriate item type based on shape_type
        if shape_type == "detection_bbox" and bbox is not None:
            item = EditableBboxItem(
                bbox, label, c,
                score=score,
                instance_id=instance_id,
                area=area
            )
        else:
            item = EditablePolygonItem(
                poly, label, c,
                shape_type=shape_type,
                bbox=bbox,
                score=score,
                instance_id=instance_id,
                component_id=component_id,
                area=area
            )

        # Connect change callback to track geometry edits
        item.on_change_callback = self.on_polygon_geometry_changed

        # Initialize individual visibility state
        item._individual_hidden = False

        self.view.scene.addItem(item)
        self.polygon_items.append(item)

        # Only add to list if it matches current mode
        mode_type_map = {
            "detection": "detection_bbox",
            "instance": "instance_polygon",
            "semantic": "semantic_polygon"
        }
        target_type = mode_type_map.get(self.annotation_mode, "instance_polygon")

        if shape_type == target_type:
            text = f"{self.list_masks.count()}: {label}"
            if score is not None:
                text += f" ({score:.2f})"
            if component_id is not None:
                text += f" [comp_{component_id}]"
            self.list_masks.addItem(text)

    def delete_polygon(self):
        """Deletes the currently selected polygon (from list or canvas) and updates the list."""
        item_to_delete = None
        row = -1

        # First priority: Find item being edited on canvas
        for i, item in enumerate(self.polygon_items):
            if hasattr(item, 'is_editing') and item.is_editing:
                item_to_delete = item
                row = i
                break

        # Second priority: Find selected item on canvas (not in edit mode)
        if item_to_delete is None:
            for i, item in enumerate(self.polygon_items):
                if item.isSelected():
                    # Check if it matches current mode
                    mode_type_map = {
                        "detection": "detection_bbox",
                        "instance": "instance_polygon",
                        "semantic": "semantic_polygon"
                    }
                    target_type = mode_type_map.get(self.annotation_mode, "instance_polygon")

                    if item.shape_type == target_type:
                        item_to_delete = item
                        row = i
                        break

        # Third priority: Get selected item from list
        if item_to_delete is None:
            list_row = self.list_masks.currentRow()
            if list_row >= 0 and list_row < len(self.polygon_items):
                item_to_delete = self.polygon_items[list_row]
                row = list_row

        # Delete the found item
        if item_to_delete is not None and row >= 0:
            # Save state before deletion for undo
            self.save_undo_state("delete", row)

            # Exit edit mode if this item is being edited
            if hasattr(item_to_delete, 'is_editing') and item_to_delete.is_editing:
                item_to_delete.set_editing(False)

            self.view.scene.removeItem(item_to_delete)
            del self.polygon_items[row]
            self.list_masks.takeItem(row)
            self.refresh_mask_list_indices()
            self.mark_dirty()  # Deletion
            self.lbl_info.setText(f"Deleted annotation")
        else:
            self.lbl_info.setText("No annotation selected to delete")

    def clear_all_polygons(self):
        """Clears all annotations after user confirmation."""
        if not self.polygon_items:
            return

        reply = QMessageBox.question(
            self,
            'Clear All Annotations',
            f'Are you sure you want to delete all {len(self.polygon_items)} annotations?\n\nThis action cannot be undone.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Clear all items from scene
            for item in self.polygon_items:
                self.view.scene.removeItem(item)

            # Clear internal lists
            self.polygon_items = []
            self.list_masks.clear()
            self.undo_history = []  # Clear undo history
            self.mark_dirty()
            self.lbl_info.setText("All annotations cleared")

    def record_single_action(self, action):
        """
        Records a single atomic action for undo (vertex move, bbox resize, etc.).

        Args:
            action (dict): Action details with 'type' and type-specific data
        """
        self.undo_history.append(action)
        if len(self.undo_history) > self.max_undo_steps:
            self.undo_history.pop(0)

    def save_undo_state(self, action_type, data=None):
        """
        Saves complete annotation state for major operations (delete polygon, add polygon, etc.).

        Args:
            action_type (str): Type of action - "delete", "add", "modify", "edit"
            data: Additional data depending on action type
        """
        # Create a deep copy of current polygon state
        state = {
            'action': action_type,
            'data': data,
            'polygon_items': []
        }

        # Save polygon data
        for item in self.polygon_items:
            poly_data = {
                'polygon': QPolygonF(item.polygon()),  # Deep copy
                'label': item.label,
                'color': QColor(item.base_color),
                'shape_type': item.shape_type,
                'bbox': item.bbox[:] if item.bbox else None,  # Copy list
                'score': item.score,
                'instance_id': item.instance_id,
                'component_id': item.component_id,
                'area': item.area
            }
            state['polygon_items'].append(poly_data)

        # Store as a compound action
        self.record_single_action(state)

    def undo_action(self):
        """Undoes the last single operation (vertex move, bbox resize, etc.)."""
        # In Create mode, undo temporary points instead of history
        if self.view.mode == "create":
            if self.view.temp_points:
                self.view.undo_last_point()
                self.lbl_info.setText("Undo: Last point")
            else:
                self.lbl_info.setText("Nothing to undo in create mode")
            return

        if not self.undo_history:
            self.lbl_info.setText("Nothing to undo")
            return

        action = self.undo_history.pop()
        action_type = action.get('type') or action.get('action')

        if action_type == 'vertex_move':
            # Undo a vertex move
            item = action['item']
            if item in self.polygon_items:
                poly = item.polygon()
                poly.replace(action['index'], action['old_pos'])
                item.setPolygon(poly)
                item.create_handles()
                self.lbl_info.setText("Undo: Vertex move")
                self.mark_dirty()

        elif action_type == 'vertex_delete':
            # Undo a vertex deletion (re-insert the vertex)
            item = action['item']
            if item in self.polygon_items:
                poly = item.polygon()
                poly.insert(action['index'], action['deleted_point'])
                item.setPolygon(poly)
                item.create_handles()
                self.lbl_info.setText("Undo: Vertex delete")
                self.mark_dirty()

        elif action_type == 'vertex_insert':
            # Undo a vertex insertion (remove the vertex)
            item = action['item']
            if item in self.polygon_items:
                poly = item.polygon()
                if poly.count() > 3:
                    poly.remove(action['index'])
                    item.setPolygon(poly)
                    item.create_handles()
                    self.lbl_info.setText("Undo: Vertex insert")
                    self.mark_dirty()

        elif action_type == 'bbox_resize':
            # Undo a bbox resize
            item = action['item']
            if item in self.polygon_items and isinstance(item, EditableBboxItem):
                old_bbox = action['old_bbox']
                x_min, y_min, x_max, y_max = old_bbox
                item.setPos(0, 0)
                item.setRect(x_min, y_min, x_max - x_min, y_max - y_min)
                item.bbox = old_bbox[:]
                item.update_handle_positions()
                self.lbl_info.setText("Undo: Bbox resize")
                self.mark_dirty()

        elif action_type in ['delete', 'add', 'modify', 'edit']:
            # Undo compound operations (restore full state)
            # Clear current items
            for item in self.polygon_items:
                self.view.scene.removeItem(item)

            self.polygon_items = []
            self.list_masks.clear()

            # Restore saved items
            for poly_data in action['polygon_items']:
                self.add_polygon_item(
                    poly_data['polygon'],
                    poly_data['label'],
                    shape_type=poly_data['shape_type'],
                    bbox=poly_data['bbox'],
                    score=poly_data['score'],
                    instance_id=poly_data['instance_id'],
                    component_id=poly_data['component_id'],
                    area=poly_data['area']
                )

            self.refresh_canvas_visibility()
            self.mark_dirty()
            self.lbl_info.setText(f"Undo: {action_type}")

        else:
            self.lbl_info.setText(f"Unknown undo action: {action_type}")

    def refresh_mask_list_indices(self):
        """
        Re-indexes the mask list display after deletions to ensure sequential numbering.

        Only shows items matching the current annotation mode.
        """
        self.refresh_mask_list()

    def set_batch_mode(self, is_batch):
        """
        Switches between single and batch processing mode.

        Args:
            is_batch (bool): True for batch mode, False for single mode
        """
        self.batch_mode = is_batch

        # Update RUN button text
        if is_batch:
            self.btn_sam.setText("RUN BATCH")
            self.btn_batch_delete.setVisible(True)
        else:
            self.btn_sam.setText("RUN")
            self.btn_batch_delete.setVisible(False)

    def run_sam(self):
        """Starts the SAM3 inference thread for single or batch processing."""
        if not self.image_files:
            return
        labels = list(self.label_colors.keys())
        if not labels:
            return

        if self.batch_mode:
            self.run_batch_sam()
        else:
            self.run_single_sam()

    def run_single_sam(self):
        """Runs SAM3 inference on the current image only."""
        labels = list(self.label_colors.keys())
        self.btn_sam.setText("Running...")
        self.btn_sam.setStyleSheet("background-color: orange; color: black; padding: 10px; font-weight: bold;")
        self.btn_sam.setEnabled(False)
        QApplication.processEvents()

        path = os.path.join(self.input_dir, self.image_files[self.current_img_idx])
        self.worker = Sam3Worker(self.model, self.processor, path, labels)
        self.worker.finished_signal.connect(self.sam_done)
        self.worker.error_signal.connect(self.sam_error)
        self.worker.start()

    def run_batch_sam(self):
        """Runs SAM3 inference on all images in the folder with inline progress bar."""
        labels = list(self.label_colors.keys())

        # Check for unsaved changes before batch processing
        if not self.check_unsaved_changes():
            return

        total_images = len(self.image_files)

        # Confirm batch processing
        reply = QMessageBox.question(
            self,
            'Batch Processing',
            f'Process all {total_images} images with auto-annotation?\n\n'
            f'This will generate annotations for all images in the folder.\n'
            f'Existing annotations will be overwritten.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Setup progress bar
        self.batch_cancelled = False
        self.progress_bar.setMaximum(total_images)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_cancel_batch.setVisible(True)
        self.lbl_progress.setVisible(True)

        # Disable UI during batch processing
        self.btn_sam.setEnabled(False)
        self.btn_batch_delete.setEnabled(False)
        self.radio_single.setEnabled(False)
        self.radio_batch.setEnabled(False)

        # Process images
        self.batch_current_index = 0
        self.batch_labels = labels
        self.process_next_batch_image()

    def cancel_batch_processing(self):
        """Cancels the ongoing batch processing."""
        self.batch_cancelled = True
        self.btn_cancel_batch.setEnabled(False)
        self.btn_cancel_batch.setText("Cancelling...")

    def process_next_batch_image(self):
        """Processes the next image in batch mode."""
        if self.batch_cancelled:
            self.batch_processing_finished()
            return

        if self.batch_current_index >= len(self.image_files):
            self.batch_processing_finished()
            return

        # Update progress
        self.progress_bar.setValue(self.batch_current_index)
        current_file = self.image_files[self.batch_current_index]
        self.lbl_progress.setText(f"Processing {self.batch_current_index + 1}/{len(self.image_files)}: {current_file}")
        QApplication.processEvents()

        # Process current image
        path = os.path.join(self.input_dir, current_file)
        self.worker = Sam3Worker(self.model, self.processor, path, self.batch_labels)
        self.worker.finished_signal.connect(self.batch_sam_done)
        self.worker.error_signal.connect(self.batch_sam_error)
        self.worker.start()

    def batch_sam_done(self, res, t):
        """
        Callback for successful batch SAM3 inference on one image.

        Args:
            res (list): List of detection results
            t (float): Processing time
        """
        current_file = self.image_files[self.batch_current_index]

        # Save results to JSON directly without updating UI
        self.save_batch_result(current_file, res)

        # Move to next image
        self.batch_current_index += 1
        self.process_next_batch_image()

    def batch_sam_error(self, err):
        """
        Callback for batch SAM3 inference errors.

        Args:
            err (str): Error message
        """
        current_file = self.image_files[self.batch_current_index]
        print(f"Error processing {current_file}: {err}")

        # Continue with next image even on error
        self.batch_current_index += 1
        self.process_next_batch_image()

    def batch_processing_finished(self):
        """Cleans up after batch processing completes or is cancelled."""
        self.progress_bar.setValue(len(self.image_files))

        # Hide progress UI
        self.progress_bar.setVisible(False)
        self.btn_cancel_batch.setVisible(False)
        self.lbl_progress.setVisible(False)
        self.btn_cancel_batch.setEnabled(True)
        self.btn_cancel_batch.setText("Cancel")

        # Re-enable UI
        self.btn_sam.setEnabled(True)
        self.btn_sam.setText("RUN BATCH")
        self.btn_sam.setStyleSheet("background-color: green; color: white; padding: 10px; font-weight: bold;")
        self.btn_batch_delete.setEnabled(True)
        self.radio_single.setEnabled(True)
        self.radio_batch.setEnabled(True)

        # Update image list to show annotation indicators
        self.refresh_image_list()

        # Reload current image to show results
        self.load_image()

        if self.batch_cancelled:
            self.lbl_info.setText(f"Batch processing cancelled at {self.batch_current_index}/{len(self.image_files)}")
        else:
            self.lbl_info.setText(f"Batch processing complete: {len(self.image_files)} images")

    def save_batch_result(self, filename, res):
        """
        Saves batch processing results to JSON file.

        Args:
            filename (str): Image filename
            res (list): Detection results
        """
        if not self.output_dir:
            return

        # Get image dimensions
        image_path = os.path.join(self.input_dir, filename)
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Organize instances by class for semantic fusion
        instances_by_class = {}
        instance_id_counter = 1

        # Temporary storage for all annotations
        all_shapes = []

        # Calculate image area for filtering
        image_area = img_width * img_height
        min_mask_area = image_area * self.min_mask_area_ratio

        # Step 1: Generate instance and detection annotations
        for r in res:
            mask = r['mask']
            label = r['label']
            score = r['score']

            poly, bbox, area = mask_to_polygon(mask)

            if poly and bbox:
                # Filter out tiny masks based on area threshold
                if area < min_mask_area:
                    continue  # Skip this mask as it's too small

                # Store for semantic fusion
                if label not in instances_by_class:
                    instances_by_class[label] = []
                instances_by_class[label].append((mask, score, bbox))

                # Add instance polygon
                all_shapes.append({
                    "label": label,
                    "points": [[pt.x(), pt.y()] for pt in poly],
                    "shape_type": "instance_polygon",
                    "bbox": bbox,
                    "score": float(score),
                    "instance_id": int(instance_id_counter),
                    "area": float(area)
                })

                # Add detection bbox
                rect = QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                bbox_poly = QPolygonF([
                    rect.topLeft(),
                    rect.topRight(),
                    rect.bottomRight(),
                    rect.bottomLeft()
                ])
                all_shapes.append({
                    "label": label,
                    "points": [[pt.x(), pt.y()] for pt in bbox_poly],
                    "shape_type": "detection_bbox",
                    "bbox": bbox,
                    "score": float(score),
                    "instance_id": int(instance_id_counter),
                    "area": float(area)
                })

                instance_id_counter += 1

        # Step 2: Generate semantic segmentation annotations
        if instances_by_class:
            semantic_results = fuse_connected_instances(instances_by_class, img_height, img_width)

            for class_name, components in semantic_results.items():
                for comp_data in components:
                    # Filter out tiny semantic masks based on area threshold
                    if comp_data['area'] < min_mask_area:
                        continue  # Skip this component as it's too small

                    all_shapes.append({
                        "label": class_name,
                        "points": [[pt.x(), pt.y()] for pt in comp_data['polygon']],
                        "shape_type": "semantic_polygon",
                        "bbox": comp_data['bbox'],
                        "component_id": int(comp_data['component_id']),
                        "area": float(comp_data['area'])
                    })

        # Save to JSON
        data = {
            "imagePath": filename,
            "imageHeight": int(img_height),
            "imageWidth": int(img_width),
            "shapes": all_shapes
        }

        json_path = os.path.join(self.output_dir, os.path.splitext(filename)[0] + ".json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving {json_path}: {e}")

    def sam_done(self, res, t):
        """
        Callback for successful SAM3 inference.

        Generates three types of annotations:
        1. Detection bboxes
        2. Instance segmentation polygons
        3. Semantic segmentation polygons (fused connected instances)

        Args:
            res (list): List of dictionaries containing 'mask', 'score', and 'label'.
            t (float): Time taken for inference.
        """
        self.btn_sam.setText("RUN");
        self.btn_sam.setStyleSheet("background-color: green; color: white; padding: 10px; font-weight: bold;");
        self.btn_sam.setEnabled(True)
        self.lbl_info.setText(f"Done in {t:.2f}s")

        # Clear existing annotations
        for it in self.polygon_items:
            self.view.scene.removeItem(it)
        self.polygon_items = [];
        self.list_masks.clear()

        if res:
            self.mark_dirty()  # Auto annotation results in changes

        # Get image dimensions
        img_height = int(self.view.scene.height())
        img_width = int(self.view.scene.width())

        # Organize instances by class for semantic fusion
        instances_by_class = {}
        instance_id_counter = 1

        # Calculate image area for filtering
        image_area = img_width * img_height
        min_mask_area = image_area * self.min_mask_area_ratio

        # Step 1: Generate instance and detection annotations
        for r in res:
            mask = r['mask']
            label = r['label']
            score = r['score']

            poly, bbox, area = mask_to_polygon(mask)

            if poly and bbox:
                # Filter out tiny masks based on area threshold
                if area < min_mask_area:
                    continue  # Skip this mask as it's too small

                # Store for semantic fusion
                if label not in instances_by_class:
                    instances_by_class[label] = []
                instances_by_class[label].append((mask, score, bbox))

                # Add instance polygon
                self.add_polygon_item(
                    poly, label,
                    shape_type="instance_polygon",
                    bbox=bbox,
                    score=score,
                    instance_id=instance_id_counter,
                    area=area
                )

                # Add detection bbox (pass None for poly, bbox will be used to create EditableBboxItem)
                self.add_polygon_item(
                    None, label,
                    shape_type="detection_bbox",
                    bbox=bbox,
                    score=score,
                    instance_id=instance_id_counter,
                    area=area
                )

                instance_id_counter += 1

        # Step 2: Generate semantic segmentation annotations (fused connected instances)
        if instances_by_class:
            semantic_results = fuse_connected_instances(instances_by_class, img_height, img_width)

            for class_name, components in semantic_results.items():
                for comp_data in components:
                    # Filter out tiny semantic masks based on area threshold
                    if comp_data['area'] < min_mask_area:
                        continue  # Skip this component as it's too small

                    self.add_polygon_item(
                        comp_data['polygon'],
                        class_name,
                        shape_type="semantic_polygon",
                        bbox=comp_data['bbox'],
                        component_id=comp_data['component_id'],
                        area=comp_data['area']
                    )

        # Refresh display based on current mode
        self.refresh_canvas_visibility()

    def sam_error(self, err):
        """
        Callback for SAM3 inference errors.

        Args:
            err (str): Error message.
        """
        self.btn_sam.setText("RUN");
        self.btn_sam.setEnabled(True);
        self.lbl_info.setText(f"Error: {err}")

    def batch_delete_annotations(self):
        """
        Deletes annotation JSON files for all images in the folder.
        """
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "No output folder selected.")
            return

        if not self.image_files:
            QMessageBox.warning(self, "Warning", "No images loaded.")
            return

        # Count existing annotation files
        existing_count = 0
        for image_file in self.image_files:
            json_path = os.path.join(self.output_dir, os.path.splitext(image_file)[0] + ".json")
            if os.path.exists(json_path):
                existing_count += 1

        if existing_count == 0:
            QMessageBox.information(self, "Info", "No annotation files found to delete.")
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            'Batch Delete Annotations',
            f'Found {existing_count} annotation file(s) in the output folder.\n\n'
            f'Are you sure you want to delete all annotation JSON files?\n'
            f'This action cannot be undone.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Setup inline progress bar
        self.batch_cancelled = False
        self.progress_bar.setMaximum(len(self.image_files))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_cancel_batch.setVisible(True)
        self.btn_cancel_batch.setEnabled(True)
        self.lbl_progress.setVisible(True)

        # Disable buttons during deletion
        self.btn_batch_delete.setEnabled(False)
        self.btn_sam.setEnabled(False)

        deleted_count = 0
        for i, image_file in enumerate(self.image_files):
            if self.batch_cancelled:
                break

            self.progress_bar.setValue(i)
            self.lbl_progress.setText(f"Deleting {i + 1}/{len(self.image_files)}: {image_file}")
            QApplication.processEvents()

            json_path = os.path.join(self.output_dir, os.path.splitext(image_file)[0] + ".json")
            if os.path.exists(json_path):
                try:
                    os.remove(json_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {json_path}: {e}")

        self.progress_bar.setValue(len(self.image_files))

        # Hide progress UI
        self.progress_bar.setVisible(False)
        self.btn_cancel_batch.setVisible(False)
        self.lbl_progress.setVisible(False)
        self.btn_cancel_batch.setText("Cancel")

        # Re-enable buttons
        self.btn_batch_delete.setEnabled(True)
        self.btn_sam.setEnabled(True)

        # Clear current annotations from view
        for item in self.polygon_items:
            self.view.scene.removeItem(item)
        self.polygon_items = []
        self.list_masks.clear()
        self.mark_clean()

        # Update image list to remove annotation indicators
        self.refresh_image_list()

        if self.batch_cancelled:
            self.lbl_info.setText(f"Batch delete cancelled: {deleted_count} files deleted")
        else:
            self.lbl_info.setText(f"Batch delete complete: {deleted_count} annotation files deleted")

        QMessageBox.information(
            self,
            "Batch Delete Complete",
            f"Successfully deleted {deleted_count} annotation file(s)."
        )

    def save_result(self):
        """
        Exports the current annotations to a JSON file with all three annotation types.

        Saves detection bboxes, instance polygons, and semantic polygons with their metadata.
        Resets the dirty flag upon success.
        """
        if not self.output_dir:
            self.lbl_info.setText("Warn: No Output Folder Selected");
            return

        fname = self.image_files[self.current_img_idx]
        jp = os.path.join(self.output_dir, os.path.splitext(fname)[0] + ".json")

        # Build shapes list with all annotation types and metadata
        shapes = []
        for item in self.polygon_items:
            shape_data = {
                "label": item.label,
                "points": [[pt.x(), pt.y()] for pt in item.polygon()],
                "shape_type": item.shape_type
            }

            # Add optional metadata fields
            if item.bbox is not None:
                shape_data["bbox"] = item.bbox

            if item.score is not None:
                shape_data["score"] = float(item.score)

            if item.instance_id is not None:
                shape_data["instance_id"] = int(item.instance_id)

            if item.component_id is not None:
                shape_data["component_id"] = int(item.component_id)

            if item.area is not None:
                shape_data["area"] = float(item.area)

            shapes.append(shape_data)

        data = {
            "imagePath": fname,
            "imageHeight": int(self.view.scene.height()),
            "imageWidth": int(self.view.scene.width()),
            "shapes": shapes
        }

        try:
            with open(jp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.lbl_info.setText(f"Saved JSON: {fname}")
            self.mark_clean()  # Reset dirty flag

            # Update image list to show annotation indicator
            self.refresh_image_list()
        except Exception as e:
            self.lbl_info.setText(f"Save Failed: {e}")

    # --- Navigation with Safe Check ---
    def prev_img(self):
        """Moves to the previous image, checking for unsaved changes first."""
        if not self.check_unsaved_changes(): return
        if self.current_img_idx > 0: self.current_img_idx -= 1; self.load_image()

    def next_img(self):
        """Moves to the next image, checking for unsaved changes first."""
        if not self.check_unsaved_changes(): return
        if self.current_img_idx < len(self.image_files) - 1: self.current_img_idx += 1; self.load_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())