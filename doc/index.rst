Raytracer Logics
================

The :class:`cg::Raytracer` renders a :class:`cg::Scene` into a
:class:`cg::Texture` by recursively tracing reflected objects of type
:class:`cg::Ray`. The rays are being creating with help of a
:class:`cg::Camera` that can either be a :class:`cg::OrthographicCamera` or a
:class:`cg::PerspectiveCamera`. Cameras have a virtual image plane with a
specified resolution. Using the orthographic model the initial ray directions
are parallel and go through each pixel of the camera's image plane. Using the
perspective model rays have the same origin and shoot through each of the image
plane's pixels. 

Rays can intersect with :class:`cg::Renderable` objects of a
:class:`cg::Scene`.  For each ray that was casted through the camera's image
plane every single renderable object is checked for an intersection. If the ray
hits an object at some point this point is going to be shaded using the
:class:`cg::Material` of the renderable object. A :class:`cg::Material` needs a
:class:`cg::ColorSource`.

.. image:: ../images/render_result.jpg

ToDo List
=========

.. todolist::

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   building
   .. todo_list
   api/library_root
