********************
Building the project
********************

This section tells you how to build and execute the sources of the raytracer.

Windows
=======

.. todo::
   Write Windows build guide

Visual Studio
-------------
Download Visual Studio 2019 Community edition from
`here <https://visualstudio.microsoft.com/downloads/>`_. Follow the install
instructions and open Visual Studio.

Linux
=====

Installing all needed tools
---------------------------

You will need a compiler like `GCC <https://gcc.gnu.org/>`_ or
`Clang <https://clang.llvm.org/>`_ and `CMake <https://cmake.org/>`_.
Use your systems's package manager to install the required programs:

Debian / Ubuntu

.. code:: bash

   sudo apt install gcc cmake

Arch / Manjaro

.. code:: bash

   sudo pacman -S gcc cmake

Fedora, Red Hat

.. code:: bash

   sudo yum install gcc cmake

Mandriva

.. code:: bash

   sudo urpmi install gcc cmake

Build using CMake GUI
---------------------

CMake works either with a graphical user interface or in terminal. Using the
GUI you first have to specify where the source is located and where to build
the binaries. Typically the binaries go into a subdirectory of the source
directory ``build/``.

.. image:: ../images/cmake_1.jpg

Next hit *Configure* to let CMake create its caching files.

.. image:: ../images/cmake_2.jpg

Press yes in the prompt to let CMake create the build directory if it does not
yet exist. In the window CMakeSetup leave everything as is unless you know what
you are doing and press *Finish*.

.. image:: ../images/cmake_3.jpg

Now you may configure the project. The variable ``CMAKE_BUILD_TYPE`` should be
set to ``Release``. 

.. image:: ../images/cmake_4.jpg

This will significantly speed up execution of the compiled
program. Lastly press *Generate* to finally let CMake generate the actual build files.

.. image:: ../images/cmake_5.jpg

Now you have to open a terminal and follow the steps from
:ref:`execute-linux-label`

Build using Terminal
--------------------

TL;DR
~~~~~

.. code:: bash

   cd <path/to>/raytracer
   mkdir build
   cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release
   cd build
   cmake --build . -- raytracer
   ./raytracer

Step by Step
~~~~~~~~~~~~

Once everything is installed open a terminal and navigate to project directory
where the sources are located:

.. code:: bash

   cd <path/to>/raytracer

Create a build directory by entering the command:

.. code:: bash

   mkdir build

We will need CMake to setup the build process. By executing the following
command CMake will automatically create a Makefile in your build directory.
The option ``-DCMAKE_BUILD_TYPE=Release`` tells CMake to configure the Makefile
in such way that compiler optimizations are turned on so the program will
take much less time in execution:

.. code:: bash

   cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release

Navigate to the build directory:

.. code:: bash

   cd build

.. _execute-linux-label:

Execute
~~~~~~~

Now everything is set up and the actual build process can start. You can either
let CMake execute your Makefile:

.. code:: bash

   cmake --build . -- raytracer

Or directly execute the Makefile:

.. code:: bash

   make raytracer

To start the compiled binary just type:

.. code:: bash

  ./raytracer

Mac
===

.. todo::
   Write Mac build guide
