.. image:: https://github.com/matthiaskoenig/zonation-image-analysis/raw/develop/docs/images/favicon/zonation-image-analysis-100x100-300dpi.png
   :align: left
   :alt: zonation-image-analysis logo

Zonation image analysis
=======================

.. image:: https://github.com/matthiaskoenig/pymetadata/workflows/CI-CD/badge.svg
   :target: https://github.com/matthiaskoenig/pymetadata/workflows/CI-CD
   :alt: GitHub Actions CI/CD Status

.. image:: https://img.shields.io/pypi/l/pymetadata.svg
   :target: http://opensource.org/licenses/LGPL-3.0
   :alt: GNU Lesser General Public License 3

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Black

.. image:: http://www.mypy-lang.org/static/mypy_badge.svg
   :target: http://mypy-lang.org/
   :alt: mypy

This repository implements functionality for the image analysis based on whole-slide images (WSI) and large images either using brightfield or fluorescence microscopy. A key analysis is the analysis and quantification of zonation patterns from fluorscence images of the liver using double stainings with CYP2E1 and E-catharin.

Features include among others

- helper functions for reading and writing pyramidal formats (OME-TIF)
- tools for quantification of zonation patterns
- image processing helpers

.. image:: https://github.com/matthiaskoenig/zonation-image-analysis/raw/develop/docs/images/zonation.png
   :align: left
   :alt: zonation example

Installation
============

Install python package
----------------------
The latest develop version can be installed from source via::

    pip install git+https://github.com/matthiaskoenig/zonation-image-analysis.git@develop

Or via cloning the repository and installing via::

    mkvirtualenv zonation --python=python3.10
    git clone https://github.com/matthiaskoenig/zonation-image-analysis.git
    cd zonation-image-analysis
    pip install -e . --upgrade
    
For developing use::

    pip install -e .[development] --upgrade


License
=======

* Source Code: `LGPLv3 <http://opensource.org/licenses/LGPL-3.0>`__
* Documentation: `CC BY-SA 4.0 <http://creativecommons.org/licenses/by-sa/4.0/>`__

The zonation-image-analysis source is released under both the GPL and LGPL licenses version 2 or
later. You may choose which license you choose to use the software under.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License or the GNU Lesser General Public
License as published by the Free Software Foundation, either version 2 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

Funding
=======
Matthias König is supported by the Federal Ministry of Education and Research (BMBF, Germany)
within the research network Systems Medicine of the Liver (**LiSyM**, grant number 031L0054) 
and by the German Research Foundation (DFG) within the Research Unit Programme FOR 5151 
"`QuaLiPerF <https://qualiperf.de>`__ (Quantifying Liver Perfusion-Function Relationship in Complex Resection - 
A Systems Medicine Approach)" by grant number 436883643 and by grant number 
465194077 (Priority Programme SPP 2311, Subproject SimLivA).

© 2022-2023 Jonas Küttner, Jan Grzegorzewski, and Matthias König 
