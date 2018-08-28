
Installation
============

The easiest way to install this package is within the Anaconda environment:

.. code:: bash

    conda install -c conda-forge statmorph

Alternatively, assuming that you already have recent versions of scipy,
scikit-image, astropy and photutils installed, statmorph can also be
installed via PyPI:

.. code:: bash

    pip install statmorph

Finally, if you prefer a manual installation, download the latest release
from the `GitHub repository <https://github.com/vrodgom/statmorph/releases>`_,
extract the contents of the zipfile, and run:

.. code:: bash

    python setup.py install

**Running the built-in tests**

To test that the installation was successful, run:

.. code:: bash

    python -c "import statmorph.tests; statmorph.tests.runall()"
