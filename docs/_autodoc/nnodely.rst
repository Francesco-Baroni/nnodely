nnodely package
===============

.. COMMENT
    Subpackages
    -----------

.. COMMENT
   toctree::
   :maxdepth: 4

.. COMMENT
   nnodely.exporter
   nnodely.visualizer

.. COMMENT
    Submodules
    ----------

activation module
-----------------

.. automodule:: nnodely.activation
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.activation.Relu
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.activation.Tanh
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.activation.ELU
        :undoc-members:
        :no-inherited-members:

arithmetic module
-----------------

.. automodule:: nnodely.arithmetic
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.arithmetic.Add
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.arithmetic.Sub
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.arithmetic.Mul
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.arithmetic.Div
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.arithmetic.Pow
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.arithmetic.Neg
        :undoc-members:
        :no-inherited-members:

earlystopping module
--------------------

.. automodule:: nnodely.earlystopping
    :undoc-members:
    :no-inherited-members:

    .. autofunction:: nnodely.earlystopping.early_stop_patience
    .. autofunction:: nnodely.earlystopping.select_best_model
    .. autofunction:: nnodely.earlystopping.mean_stopping
    .. autofunction:: nnodely.earlystopping.standard_early_stopping

fir module
----------

.. automodule:: nnodely.fir
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.fir.Fir
        :undoc-members:
        :no-inherited-members:

fuzzify module
--------------

.. automodule:: nnodely.fuzzify
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.fuzzify.Fuzzify
        :undoc-members:
        :no-inherited-members:

initializer module
------------------

.. automodule:: nnodely.initializer
    :undoc-members:
    :no-inherited-members:

    .. autofunction:: nnodely.initializer.init_constant
    .. autofunction:: nnodely.initializer.init_negexp
    .. autofunction:: nnodely.initializer.init_exp
    .. autofunction:: nnodely.initializer.init_lin

input module
------------

.. automodule:: nnodely.input
    :no-undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.input.Input
        :undoc-members:
        :inherited-members:
        :exclude-members: count, reset_count

    .. autoclass:: nnodely.input.State
        :undoc-members:
        :inherited-members:
        :exclude-members: count, reset_count


linear module
-------------

.. automodule:: nnodely.linear
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.linear.Linear
        :undoc-members:
        :no-inherited-members:

..  
   automodule:: nnodely.activation
   :members:
   :undoc-members:
   :show-inheritance:

.. COMMENT
    nnodely.arithmetic module
    -------------------------

.. COMMENT
   automodule:: nnodely.arithmetic
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.earlystopping module
    ----------------------------

.. COMMENT
   automodule:: nnodely.earlystopping
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.fir module
    ------------------

.. COMMENT
   automodule:: nnodely.fir
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.fuzzify module
    ----------------------

.. COMMENT
   automodule:: nnodely.fuzzify
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.initializer module
    --------------------------

.. COMMENT
   automodule:: nnodely.initializer
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.input module
    --------------------

.. COMMENT
   automodule:: nnodely.input
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.linear module
    ---------------------

.. COMMENT
   automodule:: nnodely.linear
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.localmodel module
    -------------------------

.. COMMENT
   automodule:: nnodely.localmodel
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.logger module
    ---------------------

.. COMMENT
   automodule:: nnodely.logger
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.loss module
    -------------------

.. COMMENT
   automodule:: nnodely.loss
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.model module
    --------------------

.. COMMENT
   automodule:: nnodely.model
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.modeldef module
    -----------------------

.. COMMENT
   automodule:: nnodely.modeldef
   :members:
   :undoc-members:
   :show-inheritance:

Modely 
======

..  automodule:: nnodely.nnodely
   :members:
   :undoc-members:
   :show-inheritance:

..  automodule:: Modely
   :members:
   :undoc-members:
   :show-inheritance:

.. COMMENT
    nnodely.optimizer module
    ------------------------

.. COMMENT
   automodule:: nnodely.optimizer
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.output module
    ---------------------

.. COMMENT
   automodule:: nnodely.output
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.parameter module
    ------------------------

.. COMMENT
   automodule:: nnodely.parameter
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.parametricfunction module
    ---------------------------------

.. COMMENT
   automodule:: nnodely.parametricfunction
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.part module
    -------------------

.. COMMENT
   automodule:: nnodely.part
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.relation module
    -----------------------

.. COMMENT
   automodule:: nnodely.relation
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.trigonometric module
    ----------------------------

.. COMMENT
   automodule:: nnodely.trigonometric
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    nnodely.utils module
    --------------------

.. COMMENT
   automodule:: nnodely.utils
   :members:
   :undoc-members:
   :show-inheritance:
.. COMMENT
    Module contents
    ---------------

.. COMMENT
   automodule:: nnodely
   :members:
   :undoc-members:
   :show-inheritance:
