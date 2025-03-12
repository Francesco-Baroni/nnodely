Modely
======

..  automodule:: nnodely.nnodely
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: resultAnalysis, getWorkspace

Model structured NN Inputs Outputs and Parameters
=================================================

Input module
------------

.. automodule:: nnodely.input
    :no-undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.input.Input
        :undoc-members:
        :inherited-members:
        :exclude-members: count, resetCount

    .. autoclass:: nnodely.input.State
        :undoc-members:
        :inherited-members:
        :exclude-members: count, resetCount

Output module
-------------

.. automodule:: nnodely.output
    :no-undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.output.Output
        :undoc-members:
        :inherited-members:
        :exclude-members: count, resetCount, closedLoop, connect, tw, sw, z

Parameter module
----------------

.. automodule:: nnodely.parameter
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.parameter.Constant
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.parameter.SampleTime
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.parameter.Parameter
        :undoc-members:
        :no-inherited-members:

Initializer module
^^^^^^^^^^^^^^^^^^

    .. automodule:: nnodely.initializer
        :undoc-members:
        :no-inherited-members:

        .. autofunction:: nnodely.initializer.init_constant
        .. autofunction:: nnodely.initializer.init_negexp
        .. autofunction:: nnodely.initializer.init_exp
        .. autofunction:: nnodely.initializer.init_lin

Relation module
---------------

.. automodule:: nnodely.relation
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.relation.Stream
        :members:
        :no-inherited-members:

Model structured NN building blocks
===================================

Activation module
-----------------

.. automodule:: nnodely.activation
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.activation.Relu
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.activation.ELU
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.activation.Sigmoid
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.activation.Softmax
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.activation.Identity
        :undoc-members:
        :no-inherited-members:

Arithmetic module
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

Fir module
----------

.. automodule:: nnodely.fir
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.fir.Fir
        :undoc-members:
        :no-inherited-members:

Fuzzify module
--------------

.. automodule:: nnodely.fuzzify
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.fuzzify.Fuzzify
        :undoc-members:
        :no-inherited-members:

Equation Learner module
-----------------------

.. automodule:: nnodely.equationlearner
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.equationlearner.EquationLearner
        :undoc-members:
        :no-inherited-members:

Linear module
-------------

.. automodule:: nnodely.linear
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.linear.Linear
        :undoc-members:
        :no-inherited-members:


Localmodel module
-----------------

.. automodule:: nnodely.localmodel
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.localmodel.LocalModel
        :undoc-members:
        :no-inherited-members:

Part module
-----------

.. automodule:: nnodely.part
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.part.Part
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.part.Select
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.part.SamplePart
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.part.SampleSelect
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.part.TimePart
        :undoc-members:
        :no-inherited-members:

Trigonometric module
--------------------

.. automodule:: nnodely.trigonometric
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.trigonometric.Sin
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.trigonometric.Cos
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.trigonometric.Tan
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.trigonometric.Tanh
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.trigonometric.Cosh
        :undoc-members:
        :no-inherited-members:

    .. autoclass:: nnodely.trigonometric.Sech
        :undoc-members:
        :no-inherited-members:

Parametric Function module
--------------------------

.. automodule:: nnodely.parametricfunction
    :undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.parametricfunction.ParamFun
        :undoc-members:
        :no-inherited-members:

Training
========

optimizer module
----------------

.. automodule:: nnodely.optimizer
    :no-undoc-members:
    :no-inherited-members:

    .. autoclass:: nnodely.optimizer.SGD
        :undoc-members:
        :inherited-members:
        :exclude-members: add_option_to_params, replace_key_with_params, get_torch_optimizer

    .. autoclass:: nnodely.optimizer.Adam
        :undoc-members:
        :inherited-members:
        :exclude-members: add_option_to_params, replace_key_with_params, get_torch_optimizer

earlystopping module
--------------------

.. automodule:: nnodely.earlystopping
    :undoc-members:
    :no-inherited-members:

    .. autofunction:: nnodely.earlystopping.early_stop_patience
    .. autofunction:: nnodely.earlystopping.select_best_model
    .. autofunction:: nnodely.earlystopping.mean_stopping
    .. autofunction:: nnodely.earlystopping.standard_early_stopping

Additional information
======================

.. include:: overview.md
   :parser: myst