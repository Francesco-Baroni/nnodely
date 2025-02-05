from nnodely import *
from pprint import pprint

from nnodely.hyperparameter import MyMathOBJ
test = HyperParameter('TEST')
C = Constant('C', values = test)
a = Input('a',dimensions = test)
b = Input('b')
f = Fuzzify(output_dimension = test, range=[0,10])(b.tw(test))
o = Output('o', Linear(a.last()) + C + Linear(Fir(output_dimension = test*2)(f)))
#
model = Modely(visualizer=TextVisualizer())
model.addModel('model', o)
pprint(model.model_def.json)
model.neuralizeModel(0.01,hyperparameters={'TEST':2})
print(model({'a':[[1,2],[3,2]]}))



test = 4
model = Modely(visualizer=TextVisualizer())
for test in range(4):
    C = Constant('C'+str(test), values = test)
    a = Input('a'+str(test),dimensions = test)
    b = Input('b')
    f = Fuzzify(output_dimension = test, range=[0,10])(b.tw(test))
    o = Output('o'+str(test), Linear(a.last()) + C + Linear(Fir(output_dimension = test*2)(f)))
    #
    model.addModel('model'+str(test), o)
    pprint(model.model_def.json)

model.neuralizeModel(0.01)

print(model({'a':[[1,2],[3,2]]}))

# class MyCustomType:
#     pass


# Creazione di un oggetto con tipo personalizzato
#obj = MyMathOBJ("123", MyCustomType)

# Verifica del tipo e del valore
# print(type(obj))  # MyCustomType
# print(obj)      # "123"
# print(obj.json)      # "123"
# print(obj.dim)      # "123"

# Creazione di un oggetto con un tipo standard
# obj2 = MyMathOBJ("456", int)
#
# print(type(obj2))  # int
# print(obj2)      # "456"