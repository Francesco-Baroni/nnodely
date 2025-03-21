from nnodely import *
import torch

# Custom visualizer for results
class FunctionVisualizer(TextVisualizer):
    def showResults(self):
        import torch
        import matplotlib.pyplot as plt
        data_x = torch.arange(-20,20,0.1)
        plt.title('Function Data')
        plt.plot(data_x, parametric_fun(data_x, data_a, data_b, data_c,data_d), label=f'y_target')
        plt.plot(data_x, dx_parametric_fun(data_x, data_a, data_b, data_c, data_d), label=f'dy_dx_target')
        for key in self.n4m.model_def['Outputs'].keys():
            plt.plot(data_x, m({'x': data_x})[key],  '-.', label=key)

        plt.grid(True)
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


x = Input('x')
y_target = Input('y')
dy_dx_target = Input('dy_dx')

x_last = x.last()

def parametric_fun(x,a,b,c,d):
    import torch
    return x**3*a+x**2*b+torch.sin(x)*c+d

def dx_parametric_fun(x,a,b,c,d):
    import torch
    return (3*x**2*a)+(2*x*b)+c*torch.cos(x)


fun = ParamFun(parametric_fun)(x_last)
approx_y = Output('out', fun)
approx_dy_dx = Output('d_out', Derivate(fun, x_last))

m = Modely(visualizer=FunctionVisualizer(),seed=12)


# Create the target functions
data_x = torch.rand(10)*200-100
data_a = 0.02
data_b = -0.03
data_c = 15
data_d = 25
dataset = {'x': data_x, 'y': parametric_fun(data_x,data_a,data_b,data_c,data_d), 'dy_dx': dx_parametric_fun(data_x,data_a,data_b,data_c,data_d)}

# d y_approx / d x == dy_dx
# Se x era una time window and dy_dx dovrà essere una time window
m.addModel('model',[approx_dy_dx, approx_y])
m.addMinimize('err', approx_y, y_target.last())
m.addMinimize('sob_err', approx_dy_dx, dy_dx_target.last())
m.neuralizeModel()
m.loadData('data', dataset)
print(m(m.getSamples('data')))

m.trainModel(num_of_epochs=5000,lr=0.1,minimize_gain={'sob_err':5})


