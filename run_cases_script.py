from moose_calc_routines import *
from sympy import *
import sympy as sp
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
init_printing()

x, y = var('x y')

# INS turbulence
u = 0.4*sin(0.5*pi*x) + 0.4*sin(pi*y) + 0.7*sin(0.2*pi*x*y) + 0.5
v = 0.6*sin(0.8*pi*x) + 0.3*sin(0.3*pi*y) + 0.2*sin(0.3*pi*x*y) + 0.3
p = 0.5*sin(0.5*pi*x) + 1.0*sin(0.3*pi*y) + 0.5*sin(0.2*pi*x*y) + 0.5
k = 0.4*sin(0.7*pi*x) + 0.9*sin(0.7*pi*y) + 0.7*sin(0.4*pi*x*y) + 0.4
eps = 0.6*sin(0.3*pi*x) + 0.9*sin(0.9*pi*y) + 0.8*sin(0.6*pi*x*y) + 0.5

uvec = sp.Matrix([u, v])
# nvecs = {'left' : sp.Matrix([-1, 0]), 'top' : sp.Matrix([0, 1]), \
#          'right' : sp.Matrix([1, 0]), 'bottom' : sp.Matrix([0, -1])}

# neumann_source_dict = {bnd_name : prep_moose_input(wall_function_momentum_traction(uvec, nvec, p, k, eps, x, y, "vel",
#                                                                                    symbolic=False, parts=True)[0]
#                                                    -bc_terms_momentum_traction(uvec, nvec, p, k, eps, x, y,
#                                                                                symbolic=False, parts=True)[0])
#                        for bnd_name, nvec in nvecs.items()}
# neumann_source_dict = {bnd_name : prep_moose_input(ins_epsilon_wall_function_bc(nvec, k, eps, x, y)
#                                                    -bc_terms_eps(nvec, k, eps, x, y)[0,0])
#                        for bnd_name, nvec in nvecs.items()}
# bounds_dict = {'left' : 'top right bottom', 'top' : 'right bottom left',
#                'right' : 'bottom left top', 'bottom' : 'left top right'}

h_list = ['5', '10']
h_array = np.array([.2, .1])
base = "k_epsilon_test_python"

volume_source = {
                'u' : prep_moose_input(L_momentum_laplace(uvec, p, k, eps, x, y)[0]),
                'v' : prep_moose_input(L_momentum_laplace(uvec, p, k, eps, x, y)[1]),
                'p' : prep_moose_input(L_pressure(uvec, x, y)),
                'k' : prep_moose_input(L_kin(uvec, k, eps, x, y)),
                'eps' : prep_moose_input(L_eps(uvec, k , eps, x, y))}
solution_dict = {'u' : u, 'v' : v, 'p' : p,
                 'k' : k,
                 'eps' : eps}

# mms_bc_cases(h_list, neumann_source_dict, volume_source, solution_dict, bounds_dict, base, test_var="u")
# mms_kernel_cases(h_list, volume_source, solution_dict, base)
plot_order_accuracy(h_array, base, optional_save_string='laplace')
plot_order_accuracy(h_array, base, optional_save_string='laplace')
plot_order_accuracy(h_array, base, optional_save_string='laplace')
plot_order_accuracy(h_array, base, optional_save_string='laplace')
