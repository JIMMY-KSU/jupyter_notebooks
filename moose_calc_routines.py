import sympy as sp
import re
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
import os

'''
Calculus methods
'''

def eye2():
    return sp.Matrix([[sp.Integer(1), sp.Integer(0)], [sp.Integer(0), sp.Integer(1)]])

def zeroVec2():
    return sp.Matrix([sp.Integer(0), sp.Integer(0)])

def gradVec2(u_vec, x, y):
    return sp.Matrix([[sp.diff(u_vec[0], x), sp.diff(u_vec[1],x)], [sp.diff(u_vec[0], y), sp.diff(u_vec[1], y)]])

def divTen2(tensor, x, y):
    return sp.Matrix([sp.diff(tensor[0,0], x) + sp.diff(tensor[1,0], y), sp.diff(tensor[0, 1], x) + sp.diff(tensor[1,1], y)])

def divVec2(u_vec, x, y):
    return sp.diff(u_vec[0], x) + sp.diff(u_vec[1], y)

def gradScalar2(u, x, y):
    return sp.Matrix([sp.diff(u, x), sp.diff(u,y)])

def strain_rate(u_vec, x, y):
    return gradVec2(u_vec, x, y) + gradVec2(u_vec, x, y).transpose()

def strain_rate_squared_2(u_vec, x, y):
    tensor = gradVec2(u_vec, x, y) + gradVec2(u_vec, x, y).transpose()
    rv = 0
    for i in range(2):
        for j in range(2):
            rv += tensor[i, j] * tensor[i, j]
    return rv

def laplace2(u, x, y):
    return sp.diff(sp.diff(u, x), x) + sp.diff(sp.diff(u, y), y)

'''
Kernel operators and corresponding surface integral terms
'''

def L_diffusion(u, x, y):
    return -laplace2(u, x, y)

def bc_terms_diffusion(u, nvec, x, y):
    return (-nvec.transpose() * gradScalar2(u, x, y))[0,0]

def L_momentum_traction(uvec, p, k, eps, x, y):
    cmu = 0.09
    mu, rho = sp.var('mu rho')
    visc_term = (-mu * divTen2(gradVec2(uvec, x, y) + gradVec2(uvec, x, y).transpose(), x, y)).transpose()
    conv_term = rho * uvec.transpose() * gradVec2(uvec, x, y)
    pressure_term = gradScalar2(p, x, y).transpose()
    turbulent_visc_term = -(divTen2(rho * cmu * k**2 / eps * (gradVec2(uvec, x, y) + gradVec2(uvec, x, y).transpose()), x, y)).transpose()
    # print(visc_term.shape, conv_term.shape, pressure_term.shape, sep="\n")
    source = conv_term + visc_term + pressure_term + turbulent_visc_term
    return source

def bc_terms_momentum_traction(uvec, nvec, p, k, eps, x, y, symbolic=True, parts=True):
    if symbolic:
        cmu = sp.var('c_{\mu}')
    else:
        cmu = 0.09
    mu, rho = sp.var('mu rho')
    visc_term = (-mu * nvec.transpose() * (gradVec2(uvec, x, y) + gradVec2(uvec, x, y).transpose())).transpose()
    if parts:
        pressure_term = (nvec.transpose() * eye2() * p).transpose()
    else:
        pressure_term = zeroVec2()
    turbulent_visc_term = -(nvec.transpose() * (rho * cmu * k**2 / eps * (gradVec2(uvec, x, y) + gradVec2(uvec, x, y).transpose()))).transpose()
    return visc_term + turbulent_visc_term + pressure_term

def bc_terms_momentum_traction_no_turbulence(uvec, nvec, p, x, y, parts=True):
    mu, rho = sp.var('mu rho')
    # visc_term = (-mu * nvec.transpose() * (gradVec2(uvec, x, y) + gradVec2(uvec, x, y).transpose())).transpose()
    visc_term = (-mu * nvec.transpose() * strain_rate(uvec, x, y)).transpose()
    if parts:
        pressure_term = (nvec.transpose() * eye2() * p).transpose()
    else:
        pressure_term = zeroVec2()
    return visc_term + pressure_term

def L_momentum_laplace(uvec, p, k, eps, x, y):
    cmu = 0.09
    mu, rho = var('mu rho')
    visc_term = (-mu * divTen2(gradVec2(uvec, x, y), x, y)).transpose()
    conv_term = rho * uvec.transpose() * gradVec2(uvec, x, y)
    pressure_term = gradScalar2(p, x, y).transpose()
    turbulent_visc_term = -(divTen2(rho * cmu * k**2 / eps * (gradVec2(uvec, x, y)), x, y)).transpose()
    # print(visc_term.shape, conv_term.shape, pressure_term.shape, sep="\n")
    source = conv_term + visc_term + pressure_term + turbulent_visc_term
    return source

def L_pressure(uvec, x, y):
    return -divVec2(uvec, x, y)

def L_kin(uvec, k, eps, x, y):
    cmu = 0.09
    sigk = 1.
    sigeps = 1.3
    c1eps = 1.44
    c2eps = 1.92
    conv_term = rho * uvec.transpose() * gradScalar2(k, x, y)
    diff_term = - divVec2((mu + rho * cmu * k**2 / eps / sigk) * gradScalar2(k, x, y), x, y)
    creation_term = - rho * cmu * k**2 / eps / 2 * strain_rate_squared_2(uvec, x, y)
    destruction_term = rho * eps
    terms = [conv_term[0,0], diff_term, creation_term, destruction_term]
    L = 0
    for term in terms:
        L += term
    return L

def L_eps(uvec, k, eps, x, y):
    cmu = 0.09
    sigk = 1.
    sigeps = 1.3
    c1eps = 1.44
    c2eps = 1.92
    conv_term = rho * uvec.transpose() * gradScalar2(eps, x, y)
    diff_term = - divVec2((mu + rho * cmu * k**2 / eps / sigeps) * gradScalar2(eps, x, y), x, y)
    creation_term = - rho * c1eps * cmu * k / 2 * strain_rate_squared_2(uvec, x, y)
    destruction_term = rho * c2eps * eps**2 / k
    terms = [conv_term[0,0], diff_term, creation_term, destruction_term]
    L = 0
    for term in terms:
        L += term
    return L

'''
Boundary condition operators
'''

def wall_function_momentum_traction(uvec, nvec, p, k, eps, x, y, tau_type, symbolic=True, parts=True):
    import pdb; pdb.set_trace()
    if symbolic:
        cmu = sp.var('c_{\mu}')
        yStarPlus = sp.var('y_{\mu}')
    else:
        cmu = 0.09
        yStarPlus = 1.1
    if tau_type == "vel":
        uvec_norm = sp.sqrt(uvec.transpose() * uvec)
        uTau = uvec_norm / yStarPlus
    elif tau_type == "kin":
        uTau = cmu**.25 * sp.sqrt(k)
    else:
        raise ValueError("Must either pass 'vel' or 'kin' for tau_type")

    mu, rho = sp.var('mu rho')
    normal_stress_term = (-nvec.transpose() * mu * strain_rate(uvec, x, y) * nvec * nvec.transpose()).transpose()
    tangential_stress_term = uTau / yStarPlus * uvec
    muT = rho * cmu * k * k / eps
    # turbulent_stress_term = (-nvec.transpose() * muT * strain_rate(uvec, x, y)).transpose()
    turbulent_stress_term = (-nvec.transpose() * strain_rate(uvec, x, y)).transpose()
    # turbulent_stress_term = (-nvec.transpose() * sp.Matrix([[1, 1], [1, 1]])).transpose()
    if parts:
        pressure_term = (nvec.transpose() * eye2() * p).transpose()
    else:
        pressure_term = zeroVec2()
    return normal_stress_term + tangential_stress_term + turbulent_stress_term + pressure_term
    # return pressure_term + normal_stress_term + turbulent_stress_term
    # return normal_stress_term + tangential_stress_term + pressure_term

def no_bc_bc(uvec, nvec, p, x, y, parts=True):
    mu, rho = sp.var('mu rho')
    # visc_term = (-mu * nvec.transpose() * (gradVec2(uvec, x, y) + gradVec2(uvec, x, y).transpose())).transpose()
    visc_term = (-mu * nvec.transpose() * strain_rate(uvec, x, y)).transpose()
    # visc_term = (-mu * nvec.transpose() * sp.Matrix([[1, 1], [1, 1]])).transpose()
    import pdb; pdb.set_trace()
    if parts:
        pressure_term = (nvec.transpose() * eye2() * p).transpose()
    else:
        pressure_term = zeroVec2()
    return visc_term + pressure_term

def vacuum(u, nvec):
    return u / sp.Integer(2)

'''
Writing utilities
'''

def prep_moose_input(sym_expr):
    rep1 = re.sub(r'\*\*',r'^',str(sym_expr))
    rep2 = re.sub(r'mu',r'${mu}',rep1)
    rep3 = re.sub(r'rho',r'${rho}',rep2)
    return rep3

def write_all_functions(uVecNew, p, kinNew, epsilonNew, x, y):
    target = open('/home/lindsayad/python/mms_input.txt','w')
    target.write("[Functions]" + "\n")
    target.write("  [./u_source_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + prep_moose_input(L_momentum_traction(uVecNew, p, kinNew, epsilonNew, x, y)[0]) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./v_source_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + prep_moose_input(L_momentum_traction(uVecNew, p, kinNew, epsilonNew, x, y)[1]) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./p_source_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + prep_moose_input(L_pressure(uVecNew, x, y)) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./kin_source_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + prep_moose_input(L_kin(uVecNew, kinNew, epsilonNew, x, y)) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./epsilon_source_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + prep_moose_input(L_eps(uVecNew, kinNew, epsilonNew, x, y)) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./u_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + str(uNew) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./v_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + str(vNew) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./p_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + str(pNew) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./kin_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + str(kinNew) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./epsilon_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + str(epsilonNew) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("[]" + "\n")
    target.close()

def write_reduced_functions(uVecNew, kinNew, epsilonNew, x, y):
    target = open('/home/lindsayad/python/mms_input.txt','w')
    target.write("[Functions]" + "\n")
    target.write("  [./u_source_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + prep_moose_input(L_momentum_traction(uVecNew, sp.Integer(0), kinNew, epsilonNew, x, y)[0]) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./kin_source_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + prep_moose_input(L_kin(uVecNew, kinNew, epsilonNew, x, y)) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./epsilon_source_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + prep_moose_input(L_eps(uVecNew, kinNew, epsilonNew, x, y)) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./u_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + str(prep_moose_input(uVecNew[0])) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./kin_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + str(prep_moose_input(kinNew)) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("  [./epsilon_func]" + "\n")
    target.write("    type = ParsedFunction" + "\n")
    target.write("    value = '" + str(prep_moose_input(epsilonNew)) + "'" + "\n")
    target.write("  [../]" + "\n")
    target.write("[]" + "\n")
    target.close()

from random import randint, random, uniform

def sym_func(x, y, L):
    return round(uniform(.1, .99),1) + round(uniform(.1, .99),1) * sp.sin(round(uniform(.1, .99),1) * sp.pi * x / L) \
                            + round(uniform(.1, .99),1) * sp.sin(round(uniform(.1, .99),1) * sp.pi * y / L) \
                            + round(uniform(.1, .99),1) * sp.sin(round(uniform(.1, .99),1) * sp.pi * x * y / L)

'''
Context manager for changing the current working directory
Courtesy of http://stackoverflow.com/a/13197763/4493669
'''
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

'''
Function for running MMS simulation cases
'''
def run_many_cases(h_list, source_dict, bounds_dict, base):
    with cd("/home/lindsayad/projects/moose/modules/navier_stokes/tests/mms"):
        for h in h_list:
            for bnd, anti_bnds in bounds_dict.items():
                call(["navier_stokes-opt", "-i", base + ".i", "Mesh/nx=" + h, "Mesh/ny=" + h,
                      "mu=1.5", "rho=2.5",
                      "BCs/u/boundary=" + anti_bnds,
                      "BCs/u_fn_neumann/boundary=" + bnd,
                      "Outputs/csv/file_base=" + h + "_" + bnd + "_" + base,
                     "Functions/u_bc_func/value=" + source_dict[bnd]])

'''
Function for preparing order of accuracy plots
'''
def plot_order_accuracy(boundary, h_array, base):
    with cd("/home/lindsayad/projects/moose/modules/navier_stokes/tests/mms"):
        variable_names = {}
        init_data = np.genfromtxt(str(int(1/h_array[0])) + "_" + str(boundary) + "_" + str(base) + ".csv", \
                                  dtype=float, names=True, delimiter=',')
        for name in init_data.dtype.names:
            if name != 'time':
                variable_names[name] = np.array([])
                variable_names[name] = np.append(variable_names[name], init_data[name][-1])
        for h in h_array[1:]:
            data = np.genfromtxt(str(int(1/h)) + "_" + str(boundary) + "_" + str(base) + ".csv", \
                                dtype=float, names=True, delimiter=',')
            for name in variable_names:
                variable_names[name] = np.append(variable_names[name], data[name][-1])
    for name, data_array in variable_names.items():
        plt.scatter(np.log(h_array), np.log(data_array), label=name)
        z = np.polyfit(np.log(h_array), np.log(data_array), 1)
        p = np.poly1d(z)
        plt.plot(np.log(h_array), p(np.log(h_array)), '-')
        equation = "y=%.3fx+%.3f" % (z[0],z[1])
        plot_point = (np.log(h_array).min() + np.log(h_array).max()) / 2.
        plt.annotate(equation, xy=(plot_point, p(plot_point)), xytext=(.9 * plot_point, 1.1 * p(plot_point)),
             arrowprops=dict(arrowstyle='->'))#, connectionstyle='arc3,rad=-0.5'))
    plt.legend()
    plt.savefig("/home/lindsayad/Pictures/" + str(boundary) + "_" + str(base) + ".eps", format='eps')
    plt.show()
