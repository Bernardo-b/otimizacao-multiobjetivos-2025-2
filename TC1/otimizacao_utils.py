import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time
from tqdm import tqdm


def criar_modelo_milp(m_agents, n_tasks, a_recursos, c_custos, b_capacidades,
                      objetivo='custo', epsilon=None):
    """Cria modelo MILP para atribuição de tarefas"""
    model = pyo.ConcreteModel()

    model.I = pyo.RangeSet(0, m_agents - 1)
    model.J = pyo.RangeSet(0, n_tasks - 1)

    model.x = pyo.Var(model.I, model.J, domain=pyo.Binary)
    model.L_max = pyo.Var(domain=pyo.NonNegativeReals)
    model.L_min = pyo.Var(domain=pyo.NonNegativeReals)

    model.rest_atribuicao = pyo.Constraint(model.J,
        rule=lambda m, j: sum(m.x[i, j] for i in m.I) == 1)

    model.rest_capacidade = pyo.Constraint(model.I,
        rule=lambda m, i: sum(a_recursos[i, j] * m.x[i, j] for j in m.J) <= b_capacidades[i])

    model.rest_L_max = pyo.Constraint(model.I,
        rule=lambda m, i: m.L_max >= sum(a_recursos[i, j] * m.x[i, j] for j in m.J))

    model.rest_L_min = pyo.Constraint(model.I,
        rule=lambda m, i: m.L_min <= sum(a_recursos[i, j] * m.x[i, j] for j in m.J))

    if epsilon is not None:
        model.rest_epsilon = pyo.Constraint(rule=lambda m: m.L_max - m.L_min <= epsilon)

    if objetivo == 'custo':
        model.obj = pyo.Objective(
            rule=lambda m: sum(c_custos[i, j] * m.x[i, j] for i in m.I for j in m.J),
            sense=pyo.minimize)
    elif objetivo == 'equilibrio':
        model.obj = pyo.Objective(rule=lambda m: m.L_max - m.L_min, sense=pyo.minimize)

    return model


def resolver_milp(model, a_recursos, c_custos, solver_name='glpk', verbose=False, time_limit=300):
    """Resolve modelo MILP e retorna solução"""
    solver = SolverFactory(solver_name)

    if solver_name == 'glpk':
        solver.options['tmlim'] = time_limit
    elif solver_name == 'cplex':
        solver.options['timelimit'] = time_limit

    inicio = time.time()
    resultado = solver.solve(model, tee=verbose)
    tempo_solucao = time.time() - inicio

    if resultado.solver.termination_condition != pyo.TerminationCondition.optimal:
        return None

    m_agents = len(model.I)
    n_tasks = len(model.J)

    x_sol = np.zeros((m_agents, n_tasks))
    for i in model.I:
        for j in model.J:
            x_sol[i, j] = pyo.value(model.x[i, j])

    f_C = np.sum(c_custos * x_sol)
    cargas = np.sum(a_recursos * x_sol, axis=1)
    f_E = cargas.max() - cargas.min()

    return {
        'x': x_sol,
        'f_C': f_C,
        'f_E': f_E,
        'cargas': cargas,
        'tempo': tempo_solucao,
        'status': str(resultado.solver.termination_condition)
    }


def metodo_epsilon_restrito(m, n, a, c, b, f_E_min, f_E_max, num_pontos=15,
                            solver_name='glpk', verbose=False):
    """Aplica método ε-Restrito para gerar Fronteira de Pareto

    Nota: O primeiro ponto costuma demorar mais (warm-up do solver + problema mais restrito)
    """
    epsilon_values = np.linspace(f_E_min, f_E_max, num_pontos)
    fronteira = []
    tempos = []

    pbar = tqdm(epsilon_values, desc="Gerando Fronteira de Pareto", unit="ponto")
    for idx, eps in enumerate(pbar):
        t_inicio = time.time()

        modelo = criar_modelo_milp(m, n, a, c, b, objetivo='custo', epsilon=eps)
        sol = resolver_milp(modelo, a, c, solver_name=solver_name, verbose=False)

        t_fim = time.time()
        tempo_ponto = t_fim - t_inicio
        tempos.append(tempo_ponto)

        if sol:
            sol['epsilon'] = eps
            fronteira.append(sol)

        if idx == 0:
            pbar.set_postfix_str(f"1º ponto: {tempo_ponto:.1f}s (warm-up)")
        else:
            pbar.set_postfix_str(f"último: {tempo_ponto:.1f}s, média: {np.mean(tempos[1:]):.1f}s")

    return fronteira


def calcular_delta(fronteira, ponto_utopico):
    """Calcula indicador Delta (diversidade)"""
    if len(fronteira) < 2:
        return np.nan

    pontos = np.array([[sol['f_C'], sol['f_E']] for sol in fronteira])
    pontos = pontos[pontos[:, 1].argsort()]

    distancias = np.sqrt(np.sum(np.diff(pontos, axis=0)**2, axis=1))
    d_avg = np.mean(distancias)

    d_f = np.sqrt((pontos[0, 0] - ponto_utopico[0])**2 + (pontos[0, 1] - ponto_utopico[1])**2)
    d_l = np.sqrt((pontos[-1, 0] - ponto_utopico[0])**2 + (pontos[-1, 1] - ponto_utopico[1])**2)

    numerador = d_f + d_l + np.sum(np.abs(distancias - d_avg))
    denominador = d_f + d_l + (len(pontos) - 1) * d_avg

    return numerador / denominador if denominador > 0 else np.nan


def calcular_hipervolume_2d(fronteira, ponto_referencia):
    """Calcula hipervolume 2D (área dominada)"""
    if len(fronteira) == 0:
        return 0.0

    pontos = np.array([[sol['f_C'], sol['f_E']] for sol in fronteira])
    pontos = pontos[pontos[:, 0].argsort()]

    f_C_ref, f_E_ref = ponto_referencia

    area = 0.0
    for i in range(len(pontos)):
        largura = pontos[i, 0] - (f_C_ref if i == 0 else pontos[i-1, 0])
        altura = f_E_ref - pontos[i, 1]
        area += largura * altura

    if len(pontos) > 0:
        area += (f_C_ref - pontos[-1, 0]) * (f_E_ref - pontos[-1, 1])

    return abs(area)
