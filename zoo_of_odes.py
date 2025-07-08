import numpy as np
import scipy.integrate as integrate

def get_rhs_func(ode_name, params):
    return _rhs_funcs_and_solns[ode_name][0](**params)

def get_solution(ode_name, params, initial_conditions, t_start, t_end, h):
    # Discrete grid we will return x(t_0), x(t_1), ..., x(t_{N-1}) values for.
    t_grid = np.arange(t_start, t_end, h)
    return _rhs_funcs_and_solns[ode_name][1](**params, **initial_conditions, t=t_grid), t_grid

def _get_rhs_const_acceleration(a):
    def rhs(x, first_deriv, t):
        return a
    return rhs

def _get_soln_const_acceleration(a, x0, v0, t):
    return x0 + v0*t + 0.5*a*t**2

def _get_rhs_damped_harmonic_oscillator(omega, nu):
    omega2 = omega**2
    def rhs(x, first_deriv, t):
        return -nu*first_deriv - omega2*x
    return rhs

def _get_soln_damped_harmonic_oscillator(omega, nu, amplitude0, phase0, t):
    angular_freq = np.sqrt(omega**2 - 0.25*(nu**2))
    return amplitude0 * np.exp(-nu*t/2) * np.sin(angular_freq*t + phase0)

def _get_rhs_qd_oscillator(omega, nu):
    omega2 = omega**2
    def rhs(x, first_deriv, t):
        return -nu*first_deriv*np.abs(first_deriv) - omega2*x
    return rhs

def _get_soln_qd_oscillator(omega, nu, x0, v0, t):
    # I don't have an analytic solution for this case, so integrate numerical using SciPy's solver.
    # Treat as a first-order dynamical system, coordinates [x(t), dx(t)/dt].

    rhs = _get_rhs_qd_oscillator(omega, nu)  # Might as well reuse this
    # Need to give the solver the derivative of all components w.r.t. time.
    def f(t, x_vec):
        return np.array([
            x_vec[1],  # This element is dx/dt
            rhs(x_vec[0], x_vec[1], t),  # This element is d2x/dt2
        ])

    # Solver can use the Jacobian if given it
    def jac(t, x_vec):
        omega2 = omega ** 2
        return np.array([
            [0, 1.0],
            [-omega2, -2 * nu * np.abs(x_vec[1])]
        ])

    ode_integrator = integrate.ode(f=f, jac=jac)
    ode_integrator.set_integrator('vode', atol=1e-8, rtol=1e-6, nsteps=5000)
    ode_integrator.set_initial_value(np.array([x0, v0]), t[0])

    x_to_return = np.zeros_like(t)
    x_to_return[0] = x0

    for i, tt in enumerate(t[1:]):
        x_from_integrator = ode_integrator.integrate(tt)
        try:
            assert ode_integrator.successful()
        except AssertionError:
            print(f'Integrator failed at step {i + 1}, integrating from {t[i]} to {t[i + 1]}')
        x_to_return[i + 1] = x_from_integrator[0]

    return x_to_return

def _get_rhs_quartic_double_well(a, nu):
    def rhs(x, first_deriv, t):
        return -x**3 + a**2 * x - nu*first_deriv
    return rhs

def _get_soln_quartic_double_well(a, nu, x0, v0, t):
    # I don't have an analytic solution for this case, so integrate numerical using SciPy's solver.
    # Treat as a first-order dynamical system, coordinates [x(t), dx(t)/dt].

    rhs = _get_rhs_quartic_double_well(a, nu)  # Might as well reuse this
    # Need to give the solver the derivative of all components w.r.t. time.
    def f(t, x_vec):
        return np.array([
            x_vec[1],  # This element is dx/dt
            rhs(x_vec[0], x_vec[1], t),  # This element is d2x/dt2
        ])

    # Solver can use the Jacobian if given it
    def jac(t, x_vec):
        return np.array([
            [0, 1.0],
            [-3*x_vec[0]**2 + a**2, -nu * x_vec[1]]
        ])

    ode_integrator = integrate.ode(f=f, jac=jac)
    ode_integrator.set_integrator('vode', atol=1e-10, rtol=1e-9, nsteps=5000)
    ode_integrator.set_initial_value(np.array([x0, v0]), t[0])

    x_to_return = np.zeros_like(t)
    x_to_return[0] = x0

    for i, tt in enumerate(t[1:]):
        x_from_integrator = ode_integrator.integrate(tt)
        try:
            assert ode_integrator.successful()
        except AssertionError:
            print(f'Integrator failed at step {i + 1}, integrating from {t[i]} to {t[i + 1]}')
        x_to_return[i + 1] = x_from_integrator[0]

    return x_to_return

_rhs_funcs_and_solns = {
    'constant_acceleration': (_get_rhs_const_acceleration, _get_soln_const_acceleration),
    'damped_harmonic_oscillator': (_get_rhs_damped_harmonic_oscillator, _get_soln_damped_harmonic_oscillator),
    'harmonic_oscillator_with_quadratic_drag': (_get_rhs_qd_oscillator, _get_soln_qd_oscillator),
    'quartic_double_well': (_get_rhs_quartic_double_well, _get_soln_quartic_double_well),
}
