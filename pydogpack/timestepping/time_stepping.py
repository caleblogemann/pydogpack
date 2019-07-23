import numpy as np

def time_step_loop_explicit(q_init, time_initial, time_final, delta_t,
        explicit_runge_kutta, rhs_function):

    time_step_function = (lambda q, time, delta_t: 
        explicit_runge_kutta.time_step(q, time, delta_t, rhs_function))
    return __time_step_loop(q_init, time_initial, time_final, delta_t,
        time_step_function)

def time_step_loop_implicit(q_init, time_initial, time_final, delta_t, 
        implicit_runge_kutta, rhs_function, solve_operator):

    time_step_function = (lambda q, time, delta_t: 
        implicit_runge_kutta.time_step(q, time, delta_t, rhs_function,
        solve_operator))
    return __time_step_loop(q_init, time_initial, time_final, delta_t,
        time_step_function)

def time_step_loop_imex(q_init, time_initial, time_final, delta_t, 
        imex_runge_kutta, explicit_operator, implicit_operator, solve_operator):

    time_step_function = lambda q, time, delta_t: imex_runge_kutta.time_step(q, 
        time, delta_t, explicit_operator, implicit_operator, solve_operator)
    return __time_step_loop(q_init, time_initial, time_final, delta_t,
        time_step_function)

def __time_step_loop(q_init, time_initial, time_final, delta_t,
        time_step_function):
    time_current = time_initial
    q = np.copy(q_init)
    while (time_current < time_final):
        delta_t = min([delta_t, time_final - time_current])
        q = time_step_function(q, time_current, delta_t)
        time_current += delta_t
    return q