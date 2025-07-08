import torch
from tqdm import tqdm

def _interpolate_between_samples(
        t_grid: torch.Tensor,
        idx_samples: torch.Tensor,
        x_samples: torch.Tensor,
):
    """
    Returns a linear interpolation between the points x_samples at the grid points t_grid[idx_samples].

    :param t_grid: Grid of times.
    :param idx_samples: Indices of times that x values are samples from.
    :param x_samples: Samples at times. Shape must match idx_samples.
    :return: Interpolated x values, should match the shape of t_grid.
    """
    x_interpolated = torch.full_like(t_grid, fill_value=x_samples[0].item(), dtype=torch.float64)
    for j, (i1, i2) in enumerate(zip(idx_samples[:-1], idx_samples[1:])):
        x_interpolated[i1 + 1: i2 + 1] = x_samples[j] + (x_samples[j + 1] - x_samples[j]) * (
                    t_grid[i1 + 1:i2 + 1] - t_grid[i1]) / (t_grid[i2] - t_grid[i1])
    x_interpolated[idx_samples[-1] + 1:] = x_samples[-1]
    return x_interpolated

def collapse_to_solution(
        rhs,
        h,
        t_start,
        t_end,
        idx_samples,
        x_samples,
        transformation_x2z=None,
        N_iter=5000,
        get_w_ODE=None,
        initialize_by_interpolation=True,
        logging_freq_scalars=1,
        logging_freq_grids=10,
        first_deriv_fwd_mode=True,
        show_progress=False,
        get_optimizer_from_params=None,
):
    """
    Runs the ODE Collapser algorithm and returns results.

    :param rhs: Function f(x, xdot, t). Should be able to accept torch.Tensors after JIT-compilation.
    :param h: t-grid step size.
    :param t_start: First t in our discretization grid.
    :param t_end: Last t in our discretization grid.
    :param idx_samples: Indices of t-grid that x-samples are taken from. It is very important that the
    t-grid these indices refer to matches arange(t_start, t_end, h)
    :param x_samples: Sampled values x(t_i), where t_i are the values of t-grid specified by idx_samples.
    :param transformation_x2z: Optional linear transformation to apply to the x(t). The optimization will be
    performed in z-space. This transformation must be invertible. Defaults to transforming to gradient-space.
    :param N_iter: Number of iterations of optimization to perform.
    :param get_w_ODE: Function returning the weight to apply to the L_ODE loss. Default is a warm-up period
    followed by a linear ramp-up.
    :param initialize_by_interpolation: If True, will initialize the x(t) to linearly interpolate between the
    sampled datapoints. If False, will initialize to zero in z-space.
    :param logging_freq_scalars: Frequency with which to log scalar quantities.
    :param logging_freq_grids: Frequency with which to log grid quantities. These are larger and will consume
    more memory if logged frequently. This might also slow down the optimization.
    :param first_deriv_fwd_mode: If True, will use xdot_i = (x_{i+1} - x_i)/h.
    If False, will use xdot_i = (x_i - x_{i-1})/h.
    :param show_progress: If True, will use tqdm to display a progress bar for the optimizer iterations.
    :param get_optimizer_from_params: Function that takes a list of parameters as input and returns the
    optimizer that we will step. Will be passed the z-space discretization as the parameters list.
    If not supplied, will default to constructing a LBFGS optimizer.
    :return: A Dict containing 'x_solution_grid', 'log_scalar' and 'log_grids' as keys.
    'x_solution_grid' corresponds to the end result of the optimization (after transformation back from z-space).
    """
    # Get the number of grid points
    t_grid = torch.arange(t_start, t_end, h)
    n_grid = t_grid.shape[0]

    # If the inputs are not torch tensors, try and make tensors from them.
    assert idx_samples.shape[0] == x_samples.shape[0]
    if not isinstance(idx_samples, torch.Tensor):
        idx_samples = torch.tensor(idx_samples, requires_grad=False)
    if not isinstance(x_samples, torch.Tensor):
        x_samples = torch.tensor(x_samples, requires_grad=False)

    # I believe this is necessary to avoid errors if we ever move the tensors to the gpu
    # (and to avoid a deprecation warning even if they are on the cpu).
    rhs = torch.compile(rhs)

    # Default transformation
    if transformation_x2z is None:
        # The default option transforms the problem into gradient-space plus a constant.
        transformation_x2z = torch.zeros((n_grid, n_grid), dtype=torch.float64)
        transformation_x2z[0, 0] = 1.0  # z_0 = x(t_0)
        for i in range(1, n_grid):
            # z_i = x(t_i) - x(t_{i-1}) for i > 0
            transformation_x2z[i, i] = 1.0
            transformation_x2z[i, i - 1] = -1.0
    else:
        if not isinstance(transformation_x2z, torch.Tensor):
            transformation_x2z = torch.tensor(transformation_x2z)

    # Default loss-weighting schedule
    if get_w_ODE is None:
        def get_w_ODE(it, n_iterations):
            if it < 0.1 * n_iterations:
                # First 10% of steps: optimize mainly for fitting the samples
                w_ode = 1e-2
            elif it >= 0.9 * n_iterations:
                # Final 90% of steps: optimize mainly for satisfying the ODE
                w_ode = 1.0
            else:
                # Linear ramp-up of w_ODE in between these iterations
                w_ode = 1e-2 + (1 - 1e-2) * (it - 0.1 * n_iterations) / (0.8 * n_iterations)
            return w_ode

    # Invert the transformation matrix. We will need this repeatedly later.
    if torch.linalg.det(transformation_x2z).item() < 1e-4:
        error_msg = f'Transformation matrix from [x(t_i)] to [z_i] has a small determinant.'
        raise ValueError(error_msg)
    transformation_z2x = torch.linalg.inv(transformation_x2z)

    # Initialize the solution grid.
    # I don't believe there is any benefit to using random initialization, since this problem does not
    # have the same requirement for symmetry-breaking that exists with the hidden neurons of a neural network.
    if initialize_by_interpolation:
        x_interpolated = _interpolate_between_samples(t_grid, idx_samples, x_samples)
        z_solution_grid = (transformation_x2z @ x_interpolated).detach().clone().to(torch.float64).requires_grad_(True)
    else:
        z_solution_grid = torch.zeros(n_grid, dtype=torch.float64, requires_grad=True)

    # Initialize the optimizer.
    # This problem seems to benefit from using a second-order optimizer (which LBFGS is), and I believe that
    # is due to the Hessian of loss_ODE (see below for definition) having a very large condition number.
    if get_optimizer_from_params is None:
        optimizer = torch.optim.LBFGS(lr=1, history_size=10, params=[z_solution_grid])
    else:
        optimizer = get_optimizer_from_params([z_solution_grid])

    # Loss function definitions

    # The loss for the optimization problem has two parts:
    # loss_data measures the l2 error of the data relative to our current solution x(t_0), x(t_1), ..., x(t_{N-1})
    # loss_ODE measures the l2 norm of the local violation of the ODE.
    # Our aim is to bring loss_ODE to zero while keeping loss_data as small as possible.

    # Note that loss_data is normalized by the number of samples.
    def loss_data(z):
        x = transformation_z2x @ z
        x_at_sample_points = x[idx_samples]
        error_of_samples = (x_at_sample_points - x_samples)
        loss_val = 0.5 * torch.mean(error_of_samples ** 2)
        return loss_val

    # Note that this loss is a sum over only the interior points, 1, 2, ..., N-2.
    # This is because we do not have sufficient data to compute the second derivative at points 0 and N-1.
    # This should be consistent with your intuition: if we simply demand that loss_ODE = 0,
    # we would have N-2 equations in N unknowns.
    # This would (typically) have a two-dimensional space of solutions,
    # which is what we should expect for a 2nd order ODE.
    def loss_ODE(z):
        x = transformation_z2x @ z
        # Note the factor of h^-2. As h is increased (i.e. the grid is made finer) this should converge to the value
        # of the second derivative (so long as x(t) is twice-differentiable).
        second_deriv = h**(-2) * (x[:-2] - 2.0 * x[1:-1] + x[2:])
        # I've set an option for whether to use the forward or the backward numerical first derivative.
        # I don't believe this will make any difference, but it is there to allow testing / experimentation.
        if first_deriv_fwd_mode:
            first_deriv = h**(-1) * (x[2:] - x[1:-1])
        else:
            first_deriv = h**(-1) * (x[1:-1] - x[:-2])
        rhs_val = rhs(x[1:-1], first_deriv, t_grid[1:-1])
        # Note the factor of h. This cancels out the implicit factor of n_grid from the sum.
        # Alternatively, think of this loss as the (approximation to) the integral of the l2-violation of the ODE.
        loss_val = 0.5 * h * torch.sum((second_deriv - rhs_val) ** 2)
        return loss_val

    # Optimization takes place below here

    # Initialize logging history
    log_scalars = []
    log_grids = []

    iterations = tqdm(range(N_iter)) if show_progress else range(N_iter)
    for iteration in iterations:
        w_ODE = get_w_ODE(iteration, N_iter)

        # Pass forward through the network
        loss_data_torch = loss_data(z_solution_grid)
        loss_ODE_torch = loss_ODE(z_solution_grid)
        loss_total_torch = (1.0 - w_ODE) * loss_data_torch + w_ODE * loss_ODE_torch

        # Store these for logging
        loss_val_data = loss_data_torch.detach().item()
        loss_val_ODE = loss_ODE_torch.detach().item()
        loss_val_total = loss_total_torch.detach().item()

        # Step the optimizer, updating z_solution_grid.
        optimizer.zero_grad()
        loss_total_torch.backward()
        # Stepping the LBFGS optimizer requires a closure for evaluating the loss function
        if type(optimizer) is torch.optim.LBFGS:
            optimizer.step(lambda: (1.0 - w_ODE) * loss_data(z_solution_grid) + w_ODE * loss_ODE(z_solution_grid))
        else:
            optimizer.step()

        # In many of my early experiments, the solution became NaN due to numerical instability.
        # If this happens, it is useful to fail at this point.
        # It is also helpful to know which iteration this happened at.
        if z_solution_grid.isnan().any().item():
            error_msg = f'NaNs appeared in solution after iteration {iteration}'
            raise ValueError(error_msg)

        if iteration % logging_freq_scalars == 0:
            log_scalars.append({
                'iteration': iteration,
                'w_ODE': w_ODE,
                'loss_data': loss_val_data,
                'loss_ODE': loss_val_ODE,
                'loss_total': loss_val_total,
            })

        if iteration % logging_freq_grids == 0:
            log_grids.append({
                'iteration': iteration,
                'z_grid': z_solution_grid.detach().numpy(),
                'x_grid': (transformation_z2x @ z_solution_grid).detach().numpy(),
            })

    # Convert the solution back to x-space.
    # This may already be stored in the logs (if logging_freq_grids divides N_iter).
    x_solution_grid = (transformation_z2x @ z_solution_grid).detach().numpy()

    return {
        'x_solution_grid': x_solution_grid,
        'log_scalars': log_scalars,
        'log_grids': log_grids,
    }