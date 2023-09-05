from ortools.sat.python import cp_model

def main():
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # Create variables
    x = model.NewIntVar(0, 10, 'x')
    y = model.NewIntVar(0, 10, 'y')

    # Add constraints
    model.Add(x + y <= 8)
    model.Add(2 * x + y >= 10)

    # Define the objective (minimize or maximize)
    model.Maximize(x + 2 * y)

    # Solve the problem
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print(f'Optimal solution found:')
        print(f'x = {solver.Value(x)}')
        print(f'y = {solver.Value(y)}')
        print(f'Objective value = {solver.ObjectiveValue()}')
    else:
        print('The problem does not have an optimal solution.')

if __name__ == '__main__':
    main()
