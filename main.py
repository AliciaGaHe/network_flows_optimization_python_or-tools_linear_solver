import pandas as pd
import json
from ortools.linear_solver import pywraplp


def solve_problem_using_ortools_linear_solver(data_file):
    """solve the problem using or-tools linear solver and print results"""
    ### Load data ###

    # Open json file with data
    f = open(data_file)

    # Returns the data as a dictionary
    data = json.load(f)

    # Close the file
    f.close()

    ### Create the objects that the optimization model needs ###

    # Lists with sources and customers
    sSources = data["sSources"]
    sCustomers = data["sCustomers"]

    # Production limit for each source and demand for each customer
    pSourceProduction = data["pSourceProduction"]
    pCustomerDemand = data["pCustomerDemand"]

    # Transportation costs for each source and customer
    pTransportationCosts = {tuple(item["index"]): item["value"] for item in data["pTransportationCosts"]}

    # List with the available combinations of sources and customers
    sSources_Customers = [tuple(item["index"]) for item in data["pTransportationCosts"]]

    # Quantity that is mandatory to move between each source and customer
    pFixedTransportation = {tuple(item["index"]): item["value"] for item in data["pFixedTransportation"]}

    ### Create the model ###

    # Instantiate a Glop solver and naming it
    solver = pywraplp.Solver.CreateSolver("GLOP")

    # Create the decision variables
    vQuantityExchanged = {
        (s, c):  solver.NumVar(0.0, solver.infinity(), "quantity_in_tons_" + str(s) + "_" + str(c))
        for s in sSources for c in sCustomers if (s, c) in sSources_Customers
    }

    # print("Number of variables =", solver.NumVariables())

    # Create the constraints

    # Production limit for each source
    for s in sSources:
        solver.Add(
            sum(
                vQuantityExchanged[s, c] for c in sCustomers if (s, c) in sSources_Customers
            )
            <= pSourceProduction[s],
            "c01_production_%s" % s
        )

    # Demand limit for each customer
    for c in sCustomers:
        solver.Add(
            sum(
                vQuantityExchanged[s, c] for s in sSources if (s, c) in sSources_Customers
            )
            >= pCustomerDemand[c],
            "c02_demand_%s" % c
        )

    # Quantity that is mandatory to move between each source and each customer
    for s in sSources:
        for c in sCustomers:
            if (s, c) in sSources_Customers and pFixedTransportation[s, c]:
                solver.Add(
                    vQuantityExchanged[s, c] == pFixedTransportation[s, c],
                    "c03_fixed_%s_%s" % (s, c)
                )

    # print("Number of constraints =", solver.NumConstraints())

    # Create objective function
    solver.Minimize(
        sum(
            pTransportationCosts[s, c] * vQuantityExchanged[s, c]
            for s in sSources for c in sCustomers if (s, c) in sSources_Customers
        )
    )

    ### Solve the model ###
    status_code = solver.Solve()

    ### Print results if we have get an optimal solution ###
    if status_code != solver.FEASIBLE and status_code != solver.OPTIMAL:

        print("The solver could not solve the problem.")

    else:

        if status_code == solver.OPTIMAL:
            status_text = "Optimal"
        else:
            status_text = "Feasible, but not optimal"

        print("\n")
        print("Solver status: ", status_text, "\n")

        print("Total transportation cost: ", solver.Objective().Value(), "\n")

        print("Quantity exchanged between sources and customers:")
        dict_quantity_sources_customers = {
            (s, c): vQuantityExchanged[s, c].solution_value()
            for s in sSources for c in sCustomers
            if (s, c) in sSources_Customers and vQuantityExchanged[s, c].solution_value() > 0
        }
        df_quantity_sources_customers = pd.DataFrame(dict_quantity_sources_customers.values(),
                                                     index=pd.MultiIndex.from_tuples(
                                                         dict_quantity_sources_customers.keys()),
                                                     columns=['Quantity']
                                                     ).reset_index(names=['Source', 'Customer'])
        print(df_quantity_sources_customers)
        print("\n")

        print("Sensibility analysis - constraints:")
        activities = solver.ComputeConstraintActivities()
        list_sensibility_analysis_constraints = [
            {
                'Constraint': c.name(),
                'Slack': c.ub() - activities[i] if c.ub() - activities[i] != float("inf") else 0.0,
                'Shadow price': c.dual_value()
            }
            for i, c in enumerate(solver.constraints())
        ]
        df_sensibility_analysis_constraints = pd.DataFrame(list_sensibility_analysis_constraints)
        print(df_sensibility_analysis_constraints)
        print("\n")

        # Conclusions
        df_sensibility_analysis_constraints[
            df_sensibility_analysis_constraints['Slack'] == 0
            ].apply(
            lambda row:
            print_conclusions_constraints_sensibility_analysis(row['Constraint'], row['Shadow price'], sSources), axis=1)
        print("\n")

        print("Sensibility analysis - variables:")
        list_sensibility_analysis_variables = [
            {
                'Variable': v.name(),
                'Value': v.solution_value(),
                'Reduced cost': v.reduced_cost()
            }
            for v in solver.variables()]
        df_sensibility_analysis_variables = pd.DataFrame(list_sensibility_analysis_variables)
        print(df_sensibility_analysis_variables)
        print("\n")

        # Conclusions
        df_sensibility_analysis_variables[
            df_sensibility_analysis_variables['Value'] == 0
            ].apply(
            lambda row:
            print_conclusions_variables_sensibility_analysis(row['Variable'], row['Reduced cost']), axis=1)
        print("\n")


def print_conclusions_constraints_sensibility_analysis(constraint_name, shadow_price, sources):
    """print conclusions of the constraints sensibility analysis"""
    # Find the constraint number and the constraint location using the constraint name
    constraint_number = int(str(constraint_name)[2:3])
    location = str(constraint_name)[-3:]
    # Round to 2 decimals
    shadow_price = round(shadow_price, 2)
    if constraint_number <= 2 and location in sources:
        if shadow_price < 0:
            print("The total transportation cost would be reduced by",
                  abs(shadow_price),
                  "euros for each additional ton available in", location)
        elif shadow_price > 0:
            print("The total transportation cost would be increased in",
                  shadow_price,
                  "euros for each additional ton available in", location)
        else:
            print("The total transportation cost would remain equal for each additional ton available in", location)
    elif constraint_number <= 2:
        if shadow_price < 0:
            print("The total transportation cost would be reduced by",
                  abs(shadow_price),
                  "euros for each additional ton supply at", location)
        elif shadow_price > 0:
            print("The total transportation cost would be increased in",
                  shadow_price,
                  "euros for each additional ton supply at", location)
        else:
            print("The total transportation cost would remain equal for each additional ton supply at", location)


def print_conclusions_variables_sensibility_analysis(variable_name, reduced_cost):
    """print conclusions of the variables sensibility analysis"""
    # Find the source and the customer using the variable name
    source = variable_name[-7:-4]
    customer = variable_name[-3:]
    # Round to 2 decimals
    reduced_cost = round(reduced_cost, 2)
    if reduced_cost < 0:
        print("The total transportation cost would be reduced by",
              abs(reduced_cost),
              "euros for each ton supply from", source, "to", customer)
    elif reduced_cost > 0:
        print("The total cost would be increased in",
              reduced_cost,
              "euros for each ton supply from", source, "to", customer)
    else:
        print("The total transportation cost would remain equal for each ton supply from", source, "to", customer)


# Solve some transportation problems

# Base case
solve_problem_using_ortools_linear_solver("./data/data_0.json")

# Sensibility analysis - sources
# Using the base case, we move one ton of supply capacity from Gou to Arn and
# the objetive function improves in 0.2 euros (shadow price for Arn in the base case)
# solve_problem_using_ortools_linear_solver("./data/data_1.json")

# Sensibility analysis - customers - 1
# Using the base case, we increase the demand in Lon in one ton, and
# we increase one ton of supply capacity in Gou (Gou is the only source for Lon).
# Then the objetive function gets worse in 2.5 euros (shadow price for Lon in the base case)
# solve_problem_using_ortools_linear_solver("./data/data_2.json")

# Sensibility analysis - customers - 2
# Using the base case, we increase the demand in Ber in one ton, and
# we increase one ton of supply capacity in Gou (Arn is the only source for Ber).
# Then the objetive function gets worse in 2.7 euros (shadow price for Ber in the base case)
# solve_problem_using_ortools_linear_solver("./data/data_3.json")

# Sensibility analysis - routes
# Using the base case, we fixed a transportation between Arn and Ams equal to 1 ton,
# using pFixedTransportation and c03_fixed_%s_%s.
# The objetive function gets worse in 0.6 euros (reduced cost for the transportation between Arn and Ams)
# solve_problem_using_ortools_linear_solver("./data/data_4.json")
