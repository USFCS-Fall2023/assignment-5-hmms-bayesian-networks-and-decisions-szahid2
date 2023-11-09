# Sobia Zahid
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD


car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition", "Starts"),
        ("Gas", "Starts"),
        ("Starts", "Moves")
    ]
)

# Defining the parameters using CPT

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery": ['Works', "Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas": ['Full', "Empty"]},
)

cpd_radio = TabularCPD(
    variable="Radio", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works', "Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable="Ignition", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works', "Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts": ['yes', 'no'], "Ignition": ["Works", "Doesn't work"], "Gas": ['Full', "Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01], [0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no']}
)

# Associating the parameters with the model structure
car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

car_infer = VariableElimination(car_model)
print(car_infer.query(variables=["Moves"], evidence={"Radio": "turns on", "Starts": "yes"}))
q1 = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
print("Query 1: P(Battery='Doesn't work' | Moves='no') =")
print(q1)
q2 = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
print("\nQuery 2: P(Starts='no' | Radio='Doesn't turn on') =")
print(q2)
q3a = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
q3b = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
print("\nQuery 3a: P(Radio='turns on' | Battery='Works') =")
print(q3a)
print("\nQuery 3b: P(Radio='turns on' | Battery='Works', Gas='Full') =")
print(q3b)
q4a = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
q4b = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
print("\nQuery 4a: P(Ignition='Doesn't work' | Moves='no') =")
print(q4a)
print("\nQuery 4b: P(Ignition='Doesn't work' | Moves='no', Gas='Empty') =")
print(q4b)
q5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
print("\nQuery 5: P(Starts='yes' | Radio='turns on', Gas='Full') =")
print(q5)
