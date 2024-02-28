Here we solve the network flows optimization problem propused by AIMMS in https://download.aimms.com/aimms/download/manuals/AIMMS3_OM.pdf using OR-Tools linear solver.

In this problem, we have two sources, located in Arnhem and Gouda, and six customers, located in London, Berlin, Maastricht, Amsterdam, Utrecht and The Hague. For reasons of efficiency, deliveries abroad are only made by one source. So that, Arnhem delivers to Berlin and Gouda to London.

We can find the original limits of production/supply for each source, the demand for each customer and transportation costs between each source and customer in ./data/data_0.json. In this problem, the goal is to satisfy the customers’ demand while minimizing transportation costs.

In main.py, we can find the problem formulation, the way to solve it using ‘cbc’ solver and a way to print the solution and the sensibility analysis for this solution (shadow prices and reduced costs).

We have four data file in 'data' folder that we can use to try the code:
* data_0.json. This is the base case.
* data_1.json. Shadow price interpretation for source constraints. Using base case, we move one ton of supply capacity from Gouda to Arnhem and the objetive function improves in 0.2 euros (shadow price for Arnhem in the base case).
* data_2.json. Shadow price interpretation for customer constraints - case 1. Using the base case, we increase the demand in London in one ton and we increase one ton of supply capacity in Gouda (Gouda is the only source for London). Then the objetive function gets worse in 2.5 euros (shadow Price for London in the base case).
* data_3.json. Shadow price interpretation for customer constraints - case 2. Using the base case, we increase the demand in Berlin in one ton and we increase one ton of supply capacity in Gouda (Arnhem is the only source for Berlin). Then the objetive function gets worse in 2.7 euros (shadow price for Berlin in the base case).
* data_4.json. Reduced cost interpretation for not active routes. Using the base case, we fixed a transportation between Arnhem and Amsterdam equal to 1 ton using pFixedTransportation and c03_fixed_%s_%s. Then the objetive function gets worse in 0.6 euros (reduced cost for this transportation in the base case).