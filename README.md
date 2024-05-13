# Micron & IIITB : Layout automation tool development for Macro Place and Route.

This is the github repository for the macro place and route tool.

## Installation steps

Clone this repository

```
git clone https://github.com/hypnotic2402/layoutOptimMicron.git
cd layoutOptimMicron
```

Python Library Installation :

```
pip3 install numpy
pip3 install pandas
pip3 install matplotlib
pip3 install neo4j

```

Install Neo4J Desktop by following its documentation :

https://neo4j.com/download/

## How to run :

Start the Neo4j server on your localhost (or at any external host).

Push the netlist on to the Neo4J database with the nodes (macros) having the following properties: w (width) , h(height).

----Text about parser----

Make sure the nodes are pushed to the current active database. Then refer to main.py

Create pins for each of the macros and specify their relative positions on the macro. - Pin(x,  y)
Extract the macros from the current active database by running the extr(uri, auth) function in extract.py. Then assign the pins to the respective macros. Define the dimensions of the floor and the minimum divisible unit of length l by Floor(x,y,l). 

Create a framework class Framework(macros, nets, FL).

Place the macros using the place function in placement2 : place(iter , genVid , filename)
    The number of iterations are given by iter. Setting genvid to True will enable generation of a video for placement using OpenCV. 

Route the pins using route(disp). Setting disp to true will display the layers of routed wires.

## Example

Example graph in neo4j:

run main.py





