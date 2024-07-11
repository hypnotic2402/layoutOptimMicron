# Micron & IIITB : Layout automation tool development for Macro Place and Route.
This is the GitHub repository for the macro place and route tool. The `maskplace` directory contains the code for the RL placement tool called maskplace, while the `Global_Routing_RL` directory houses the code for the RL routing tool.

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
pip3 install networkx
pip3 install pymoo
pip3 install opencv-python
sudo apt-get install python3-opencv

```

Install Neo4J Desktop by following its documentation :

https://neo4j.com/download/

## How to run :

Start the Neo4j server on your localhost (or at any external host).

Push the netlist on to the Neo4J database with the nodes (macros) having the following properties: w (width) , h(height).

### Parser

To run the parser,
Run the Script.ipynb file in the scripts folder this would randomly choose the given number of 
Macros from the design_example folder as specified by the user and connect them randomly to formed a
closed and fully connected graph.

Once this graph with its nodes and edge properties is created this can be pushed to the Neo4J
database, from here the nodes (MACROS) can be extracted for the PnR algorithm 

### Placement and Routing
 
Make sure the nodes are pushed to the current active database. Then refer to main.py

Create pins for each of the macros and specify their relative positions on the macro. - Pin(x,  y)
Extract the macros from the current active database by running the extr(uri, auth) function in extract.py. Then assign the pins to the respective macros. Define the dimensions of the floor and the minimum divisible unit of length l by Floor(x,y,l). 

Create a framework class Framework(macros, nets, FL).

Place the macros using the place function in placement2 : place(iter , genVid , filename)
    The number of iterations are given by iter. Setting genvid to True will enable generation of a video for placement using OpenCV. 

Route the pins using route(disp). Setting disp to true will display the layers of routed wires.

## Example

Example graph in neo4j:
![pic1](https://github.com/hypnotic2402/layoutOptimMicron/assets/75616591/fd98a9e5-5b64-4c8b-be71-6a44ff12a1c3)

run main.py
![pic2](https://github.com/hypnotic2402/layoutOptimMicron/assets/75616591/96007a7f-4c22-4768-9816-5b0154c1edf1)

![pic3](https://github.com/hypnotic2402/layoutOptimMicron/assets/75616591/d92e5fda-4e5f-4702-8ddb-d575cff690ba)

![pic4](https://github.com/hypnotic2402/layoutOptimMicron/assets/75616591/b39bd85b-9714-40ab-a506-c0d26e4d2fe9)




