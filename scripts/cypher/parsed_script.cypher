CREATE (macro0: Macro {name:'X1' , x: 4660 , y: 4120})
CREATE (macro1: Macro {name:'X2' , x: 10160 , y: 4120})
CREATE (macro2: Macro {name:'X3' , x: 15820 , y: 7580})
CREATE (macro3: Macro {name:'X4' , x: 15860 , y: 1720})
CREATE (macro0)-[w0:IS_CONNECTED]->(macro1)
CREATE (macro1)-[w1:IS_CONNECTED]->(macro2)
CREATE (macro1)-[w2:IS_CONNECTED]->(macro3)
CREATE (macro2)-[w3:IS_CONNECTED]->(macro3)