// Adding Nodes and Connections

CREATE (macro1: Macro {name:'macro1' , type:'rectangle', x1: 0.1, y1: 0.2, x2: 0.4, y2: 0.6})
CREATE (macro2: Macro {name:'macro2' , type:'rectangle', x1: 0.8, y1: 0.8, x2: 0.9, y2: 0.9})
CREATE (macro3: Macro {name:'macro3' , type:'rectangle', x1: 0.7, y1: 0.1, x2: 0.9, y2: 0.3})
CREATE (macro4: Macro {name:'macro4' , type:'rectangle', x1: 0.6, y1: 0.5, x2: 0.7, y2: 0.6})

CREATE (macro1)-[w1:IS_CONNECTED]->(macro2)
CREATE (macro2)-[w2:IS_CONNECTED]->(macro3)
CREATE (macro2)-[w3:IS_CONNECTED]->(macro4)


// Projecting graph to GDS


CALL gds.graph.project(
  'macros',
  {
    Macro: {
      properties: ['x1', 'x2','y1','y2']
    }
  }, {
    IS_CONNECTED: {
      orientation: 'UNDIRECTED'
    }
})

// Training Embedding

CALL gds.beta.graphSage.train(
  'macros',
  {
    modelName: 'multiLabelModel',
    featureProperties: ['x1', 'x2', 'y1' , 'y2'],
    projectedFeatureDimension: 4,
    epochs: 10
  }
)


// Display Embedding


CALL gds.beta.graphSage.stream(
  'macros',
  {
    modelName: 'multiLabelModel'
  }
)
YIELD nodeId, embedding
