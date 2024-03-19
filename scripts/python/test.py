from ga2 import GA
from extract_from_db import extract
from ga import objF
from vis import genVid

def run_GA(n):
    nMacros , wh , nl = extract()
    ga = GA(1000 , 0.1 , 0.1 , objF , 2*nMacros , 0 , 800 , nl , wh)

    ga.opt(n)
    genVid(ga.xHis , wh)


if __name__ == '__main__':
    run_GA(1000)
