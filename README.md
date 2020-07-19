# MAL
Multi-Agent Learning Assignment

This Python program contains implementations of serval multi-agent learning algorithms and compares them with three methods. 
- Calculate the grand table of given matrix suite and strategies. 
- Perform the replictor dynamic on a grand table and display the graph of its processes.
- Find Nash equilibria of a grand table and display the Nash in terminal.

## How to use our code
The example code are all included in [Main.py](https://github.com/Meilin1996/MAL/blob/master/Main.py). To check all the results of this paper, simply run it. We will also give examples separately below. 

### To obtain a grand table:
```
grand_table = GrandTable(matrix_suite, strategies, k - 1, N)
grand_table.execute()
print(grand_table)
```
The grand table will be displayed on the terminal. 

### To use replicator dynamic and draw a graph of proportions:
```
replicator_dynamic = ReplicatorDynamic(proportions, grand_table)
replicator_dynamic.run()
```
This will save the figure of the replicator dynamic result in [img/](https://github.com/Meilin1996/MAL/tree/master/img) folder.

### To check the Nash equilibria of a grand table:
```
Nash.nash_equilibria(grand_table)
```
The Nash equilibria results will be displayed on the terminal. 
