# Learn2Climb Overview

In rock climbing, routes are given difficulty scores that represent how hard it is for a climber to get to the top. The difficulty score assigned to any given route assumes that the climber chooses the optimal sequence. However, the optimal sequence is not provided apriori. A climber could always choose, either deliberately or accidentally, to perform a suboptimal sequence (maybe they though jumping past holds would be more fun!). This renders the problem of accurately predicting the difficulty of climbing routes challenging; a dataset of climbs associated with a difficulty score that does not have an associated sequence is, in some ways, incomplete.

[Previous efforts](https://arxiv.org/pdf/2102.01788) to predict the difficulty of climbs from the sequence of moves do so by injecting human intuition with data preprocessing. They devise an algorithm, without any machine learning, to determine a plausible sequence using entirely human inputs. However, this methodology leads to predictions ultimately limited by the quality of the injected intuition. If determining the optimal sequence were easy, then this would be a non-issue. But finding the optimal sequence for climbs can be tricky, even for more experienced climbers, let alone distilling that knowledge into a deterministic formula. Thus, finding a way to optimize the sequence in a way not limited by the quality of the injected intuition stands to both be an interesting problem and enable more accurate predictions of climbs' difficulties.

Here, I show that using a dataset containing only route and their difficulty scores, without an optimal sequence, it is possible to simulataneously extract reasonable optimal sequences and use them to predict the difficulty of climbs. Furthermore, the accuracy of the predictions of grades using this method outperform the reported prediction accuracy of [other](https://arxiv.org/pdf/2102.01788) [efforts](https://cs229.stanford.edu/proj2017/final-reports/5232206.pdf) [published](https://arxiv.org/pdf/2311.12419) [online](https://github.com/andrew-houghton/moon-board-climbing).

# Methods

Each climbing route, called a 'problem', is defined by 1 or 2 starting holds, 1 or 2 finishing holds, and an arbitrary number of holds in between. The climber must start with two hands on the starting holds and work their way to the finish hold. 

I approach the problem by defining a Graph Neural Network (GNN), where the individual problems are represented as graphs. Each unique combination of left hand and right hand positions defines a node; the node values for any given problem are static. The edge values on the graph represent the difficulty of moving from one position, or node, to the next. The only human intuition injected into the algorithm is that a move consists of moving one hand at a time: at any step, the climber can move either their left or right hand, connecting them to a new position. The difficulty of each move is not known and no human intuition is provided in its calculation.

A move is characterized by an 18 element vector. The position of the climber's lefthand, $\vec{LH} = (x, y)$, is characterized by a two element tuple with first value correspond to its $x$ position and second its $y$ position. The position of their right hand is given by a similar vector $\vec{RH}$. A move is defined by a climber changing positions, so they have a $\vec{LH_1} \rightarrow \vec{LH_2}$ and $\vec{RH_1} \rightarrow \vec{RH_2}$. The first 8 elements of the vector are given by $\vec{LH_2} - \vec{LH_1}$, $\vec{RH_2} - \vec{RH_1}$, $\vec{LH_1} - \vec{RH_1}$, $\vec{LH_2} - \vec{RH_2}$. Each hold on the wall is index by its position and characterized by a two element tuple, the first value refering to a difficulty score and second to the hold's angle of rotation. The next 8 elements in the vector representing a move correspond to the hold characteristics of the holds found at the positions of $\vec{LH_1}$, $\vec{LH_2}$, $\vec{RH_1}$, and $\vec{RH_2}$. The last two elements of the move vector are either 0 or 1. The penultimate element is 0 if $\vec{LH_1}\neq \vec{RH_1}$ and 1 if $\vec{LH_1} = \vec{RH_1}$. The last element is 0 if $\vec{LH_2}\neq \vec{RH_2}$ and 1 if $\vec{LH_2} = \vec{RH_2}$. The 18 element vector is indepent of the absolute position of the climber. It only has feature detailing the change in position and the quality of the holds used, ensuring the the algorithm learns generalizable intuition for what renders a move difficult, rather than learning ad hoc that moves in certain absolute positions are more or less difficult.

The difficulty of a move is calculated from the 18 element move vector with a feedforward NN. The model for the difficulty of a move is **initialized randomly** and learned with the following procedure:

1. Use the model to calculate the edge values on all the graphs. 
2. Determine the optimal sequence using the Dijkstra algorithm
3. Train 

6. Repeat steps 1-5 until convergence is observed. 

Note, that because the model is initialized randomly, the initially determined optimal sequences are also random, and therefore completely incorrect. Since the optimal sequence gets update as the model for the difficulty of a move gets trained, it learns the most consistent rules for optimal sequences that enables the grades of teh climbs to be consistent.


The optimal sequence is then determined using the Dijkstra algorithm. The difficulty of a climb is given by the sum of the difficulty of the moves. 

In climbing, there is a heuristic formula for determining the grade of a climb. If you chain two climbs of the same difficulty back to back, the combination is given the average grade +2. Thus, the difficulty of a climb can be represented as an exponential function.


L2 loss function. Weight. 

# Data Used

Limit myself to moonboard 2016 climbs. Taken from these fool's github page.

Augment the data by reflecting each problem about the central axis of the board. 


# Future Directions


The algorithm right now does not factor in the position of your feet when doing a move. This would require

More data, by combining climbs from more than just the 2016 set. Could even go to other board systems.
