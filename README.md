# PageRank_Thrust

This is an implementation of Google's PageRank algorithm that processes and ranks the websites in the web graph provided by [web-graph.org](http://web-graph.org).

For the thrust implementation of the PageRank algorithm, we started with a sequential read of the edge list, assigned an int value to each node, and filled two host vectors, one for the outgoing and one for the incoming edges. 

Realising that our edge list also represents the list of coordinates where our matrix A is nonzero, we calculated the corresponding values vector. This was the first major insight that made our program easily parallelisable with thrust. 

Now, we were able to easily implement the vector-matrix multiplication (r\_next = Ar) by utilising a series of transformations and the permutation iterator provided by Thrust, which abstracts away how we would normally access random elements of r using array dereferencing. Calculating the sigma was made just as easy by using the built-in reduce operation. 

The most challanging part of the program, in our opinion, was writing the parallel code that calculates the five sites with the largest score in a parallelised way. We realised that sorting the whole array to just pick the top five sites would dramatically increase runtime, and therefore implemented an operator that merges two sorted arrays of five elements, and collects the top five elements of the two arrays into one sorted array. Using this operator, we were able to collect the top five elements in the whole dataset using the built-in reduce operator.

One critical consideration while creating the operator was avoiding using branches, as taking different branches in different threads comes with a big performance penalty in GPUs. We were able to achieve this by leveraging the inbuilt compare-and-swap facility of GPU by using the C ternary operator. Here is a pseudocode of our merging algorithm:

    site[5] merge_arrs(site[5] a, site[5] b)
    {
        site[5] result;

        size_t a_index = 0;
        size_t b_index = 0;
        for (size_t i = 0; i < 5; i++)
        {
            int is_b_larger = a[a_index].score < b[b_index].score;
            result[i] = is_b_larger ? b[b_index] : a[a_index];
            b_index += is_b_larger;
            a_index += (1 - is_b_larger);
        }
        return result;
    }


As the main point of improvement for our program, we could highlight our use of reduce-by-key for calculating the result vector. Even though we did not have enough time to devise an algorithm for multiplying a vector with a CSR matrix, we believe that its implementation would avoid the inefficiency of also calculating the unique values in our row vector each time the result is calculated.
