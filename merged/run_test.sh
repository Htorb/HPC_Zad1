#!/bin/bash
BRed='\033[1;31m'         # Red
BGreen='\033[1;32m'       # Green
Off='\033[0m'             # Text Reset

make clean
make

echo "----NORMAL----"
for TEST_PATH in ../test/*.in;  do
    NAME=${TEST_PATH%.*}
    OUT=$NAME.out
    M_OUT=$NAME.m_out
    
    ./gpulouvain -f $TEST_PATH -g 0.01 2> /dev/null > $M_OUT 
    
	
    OUT_VAL=$(head -n 1 $OUT) 
    M_OUT_VAL=$(head -n 1 $M_OUT)
    M_OUT_TIME=$(head -n 2 $M_OUT | tail -1)  



    DIFF=$(echo "$OUT_VAL - $M_OUT_VAL" | bc)
    if (( $(echo "0 > $DIFF" |bc -l) ));
    then
	DIFF=$(echo "$DIFF * -1" | bc) 
    fi

    if (( $(echo "0.001 > $DIFF" |bc -l) )); 
    then
        echo -e "$Off $BGreen Test passed:  $NAME $Off"
	echo -e "	execution time: $M_OUT_TIME"
    else
        echo -e "$Off $BRed Test failed:  $NAME $Off"

	echo "expected: $OUT_VAL"
	echo "got: $M_OUT_VAL" 

    fi
done

echo "----DEBUG----"

for TEST_PATH in ../test/*.in;  do
    NAME=${TEST_PATH%.*}
    OUT=$NAME.out
    M_OUT=$NAME.m_out
    
    ./gpulouvain -f $TEST_PATH -g 0.01 -d 2> /dev/null > $M_OUT 
    
	
    OUT_VAL=$(head -n 1 $OUT) 
    M_OUT_VAL=$(head -n 1 $M_OUT)

    DIFF=$(echo "$OUT_VAL - $M_OUT_VAL" | bc)
    if (( $(echo "0 > $DIFF" |bc -l) ));
    then
	DIFF=$(echo "$DIFF * -1" | bc) 
    fi

    if (( $(echo "0.001 > $DIFF" |bc -l) )); 
    then
        echo -e "$Off $BGreen Test passed:  $NAME $Off"
    else
        echo -e "$Off $BRed Test failed:  $NAME $Off"

	echo "expected: $OUT_VAL"
	echo "got: $M_OUT_VAL" 

    fi
done

