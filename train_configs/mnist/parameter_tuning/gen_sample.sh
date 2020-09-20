for rep in 5 50;
do
  for power in 25 50
  do
    for bw in 0.4;
      do 
        for heap in 5000;
        do
          cat sample.yml | sed "s/POWER/$power/g" | sed "s/REP/$rep/g" | sed "s/BWIDTH/$bw/g" | sed "s/HEAP/$heap/g" > sample.$rep.$power.$bw.$heap.yml; 
          echo "taskset -c 45-88  /home/apd10/experiments/projects/summary_based_learning/run.py --config $PWD/sample.$rep.$power.$bw.$heap.yml"
        done
      done
  done
done
      
 

   
for rep in 5 50;
do
  for power in 10
  do
    for bw in 0.2;
      do 
        for heap in 5000;
        do
          cat sample.yml | sed "s/POWER/$power/g" | sed "s/REP/$rep/g" | sed "s/BWIDTH/$bw/g" | sed "s/HEAP/$heap/g" > sample.$rep.$power.$bw.$heap.yml; 
          echo "taskset -c 45-88  /home/apd10/experiments/projects/summary_based_learning/run.py --config $PWD/sample.$rep.$power.$bw.$heap.yml"
        done
      done
  done
done
      
    
