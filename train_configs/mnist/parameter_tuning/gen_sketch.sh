for rep in 5;
do
  for power in 784
  do
    for bw in 0.01;
      do 
        for heap in 5000;
        do
          cat sketch.yml | sed "s/POWER/$power/g" | sed "s/REP/$rep/g" | sed "s/BWIDTH/$bw/g" | sed "s/HEAP/$heap/g" > sketch.$rep.$power.$bw.$heap.yml; 
          echo "/home/apd10/experiments/projects/summary_based_learning/run.py --config $PWD/sketch.$rep.$power.$bw.$heap.yml"
        done
      done
  done
done
      
    
