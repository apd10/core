for rep in 5;
do
  for power in 784
  do
    for bw in 0.01;
      do 
        for heap in 5000;
        do
          cat sample.yml | sed "s/POWER/$power/g" | sed "s/REP/$rep/g" | sed "s/BWIDTH/$bw/g" | sed "s/HEAP/$heap/g" > sample.$rep.$power.$bw.$heap.yml; 
          echo "taskset -c 45-88  /home/apd10/experiments/projects/summary_based_learning/run.py --config $PWD/sample.$rep.$power.$bw.$heap.yml"
          cat racetrain.yml | sed "s/POWER/$power/g" | sed "s/REP/$rep/g" | sed "s/BWIDTH/$bw/g" | sed "s/HEAP/$heap/g" > racetrain.$rep.$power.$bw.$heap.yml;
          echo "taskset -c 45-88  /home/apd10/experiments/projects/summary_based_learning/run.py --config $PWD/racetrain.$rep.$power.$bw.$heap.yml"
        done
      done
  done
done
      
