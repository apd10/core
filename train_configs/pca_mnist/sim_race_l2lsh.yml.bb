module: "RaceSketch"
data:
  file: "/home/apd10/experiments/projects/summary_based_learning/DATA/pca_mnist/train_data.txt"
  dataset: "csv"
  csv:
    sep: " "
    label_header: 0
  sampler: "simple"
  simple:
    batch_size: 100
save_sketch: "/home/apd10/experiments/projects/summary_based_learning/DATA/pca_mnist/race_train_l2lsh.pickle"
np_seed: 111
race:
  range: 4294967295  # (2*max_norm / bandwidth + 1)*power
  repetitions: 1
  power: 7
  num_classes: 1
  rehash: False
  max_coord: 36
  min_coord: -32
  lsh_function:
    name: "l2lsh"
    l2lsh:
      bandwidth: 4
      dimension: 15
      max_norm: 40
