from evaluators.list_evaluators import *

class ProgressEvaluator:
    def get(params, train_data, test_data, device_id):
        if params["name"] == "simple_print":
            evaluator = SimplePrintEvaluator(params["simple_print"], train_data, test_data, device_id)
        elif params["name"] == "mean_loglikelihood":
            evaluator = SampleMeanLLEvaluator(params["mean_loglikelihood"], train_data, device_id)
        else:
            raise NotImplementedError
        return evaluator
