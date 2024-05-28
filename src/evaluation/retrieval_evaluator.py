from evaluation import RetrievalConfiguration

class RetrievalEvaluator:
    @staticmethod
    def run(eval_configurations: list[RetrievalConfiguration]):
        for eval_configuration in eval_configurations:
            print(eval_configuration)
        