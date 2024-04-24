import logging


__all__ = [
    "ConvergenceChecker",
]

log = logging.getLogger(__name__)


class ConvergenceChecker:
    def __init__(self, cfg):
        """
        :param cfg: config contains
        max_iterations: if <= 0 convergence criteria is not applied
        tolerance:
        """
        self.check_flag = cfg["max_iterations"] > 0
        if self.check_flag:
            self.tolerance = cfg["tolerance"]
            self.max_iterations = cfg["max_iterations"]
        else:
            self.tolerance = 0
            self.max_iterations = 0

        self.last_loss = None
        self.iter_counter = 0

    def reset(self):
        self.last_loss = None
        self.iter_counter = 0

    def check(self, loss):
        if not self.check_flag:
            # it means it never converges with check_flag - false
            return False

        if self.last_loss is None:
            self.last_loss = loss
            return False

        if abs(self.last_loss - loss) <= self.tolerance:
            self.iter_counter += 1
        else:
            self.iter_counter = 0

        # return result of convergence criteria
        return self.iter_counter >= self.max_iterations
