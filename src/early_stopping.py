import copy
from typing import Literal

from torch import nn


class EarlyStopper:
    """
    Saving the model and stop training when the metric does not improved.
    :ivar patience: How many epochs to wait without improved.
    :ivar verbose: If True, open logs.
    :ivar mode: 'max' means higher is better. 'min' means lower is better.
    :ivar delta: The minimum change to count as improvment.
    """

    def __init__(self, patience=8, verbose=True, mode: Literal['min', 'max'] = 'max', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.best = -float('inf') if mode == 'max' else float('inf')
        self.bad_count = 0
        self.best_state: dict | None = None

    def step(self, value: float, model: nn.Module) -> bool:
        """ Check if metric improved and save the model if it did.
            (檢查模型是否進步並儲存最佳模型)
        :param model: Model to save when it improves.
        :param value: Current value of the metric.
        :return: If True, means model should stop with training.
        """
        improved = \
            (value > self.best + self.delta) \
                if self.mode == 'max' \
                else (value < self.best - self.delta)

        if improved:
            self.best = value
            self.bad_count = 0
            if self.verbose:
                print(f'[EarlyStop] Save Model.')
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.bad_count += 1
            if self.verbose:
                print(f'[EarlyStop]: {self.bad_count} out of {self.patience}')
            return self.bad_count >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Restore the model to saved best state.
         (回復模型至出現最好數值的epochs)
        :param model: Model to load the saved weights into.
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
