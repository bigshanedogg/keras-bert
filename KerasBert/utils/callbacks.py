import os
import pickle
import keras
import keras.backend.tesnorflow_backend as K
from .common import save_model, load_model, save_pickle, load_pickle, get_trained_batch

class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, model_dir, save_per_step, save_history=False, verbose=2):
        self.model_dir = model_dir
        if not model_dir.endswith("/"): self.model_dir = self.model_dir + "/"
        self.save_per_step = save_per_step
        self.save_history = save_history
        self.verbose = verbose
        self.history = dict()
        self.history_filename = "history.pickle"
        self.batch_verbose_statement = None
        self.epoch_verbose_statement = None
        if self.save_history:
            self.batch_verbose_statement = "\nBatch {batch:05d}: SaveModelCallback saved model and history into '{model_dir}'"
            self.epoch_verbose_statement = "\nEpoch {epoch:05d}: SaveModelCallback saved model and history into '{model_dir}'"
        else:
            self.batch_verbose_statement = "\nBatch {batch:05d}: SaveModelCallback saved model into '{model_dir}'"
            self.epoch_verbose_statement = "\nEpoch {epoch:05d}: SaveModelCallback saved model into '{model_dir}'"

    def on_batch_end(self, batch, logs):
        for k in logs.keys():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(logs[k])

        if batch % self.save_per_step == 0:
            # save model
            save_model(self.model_dir, self.model)
            #  save history
            if self.save_history: self.save_history()

        if batch % self.save_per_step == 0 and self.verbose > 1:
            print(self.batch_verbose_statement.format(batch=batch, model_dir=self.model_dir))

    def on_epoch_end(self, epoch, logs):
        for k in logs.keys():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(logs[k])

        # save_model
        save_model(self.model_dir, self.model)
        # save_history
        if self.save_history: self.save_history()

        if self.verbose > 0:
            print(self.epoch_verbose_statement.format(epoch=epoch, model_dir=self.model_dir))

    def _save_history(self):
        history = None
        history_dir = self.model_dir + self.history_filename
        if os.path.isfile(history_dir):
            history = load_pickle(history_dir)
        if history is not None:
            for k in self.history.keys():
                if k not in history: continue
                history[k] = self.history[k]
        else:
            history = self.history
        save_pickle(path=history_dir, data=history)
        self.history = dict()


class TransformerLrSchedulerCallback(keras.callbacks.Callback):
    def __init__(self, initial_lr, warmup_steps, d_model, trained_batch=0, verbose=2, verbose_per_batch=5000):
        self.initial_lr = self.lr = initial_lr
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.trained_batch = trained_batch
        self.verbose = verbose
        self.verbose_per_batch = verbose_per_batch
        self.batch_verbose_statement = "\nBatch {batch:05d}: TransformerLrSchedulerCallback set learning rate to '{lr}'"
        self.epoch_verbose_statement = "\nEpoch {epoch:05d}: TransformerLrSchedulerCallback set learning rate to '{lr}'"

    def on_batch_end(self, batch, logs):
        self.batch += 1
        self.lr = self.transformer_learning_rate(step=self.batch, d_model=self.d_model, warmup_steps=self.warmup_steps)
        K.set_value(self.model.optimizer.lr, self.lr)

        if self.batch % self.verbose_per_batch == 0 and self.verbose > 1:
            print(self.batch_verbose_statement.format(batch=self.batch, lr=self.lr))

    def on_epoch_end(self, epoch, logs):
        if self.verbose > 0:
            print(self.epoch_verbose_statement.format(epoch=epoch, lr=self.lr))


    def transformer_learning_rate(self, step, d_model, warmup_steps):
        arg1 = (step + 1) ** -0.5
        arg2 = (step + 1) * (warmup_steps ** -1.5)
        lr = (d_model ** -0.5) * min(arg1, arg2)
        return lr
