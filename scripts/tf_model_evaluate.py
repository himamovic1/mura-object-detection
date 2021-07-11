"""Evaluates TF2 object detection models."""
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2

from config.config import Config


def main(_):
    """
    Define arguments for training the model.

    - model_dir:                            Path to output model dir. where event and checkpoint files will be written

    - checkpoint_dir:                       Path to dir. holding a checkpoint.  If `checkpoint_dir` is provided, this
                                            binary operates in eval-only mode, writing resulting metrics to `model_dir`.

    - pipeline_config_path:                 Path to models config file

    - num_train_steps:                      Num of steps in the training (default None; config value is used)

    - eval_timeout:                         Num of seconds to wait for an eval. checkpoint before exiting (default 3600)

    - eval_on_train_data:                   Enable eval. on train data (default False; only in distributed training)

    - sample_1_of_n_eval_examples:          Will sample one of every n eval input examples (default None)

    - sample_1_of_n_eval_on_train_examples: Will sample one of every n train input examples for evaluation, where n
                                            is provided. This is only used if `eval_training_data` is True. (default 5)

    - use_tpu:                              Whether the job is executing on a TPU (default False)

    - tpu_name:                             Name of the Cloud TPU for Cluster Resolvers (default None)

    - num_workers:                          When num_workers > 1, training uses MultiWorkerMirroredStrategy.
                                            When num_workers = 1 it uses MirroredStrategy (default 1)

    - checkpoint_every_n:                   Integer defining how often we checkpoint (default 1000)

    - record_summaries:                     Whether or not to record summaries defined by the model or the training
                                            pipeline. This does not impact the summaries of the loss values which are
                                            always recorded. (default True)
    """
    tf.config.set_soft_device_placement(True)

    model_lib_v2.eval_continuously(
        pipeline_config_path=Config.TF_FLAG_PIPELINE_CONFIG_PATH,
        model_dir=Config.TF_FLAG_MODEL_DIR,
        train_steps=Config.TF_FLAG_NUM_TRAINING_STEPS,
        sample_1_of_n_eval_examples=Config.TF_FLAG_SAMPLE_1_OF_N_EVAL_EXAMPLES,
        sample_1_of_n_eval_on_train_examples=(Config.TF_FLAG_SAMPLE_1_OF_N_EVAL_ON_TRAIN_EXAMPLES),
        checkpoint_dir=Config.TF_FLAG_CHECKPOINT_DIR,
        wait_interval=Config.TF_FLAG_EVAL_INTERVAL,
        timeout=Config.TF_FLAG_EVAL_TIMEOUT,
    )


if __name__ == "__main__":
    tf.compat.v1.app.run()
