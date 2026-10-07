"""Release the WebShop-owned router before the Ray task exits."""


def train_and_close_environment(trainer,environment):
    try:
        trainer.init_workers()
        return trainer.fit()
    finally:
        environment.close()
