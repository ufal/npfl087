import logging


class BasicLogger:
    logger_name = 'homo-noiser'

    @staticmethod
    def setupLogger(verbose):
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="\033[94m%(levelname)s:%(name)s:" \
                   "%(asctime)s\033[0m:  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        return logging.getLogger(BasicLogger.logger_name)
