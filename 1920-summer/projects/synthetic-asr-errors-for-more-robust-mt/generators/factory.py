from generators.base_generator import SentenceGenerator


class GeneratorFactory:
    @staticmethod
    def get_generator_class(name, *argv):
        generators_names = [cls.__name__ for cls in SentenceGenerator.__subclasses__()]
        generator_classes = SentenceGenerator.__subclasses__()
        assert name in generators_names, "Generator '{}' not found. Available: {}".format(name, generators_names)
        i = generators_names.index(name)
        return generator_classes[i](*argv)
