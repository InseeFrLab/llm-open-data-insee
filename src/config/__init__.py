from configparser import ConfigParser, ExtendedInterpolation


class MyConfigParser(ConfigParser):
    def __init__(self, *args, **kwargs):
        super(MyConfigParser, self).__init__(*args, **kwargs)
        self.section = "DEFAULT"

    def __getitem__(self, key):
        return super(MyConfigParser, self).__getitem__(self.section)[self.optionxform(key)]

    def get(self, key, fallback):
        return super(MyConfigParser, self).get(self.section, key, fallback=fallback)


# Global config
config = MyConfigParser(interpolation=ExtendedInterpolation(), allow_no_value=True)
