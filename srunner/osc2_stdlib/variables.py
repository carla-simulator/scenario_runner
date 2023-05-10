class Variable:
    args = dict()

    @classmethod
    def set_arg(cls, kw):
        cls.args.update(kw)

    @classmethod
    def get_arg(cls, key):
        return cls.args.get(key)
