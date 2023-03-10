class Singleton(type):
    def __init__(cls, name, bases, dic) -> None:
        cls._instance = None
        super().__init__(name, bases, dic)
        
    def __call__(cls, *args, **kwargs) -> object:
        if cls._instance is not None:
            return cls._instance
        obj = cls.__new__(cls)
        obj.__init__(*args, **kwargs)
        cls._instance = obj
        return cls._instance